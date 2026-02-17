# Build survey/data.js from AI, DE, ML Skills CSVs. Run from ai-learning-path: .\scripts\Build-SurveyData.ps1
# Optional: .\scripts\Build-SurveyData.ps1 -AiPath $env:TEMP\ai_skills.csv -DePath $env:TEMP\de_skills.csv -MlPath $env:TEMP\ml_skills.csv
param(
  [string]$AiPath = '',
  [string]$DePath = '',
  [string]$MlPath = ''
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
if (-not $AiPath) { $AiPath = Join-Path $Root 'AI Skills.csv' }
if (-not $DePath) { $DePath = Join-Path $Root 'DE Skills.csv' }
if (-not $MlPath) { $MlPath = Join-Path $Root 'ML Skills.csv' }
$outPath = Join-Path $Root 'survey\data.js'

function Parse-CsvLine {
  param([string]$line)
  $out = @()
  $cur = ''
  $inQ = $false
  for ($i = 0; $i -lt $line.Length; $i++) {
    $c = $line[$i]
    if ($c -eq '"') { $inQ = -not $inQ }
    elseif (-not $inQ -and $c -eq ',') { $out += $cur.Trim(); $cur = '' }
    else { $cur += $c }
  }
  $out + $cur.Trim()
}

function Split-CsvRows {
  param([string]$content)
  $lines = $content -replace "`r`n", "`n" -replace "`r", "`n" -split "`n"
  $merged = @()
  $i = 0
  while ($i -lt $lines.Count) {
    $row = $lines[$i]
    while (([regex]::Matches($row, '"').Count % 2) -ne 0 -and ($i + 1) -lt $lines.Count) {
      $i++; $row += "`n" + $lines[$i]
    }
    $merged += $row
    $i++
  }
  $merged
}

$LEVELS = @('Beginner','Novice','Competent','Proficient','Expert')

function Parse-AI {
  $content = try { Read-FileShared $AiPath } catch { [System.IO.File]::ReadAllText($AiPath).TrimStart([char]0xFEFF) }
  $lines = Split-CsvRows $content
  $headerIdx = -1
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^Subdomain,Skill,Level,Description') { $headerIdx = $i; break }
  }
  if ($headerIdx -lt 0) { return @() }
  $skills = @()
  $curSub = ''; $curSkill = ''; $levelDescs = @{}
  foreach ($row in $lines[($headerIdx+1)..($lines.Count-1)]) {
    $r = Parse-CsvLine $row
    $sub = if ($r.Count -gt 0 -and $null -ne $r[0]) { $r[0].Trim() } else { '' }
    $skill = if ($r.Count -gt 1 -and $null -ne $r[1]) { $r[1].Trim() } else { '' }
    $level = if ($r.Count -gt 2 -and $null -ne $r[2]) { $r[2].Trim() } else { '' }
    $desc = if ($r.Count -gt 3 -and $null -ne $r[3]) { $r[3].Trim() } else { '' }
    if ($sub -or $skill) {
      if ($curSkill -and $levelDescs.Count -gt 0) {
        $ordered = @()
        foreach ($L in $LEVELS) {
          $ordered += if ($levelDescs[$L]) { $levelDescs[$L] } else { '' }
        }
        $skills += @{ subdomain = $curSub; skill = $curSkill; levels = $ordered }
      }
      $curSub = if ($sub) { $sub } else { $curSub }
      $curSkill = if ($skill) { $skill } else { $curSkill }
      $levelDescs = @{}
    }
    if ($LEVELS -contains $level) { $levelDescs[$level] = $desc }
  }
  if ($curSkill -and $levelDescs.Count -gt 0) {
    $ordered = @()
    foreach ($L in $LEVELS) {
      $ordered += if ($levelDescs[$L]) { $levelDescs[$L] } else { '' }
    }
    $skills += @{ subdomain = $curSub; skill = $curSkill; levels = $ordered }
  }
  $skills
}

function Read-FileShared {
  param([string]$path)
  $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
  try {
    $sr = New-Object System.IO.StreamReader($fs, [System.Text.UTF8Encoding]::new($false))
    $sr.ReadToEnd().TrimStart([char]0xFEFF)
  } finally { $fs.Close() }
}

function Parse-Horizontal {
  param([string]$path)
  $content = Read-FileShared $path
  $lines = Split-CsvRows $content
  $headerIdx = -1
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match 'Subdomain.*Skill.*Beginner.*Novice') { $headerIdx = $i; break }
  }
  if ($headerIdx -lt 0) { return @() }
  $skills = @()
  foreach ($row in $lines[($headerIdx+1)..($lines.Count-1)]) {
    $r = Parse-CsvLine $row
    if ($r.Count -lt 7) { continue }
    $subdomain = if ($null -ne $r[0]) { $r[0].Trim() } else { '' }
    $skill = if ($null -ne $r[1]) { $r[1].Trim() } else { '' }
    if (-not $skill) { continue }
    $levels = @()
    for ($i = 2; $i -le 6; $i++) {
      $levels += if ($r.Count -gt $i -and $null -ne $r[$i]) { $r[$i].Trim() } else { '' }
    }
    $skills += @{ subdomain = $subdomain; skill = $skill; levels = $levels }
  }
  $skills
}

$ai = Parse-AI
$de = Parse-Horizontal $DePath
$ml = Parse-Horizontal $MlPath

# Convert to JSON-serializable structure (PSCustomObject)
$toObj = {
  param($arr)
  $arr | ForEach-Object {
    [PSCustomObject]@{ subdomain = $_.subdomain; skill = $_.skill; levels = @($_.levels) }
  }
}
$data = @{
  de = & $toObj $de
  ml = & $toObj $ml
  ai = & $toObj $ai
}
$json = $data | ConvertTo-Json -Depth 10 -Compress:$false
# Fix array formatting for levels
$json = $json -replace '"levels":\s*"([^"]*)"', '"levels": [$1]'
# Actually ConvertTo-Json serializes string[] as "a","b" - so levels is already array in JSON. Check:
$js = "window.DEMLAI_SURVEY_DATA = $json;`n"
$outDir = Split-Path $outPath
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
[System.IO.File]::WriteAllText($outPath, $js, [System.Text.UTF8Encoding]::new($false))
Write-Host "Wrote $outPath | DE: $($de.Count) ML: $($ml.Count) AI: $($ai.Count)"
