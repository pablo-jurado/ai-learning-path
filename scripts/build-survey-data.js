/**
 * Build survey/data.js from AI Skills.csv, DE Skills.csv, ML Skills.csv.
 * Run from repo root: node scripts/build-survey-data.js
 * (or from ai-learning-path: node ../scripts/build-survey-data.js if paths are adjusted)
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const LEVELS = ['Beginner', 'Novice', 'Competent', 'Proficient', 'Expert'];

function parseCSVLine(line) {
  const out = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
    } else if (!inQuotes && c === ',') {
      out.push(cur.trim());
      cur = '';
    } else {
      cur += c;
    }
  }
  out.push(cur.trim());
  return out;
}

function parseAIVertical(content) {
  const lines = splitCSVRows(content);
  const skills = [];
  let headerIndex = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('Subdomain,Skill,Level,Description')) {
      headerIndex = i;
      break;
    }
  }
  if (headerIndex < 0) return skills;
  let currentSubdomain = '';
  let currentSkill = '';
  let levelDescs = [];
  for (let i = headerIndex + 1; i < lines.length; i++) {
    const row = parseCSVLine(lines[i]);
    const sub = (row[0] || '').trim();
    const skill = (row[1] || '').trim();
    const level = (row[2] || '').trim();
    const desc = (row[3] || '').trim();
    if (sub || skill) {
      if (currentSkill && levelDescs.length > 0) {
        const ordered = [];
        for (const l of LEVELS) {
          const idx = levelDescs.findIndex((x) => x.level === l);
          ordered.push(idx >= 0 ? levelDescs[idx].desc : '');
        }
        skills.push({ subdomain: currentSubdomain, skill: currentSkill, levels: ordered });
      }
      currentSubdomain = sub || currentSubdomain;
      currentSkill = skill || currentSkill;
      levelDescs = [];
    }
    if (level && LEVELS.includes(level)) {
      levelDescs.push({ level, desc });
    }
  }
  if (currentSkill && levelDescs.length > 0) {
    const ordered = [];
    for (const l of LEVELS) {
      const idx = levelDescs.findIndex((x) => x.level === l);
      ordered.push(idx >= 0 ? levelDescs[idx].desc : '');
    }
    skills.push({ subdomain: currentSubdomain, skill: currentSkill, levels: ordered });
  }
  return skills;
}

function splitCSVRows(content) {
  const raw = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const lines = [];
  let cur = '';
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (c === '\n') {
      lines.push(cur);
      cur = '';
    } else {
      cur += c;
    }
  }
  if (cur) lines.push(cur);
  const merged = [];
  let i = 0;
  while (i < lines.length) {
    let row = lines[i];
    while ((row.match(/"/g) || []).length % 2 !== 0 && i + 1 < lines.length) {
      i++;
      row += '\n' + lines[i];
    }
    merged.push(row);
    i++;
  }
  return merged;
}

function parseHorizontal(content, headerText) {
  const lines = splitCSVRows(content);
  const skills = [];
  let headerIndex = -1;
  for (let i = 0; i < lines.length; i++) {
    const normalized = lines[i].replace(/\s+/g, ' ').trim();
    if (normalized.startsWith('Subdomain,Skill,Beginner,Novice,Competent,Proficient,Expert')) {
      headerIndex = i;
      break;
    }
  }
  if (headerIndex < 0) return skills;
  for (let i = headerIndex + 1; i < lines.length; i++) {
    const row = parseCSVLine(lines[i]);
    if (row.length < 7) continue;
    const subdomain = (row[0] || '').trim();
    const skill = (row[1] || '').trim();
    if (!skill) continue;
    const levels = [
      (row[2] || '').trim(),
      (row[3] || '').trim(),
      (row[4] || '').trim(),
      (row[5] || '').trim(),
      (row[6] || '').trim(),
    ];
    skills.push({ subdomain, skill, levels });
  }
  return skills;
}

function main() {
  const aiPath = path.join(ROOT, 'AI Skills.csv');
  const dePath = path.join(ROOT, 'DE Skills.csv');
  const mlPath = path.join(ROOT, 'ML Skills.csv');
  const outPath = path.join(ROOT, 'survey', 'data.js');

  const aiRaw = fs.readFileSync(aiPath, 'utf8').replace(/\uFEFF/g, '');
  const deRaw = fs.readFileSync(dePath, 'utf8').replace(/\uFEFF/g, '');
  const mlRaw = fs.readFileSync(mlPath, 'utf8').replace(/\uFEFF/g, '');

  const ai = parseAIVertical(aiRaw);
  const de = parseHorizontal(deRaw);
  const ml = parseHorizontal(mlRaw);

  const data = { de, ml, ai };
  const out = 'window.DEMLAI_SURVEY_DATA = ' + JSON.stringify(data, null, 2) + ';\n';
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, out, 'utf8');
  console.log('Wrote', outPath, '| DE:', de.length, 'ML:', ml.length, 'AI:', ai.length);
}

main();
