"""Build survey/data.js from AI, DE, ML Skills CSVs. Run: python scripts/build_survey_data.py"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = ['Beginner', 'Novice', 'Competent', 'Proficient', 'Expert']

def parse_ai_vertical(path):
    skills = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header_idx = None
    for i, row in enumerate(rows):
        if len(row) >= 4 and row[0] == 'Subdomain' and row[1] == 'Skill' and row[2] == 'Level' and row[3] == 'Description':
            header_idx = i
            break
    if header_idx is None:
        return skills
    cur_sub, cur_skill, level_descs = '', '', []
    for row in rows[header_idx + 1:]:
        sub = (row[0] if len(row) > 0 else '').strip()
        skill = (row[1] if len(row) > 1 else '').strip()
        level = (row[2] if len(row) > 2 else '').strip()
        desc = (row[3] if len(row) > 3 else '').strip()
        if sub or skill:
            if cur_skill and level_descs:
                ordered = [next((d for lv, d in level_descs if lv == L), '') for L in LEVELS]
                skills.append({'subdomain': cur_sub, 'skill': cur_skill, 'levels': ordered})
            cur_sub = sub or cur_sub
            cur_skill = skill or cur_skill
            level_descs = []
        if level in LEVELS:
            level_descs.append((level, desc))
    if cur_skill and level_descs:
        ordered = [next((d for lv, d in level_descs if lv == L), '') for L in LEVELS]
        skills.append({'subdomain': cur_sub, 'skill': cur_skill, 'levels': ordered})
    return skills

def parse_horizontal(path):
    skills = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header_idx = None
    for i, row in enumerate(rows):
        if len(row) >= 7 and 'Subdomain' in str(row[0]) and 'Skill' in str(row[1]) and 'Beginner' in str(row[2]):
            header_idx = i
            break
    if header_idx is None:
        return skills
    for row in rows[header_idx + 1:]:
        if len(row) < 7:
            continue
        subdomain = (row[0] or '').strip()
        skill = (row[1] or '').strip()
        if not skill:
            continue
        levels = [(row[i] if len(row) > i else '').strip() for i in range(2, 7)]
        skills.append({'subdomain': subdomain, 'skill': skill, 'levels': levels})
    return skills

def main():
    ai_path = os.path.join(ROOT, 'AI Skills.csv')
    de_path = os.path.join(ROOT, 'DE Skills.csv')
    ml_path = os.path.join(ROOT, 'ML Skills.csv')
    out_path = os.path.join(ROOT, 'survey', 'data.js')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ai = parse_ai_vertical(ai_path)
    de = parse_horizontal(de_path)
    ml = parse_horizontal(ml_path)
    data = {'de': de, 'ml': ml, 'ai': ai}
    js = 'window.DEMLAI_SURVEY_DATA = ' + json.dumps(data, indent=2) + ';\n'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(js)
    print('Wrote', out_path, '| DE:', len(de), 'ML:', len(ml), 'AI:', len(ai))

if __name__ == '__main__':
    main()
