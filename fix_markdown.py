"""
fix_markdown.py — Adds unsafe_allow_html=True to any st.markdown() call
that contains HTML tags and is missing the flag.
Handles: single-line calls, and triple-quoted blocks whose CLOSING line is just ''').
"""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

html_tags = ['<div', '<span', '<br', '<style', '<hr', 'style=', '<h1', '<h2', '<h3',
             '<table', '<td', '<th', '<tr', '<b>', '<i>']

fixes = []

# === Pass 1: Single-line calls missing flag ===
for i, line in enumerate(lines):
    stripped = line.rstrip()
    if '.markdown(' not in stripped:
        continue
    has_html = any(t in stripped for t in html_tags)
    if not has_html:
        continue
    if 'unsafe_allow_html' in stripped:
        continue
    # Check if next line has the flag (multiline-formatted call)
    if i + 1 < len(lines) and 'unsafe_allow_html' in lines[i + 1]:
        continue
    # Call must be complete on this line
    if not stripped.endswith(')'):
        continue
    # Must be an actual .markdown( call
    if not re.search(r'[\w\[\]]+\.markdown\(', stripped):
        continue
    fixes.append(('single', i))

# === Pass 2: Triple-quoted calls whose closing line is just """) ===
for i, line in enumerate(lines):
    stripped = line.rstrip()
    # Closing line pattern: optional whitespace + """ + optional whitespace + )
    if not re.match(r'^\s*"""\s*\)\s*$', stripped):
        continue
    if 'unsafe_allow_html' in stripped:
        continue
    # Scan backwards for opening st.markdown(f"""
    found_open = False
    body_has_html = False
    for j in range(i - 1, max(i - 120, -1), -1):
        prev = lines[j]
        # Check if this line opens a markdown call
        if re.search(r'[\w\[\]]+\.markdown\(f?"""', prev):
            # Check entire body (from j to i) for HTML
            body = ''.join(lines[j:i + 1])
            body_has_html = any(t in body for t in html_tags)
            found_open = True
            break
        # If we hit another """) or triple-quote close, stop searching
        if re.match(r'^\s*"""', prev) and prev.strip() != '"""':
            break
    if found_open and body_has_html:
        fixes.append(('triple_close', i))

# Show what we found
single_fixes = [f for f in fixes if f[0] == 'single']
triple_fixes = [f for f in fixes if f[0] == 'triple_close']
print(f'Single-line calls needing flag: {len(single_fixes)}')
for f in single_fixes:
    print(f'  line {f[1]+1}: {lines[f[1]].rstrip()[:100]}')
print(f'Triple-close calls needing flag: {len(triple_fixes)}')
for f in triple_fixes:
    print(f'  line {f[1]+1}: {lines[f[1]].rstrip()[:100]}')

if not fixes:
    print('Nothing to fix - all HTML markdown calls already have the flag.')
    sys.exit(0)

# Apply fixes
new_lines = list(lines)
for fix in fixes:
    kind = fix[0]
    i = fix[1]
    original = new_lines[i]
    if kind == 'single':
        # Replace trailing ) with , unsafe_allow_html=True)
        fixed = re.sub(r'\)\s*$', ', unsafe_allow_html=True)', original.rstrip()) + '\n'
        new_lines[i] = fixed
        print(f'Fixed single line {i+1}')
    elif kind == 'triple_close':
        # Replace """) with """, unsafe_allow_html=True)
        indent = len(original) - len(original.lstrip())
        fixed = original[:indent] + '""", unsafe_allow_html=True)\n'
        new_lines[i] = fixed
        print(f'Fixed triple-close line {i+1}')

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f'\nTotal {len(fixes)} fixes applied and saved.')
