"""
consolidate_markdown.py — Joins multiline st.markdown() calls where the HTML
is on one line ending with comma, and unsafe_allow_html=True is on the next line.
Converts to a single-line call so the flag is unambiguously part of the call.
"""
import re, sys, ast
sys.stdout.reconfigure(encoding='utf-8')

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
fixes = 0
html_sigs = ['style=', '<div', '<span', '<br', '<h1', '<h2', '<h3', '<hr', '<table', '<script']

while i < len(lines):
    line = lines[i]
    stripped = line.rstrip()

    # Check: does this line have st.markdown(..., ending with a trailing comma?
    # AND the next line is just: <whitespace>unsafe_allow_html=True)
    if (i + 1 < len(lines) and
        '.markdown(' in stripped and
        stripped.endswith(',') and
        'unsafe_allow_html' not in stripped and
        any(s in stripped for s in html_sigs)):

        next_line = lines[i + 1].rstrip()
        # Next line should be purely: whitespace + unsafe_allow_html=True)
        if re.match(r'^\s+unsafe_allow_html=True\)', next_line):
            # Merge: remove trailing comma from current line, append  , unsafe_allow_html=True)
            merged = stripped[:-1] + ', unsafe_allow_html=True)\n'
            new_lines.append(merged)
            i += 2  # skip next line (already merged)
            fixes += 1
            print(f'Fixed line {i-1}: {stripped[:80]}...')
            continue

    new_lines.append(line)
    i += 1

print(f'\nTotal merged: {fixes}')

if fixes > 0:
    # Verify syntax
    content = ''.join(new_lines)
    try:
        ast.parse(content)
        print('Syntax OK after merge.')
        with open('app.py', 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print('Saved.')
    except SyntaxError as e:
        print(f'SYNTAX ERROR after merge: {e} — NOT saving.')
else:
    print('Nothing to merge.')
