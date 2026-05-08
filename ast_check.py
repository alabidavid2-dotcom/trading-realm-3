"""
ast_check.py — Uses Python AST to find EVERY st.markdown() call
and check whether unsafe_allow_html=True is present when HTML is in the content.
"""
import ast, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('app.py', 'r', encoding='utf-8') as f:
    src = f.read()

tree = ast.parse(src)
lines = src.split('\n')

HTML_INDICATORS = ['<div', '<span', '<br', 'style=', '<style', '<hr', '<h1', '<h2', '<h3', '<table', '<b>', '<i>']

class MarkdownChecker(ast.NodeVisitor):
    def visit_Call(self, node):
        # Match: <anything>.markdown(...)
        is_markdown = (
            isinstance(node.func, ast.Attribute) and
            node.func.attr == 'markdown'
        )
        if not is_markdown:
            self.generic_visit(node)
            return

        lineno = node.lineno

        # Check for unsafe_allow_html=True keyword
        has_flag = any(
            kw.arg == 'unsafe_allow_html' and
            isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in node.keywords
        )

        # Get the first argument (the content)
        has_html = False
        if node.args:
            arg = node.args[0]
            # Try to extract string value
            try:
                # For simple string constants
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    has_html = any(t in arg.value for t in HTML_INDICATORS)
                # For f-strings (JoinedStr)
                elif isinstance(arg, ast.JoinedStr):
                    # Reconstruct the f-string template
                    parts = []
                    for value in arg.values:
                        if isinstance(value, ast.Constant):
                            parts.append(value.value)
                        else:
                            parts.append('{}')
                    template = ''.join(parts)
                    has_html = any(t in template for t in HTML_INDICATORS)
                # For variable names — we can't know statically, skip
                elif isinstance(arg, ast.Name):
                    pass  # Skip variable refs
            except Exception:
                pass

        if has_html and not has_flag:
            # Get the object name (st, status_text, col, etc.)
            obj = ast.unparse(node.func.value) if hasattr(ast, 'unparse') else '?'
            end_line = getattr(node, 'end_lineno', lineno)
            print(f'LINE {lineno}-{end_line}: {obj}.markdown(...) — HAS HTML, MISSING FLAG')
            # Show the actual lines
            for ln in range(lineno-1, min(end_line, lineno+3)):
                print(f'  {ln+1}: {lines[ln][:100]}')

        self.generic_visit(node)

checker = MarkdownChecker()
checker.visit(tree)
print('\nAST check complete.')
