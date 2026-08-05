import html
with open("frontend_2d/index.html", "r") as f:
    html_code = f.read()
with open("frontend_2d/app.js", "r") as f_js:
    js_code = f_js.read()
html_code = html_code.replace('<script src="app.js"></script>', f'<script>\n{js_code}\n</script>')
lines = html_code.split('\n')
print(f"Line 150: {lines[149]}")
print(f"Line 151: {lines[150]}")
print(f"Line 152: {lines[151]}")
