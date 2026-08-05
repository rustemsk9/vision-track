import html

with open("frontend_2d/index.html", "r") as f:
    html_code = f.read()

with open("frontend_2d/app.js", "r") as f_js:
    js_code = f_js.read()

html_code = html_code.replace('<script src="app.js"></script>', f'<script>\n{js_code}\n</script>')

lines = html_code.split('\n')
for i, line in enumerate(lines):
    if i >= 165 and i <= 175:
        print(f"Line {i+1}: {line}")

