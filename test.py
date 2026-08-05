import html

with open("frontend_2d/index.html", "r") as f:
    html_code = f.read()

with open("frontend_2d/app.js", "r") as f_js:
    js_code = f_js.read()

html_code = html_code.replace('<script src="app.js"></script>', f'<script>\n{js_code}\n</script>')
srcdoc = html.escape(html_code)
print(srcdoc[1400:1800])
