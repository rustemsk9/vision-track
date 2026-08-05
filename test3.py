import html
html_code = "<html>\n<body>\n<script>\n// comment\nconst x = 'https://';\n</script>\n</body>\n</html>"
srcdoc = html.escape(html_code).replace('\n', '&#10;')
print(f'<iframe srcdoc="{srcdoc}"></iframe>')
