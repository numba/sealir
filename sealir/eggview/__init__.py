import pathlib

basedir = pathlib.Path(__file__).parent
html_file = basedir / 'eggview.html'
js_file = basedir / 'eggview.js'
css_file = basedir / 'eggview.css'


def write_page(fout, main_content):
    with open(html_file, 'r') as fin:
        template = fin.read()

    with open(js_file, 'r') as fin:
        js = fin.read()

    with open(css_file, 'r') as fin:
        css = fin.read()

    template = template.replace("/* <!-- [TEMPLATE.STYLE] --> */", css)
    template = template.replace("/* <!-- [TEMPLATE.SCRIPT] --> */", js)
    template = template.replace("<!-- [TEMPLATE.BODY] -->", main_content)
    fout.write(template)
