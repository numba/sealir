import html
from typing import Any


from sealir.ase import Expr
from sealir.rewriter import TreeRewriter




def to_html(root: Expr) -> str:
    class ToHtml(TreeRewriter[str]):
        def rewrite_generic(self, orig: Expr, args: tuple[Any, ...], updated: bool) -> str | Expr:
            memo = self.memo
            if orig.is_metadata:
                return orig
            else:
                parts = [html.escape(orig.head)]
                parts.extend(map(str, args))
                out = f"<div class='sexpr'>{' '.join(parts)}</div>"
                return out

    cvt = ToHtml()
    root.apply_bottomup(cvt)
    return cvt.memo[root]



def style_text():
    return """
 <style>
.sexpr-container {
    display: flex;
    flex-wrap: wrap;
}

div.sexpr {
    display: flex;
    flex-direction: column;
    background-color: rgba(128, 128, 250, 0.05);
    padding: 5px;
    border: 1px solid #ccc;
    margin: 4px;
    cursor: pointer;
    transition: transform 0.3s ease-in-out;
    min-width: 800px; /* Set a minimum width to ensure proper spacing */
}

div.sexpr .collapsed {
    font-size: .8em;
    color: #333333;
}

div.sexpr .collapsed div.sexpr {
    display: none;
}

div.sexpr:hover {
    background-color: #f0f0f0;
    box-shadow: 0 0 5px rgba(0, 0, 255, 0.5);
}

</style>
"""


def write_html(content, file):
    print("<!DOCTYPE html>", file=file)
    print("<html>", file=file)
    print("<head>", file=file)
    print(style_text(), file=file)
    print(script_text(), file=file)
    print("</head>", file=file)
    print("<body>", file=file)
    print(f"<div class='sexpr-container'>{content}</div>", file=file)
    print("</body>", file=file)
    print("</html>", file=file)

def script_text():
    out = """

<script>
document.addEventListener('DOMContentLoaded', function() {
    const sexprContainers = document.querySelectorAll('.sexpr-container');

    sexprContainers.forEach(container => {
        const sexprs = container.querySelectorAll('div.sexpr');
        sexprs.forEach(sexpr => {
            // Add event listener for click
            sexpr.addEventListener('click', function(event) {
                if (this == event.target) {
                    if (this.classList.contains('collapsed')) {
                        this.classList.remove("collapsed");
                    } else {
                        // Recursively apply "collapsed" to all child elements
                        applyCollapseRecursive(this);
                    }
                }
            });


        });
    });

    function applyCollapseRecursive(element) {
        element.classList.add('collapsed');

        // Find all child elements
        const children = element.querySelectorAll('*');
        children.forEach(child => {
            applyCollapseRecursive(child);
        });
    }
});
</script>
    """
    return out