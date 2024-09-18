import html
from typing import Any

from sealir.ase import BaseExpr
from sealir import ase
from sealir.rewriter import TreeRewriter


def to_html(root: BaseExpr) -> str:

    reachable = {
        node for _, node in ase.walk_descendants_depth_first_no_repeat(root)
    }

    class ToHtml(TreeRewriter[str]):

        reference_already = set()

        def rewrite_generic(
            self, orig: BaseExpr, args: tuple[Any, ...], updated: bool
        ) -> str | BaseExpr:
            if orig in reachable:
                args = list(args)
                for i, child in enumerate(orig._args):
                    if isinstance(child, BaseExpr):
                        if (
                            not ase.is_simple(child)
                            and child in self.reference_already
                        ):
                            args[i] = (
                                f"<div class='handle_ref handle' data-ref='{child._handle}'>${child._handle}</div>"
                            )
                        else:
                            self.reference_already.add(child)

                parts = [html.escape(orig._head)]
                parts.extend(map(str, args))
                handle = f"<div class='handle_origin handle' data-ref='{orig._handle}'>${orig._handle}</div>"
                out = f"<div class='sexpr'>{handle} {' '.join(parts)}</div>"
                return out
            else:
                return None

    cvt = ToHtml()
    ase.apply_bottomup(root, cvt)
    return cvt.memo[root]


def style_text():
    return """
 <style>
.sexpr-container {
    display: flex;
    flex-wrap: wrap;
}

div {
    background: none;
}

div.sexpr {
    display: flex;
    flex-wrap: wrap;
    justify-content: left;
    align-items: top;

    background: none;

    margin: 2px;
    margin-right: 0;
    padding-top: 2px;
    padding-left: 5px;
    margin-left: 4px;
    border: 1px solid #ccc;
    border-left: 2px solid #ccc;
    border-bottom: none;
    border-right: none;
    cursor: pointer;

    min-width: 100px; /* Set a minimum width to ensure proper spacing */
}

div.sexpr .collapsed {
    color: #333333;
    background-color: rgb(200, 200, 200);
}

div.sexpr .collapsed div.sexpr {
    display: none;
}

div.sexpr:hover {
    /* box-shadow: 0 0 5px rgba(0, 0, 255, 0.5); */

  border-left-color: blue;
}

div.handle_origin {
    font-size: 0.5em;
    margin: 1px;
    color: #999999
}

div.handle_ref {
    padding: 1px;
    margin: 1px;
    text-decoration: underline;
    color: #666666
}

.selected {
    color: #ff0000;
}

#canvas {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
}
</style>
"""


def write_html(file, *contents):
    print("<!DOCTYPE html>", file=file)
    print("<html>", file=file)
    print("<head>", file=file)
    print(style_text(), file=file)
    print(script_text(), file=file)
    print("</head>", file=file)
    print("<body>", file=file)
    for content in contents:
        print(f"<div class='sexpr-container'>{content}</div>", file=file)
    print('<canvas id="canvas"></canvas>', file=file)
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

        const children = element.querySelectorAll('div.sexpr');
        children.forEach(child => {
            child.classList.add('collapsed');
            // applyCollapseRecursive(child);
        });

    }

});


document.addEventListener('DOMContentLoaded', () => {
    const handles = document.querySelectorAll('.handle');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    handles.forEach(handle => {
        handle.addEventListener('mouseover', () => {
            document.querySelectorAll(".selected").forEach(el => {
                el.classList.remove('selected');
            });


            const refNNN = handle.dataset.ref;
            const elementsWithSameRef = document.querySelectorAll(`[data-ref="${refNNN}"]`);

            elementsWithSameRef.forEach(el => {
                el.classList.add('selected');
            });

            drawEdges(ctx, Array.from(elementsWithSameRef).filter(node => {
                const parent = node.closest('.sexpr');
                return !parent || !parent.classList.contains("collapsed");
            }));
        });

    });
});

function drawEdges(ctx, elements) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centers = elements.map(el => {
        const rect = el.getBoundingClientRect();
        const scrollRect = document.scrollingElement.getBoundingClientRect();
        const canvasX = rect.left + rect.width / 2;
        const canvasY = rect.top + rect.height / 2;
        return { x: canvasX, y: canvasY };
    });

    ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.lineWidth = 2;

    for (let i = 0; i < centers.length - 1; i++) {
        console.log(centers[i]);
        ctx.beginPath();
        ctx.moveTo(centers[i].x, centers[i].y);
        ctx.lineTo(centers[i+1].x, centers[i+1].y);
        ctx.stroke();
    }
}


// Function to adjust SVG size
function adjustSVGSize() {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const svgWidth = viewportWidth;
  const svgHeight = viewportHeight;

  // Set SVG dimensions
  canvas.setAttribute('width', `${svgWidth}px`);
  canvas.setAttribute('height', `${svgHeight}px`);

  // Center the SVG
  canvas.style.position = 'fixed';
  canvas.style.top = '0';
  canvas.style.left = '0';
  canvas.style.width = `${viewportWidth}px`;
  canvas.style.height = `${viewportHeight}px`;

}


window.addEventListener('scroll', adjustSVGSize);
window.addEventListener('resize', adjustSVGSize);
</script>
    """
    return out
