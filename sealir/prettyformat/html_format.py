import html
from typing import Any, cast

from sealir import ase
from sealir.ase import SExpr
from sealir.rewriter import TreeRewriter, metadata_find_original


def find_source_md(node: SExpr) -> SExpr | None:
    def loc_test(x):
        if isinstance(x, SExpr):
            if x._head.startswith("PyAst"):
                if x._args[-1]._head == "PyAst_loc":
                    return True
        return False

    out = metadata_find_original(node, loc_test)
    if out is not None:
        return cast(SExpr, out._args[-1])
    else:
        return None


def to_html(root: SExpr) -> str:

    reachable = {
        node for _, node in ase.walk_descendants_depth_first_no_repeat(root)
    }

    class ToHtml(TreeRewriter[str]):

        reference_already: set[SExpr] = set()

        def rewrite_generic(
            self, orig: SExpr, args: tuple[Any, ...], updated: bool
        ) -> str | SExpr:
            if orig in reachable:
                args_list = list(args)
                for i, child in enumerate(orig._args):
                    if isinstance(child, SExpr):
                        if (
                            not ase.is_simple(child)
                            and child in self.reference_already
                        ):
                            args_list[i] = (
                                "<div class='handle_ref handle' "
                                f"data-ref='{child._handle}'>"
                                f"${child._handle}"
                                "</div>"
                            )
                        else:
                            self.reference_already.add(child)
                args = tuple(args_list)

                parts = [html.escape(orig._head)]
                parts.extend(map(str, args))
                handle = (
                    "<div class='handle_origin handle' "
                    f"data-ref='{orig._handle}'>"
                    f"${orig._handle}"
                    "</div>"
                )
                if src := find_source_md(orig):
                    si = cast(tuple[int, ...], src._args)
                    pp_src = f"[{si[0]}:{si[1]-1} to {si[2]}:{si[3]}]"
                    data = " ".join(
                        f"data-{k}={v}"
                        for k, v in zip(
                            ["lineStart", "colStart", "lineEnd", "colEnd"], si
                        )
                    )
                    source = f"<div class='source_info' {data}>{pp_src}</div>"
                else:
                    source = ""
                out = f"<div class='sexpr'>{handle}{source}{' '.join(parts)}</div>"
                return out
            else:
                raise AssertionError('unreachable')

    cvt = ToHtml()
    ase.apply_bottomup(root, cvt)
    res = cvt.memo[root]
    return f"<div class='sexpr-container'>{res}</div>"


def prepare_source(source_text: str) -> str:
    return f"<div class='pre-container'><pre id='source-text'>{source_text}</pre></div>"


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
  border-top-color: blue;
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

div.source_info {
    font-size: 0.6em;
}

.selected {
    color: #ff0000;
}

div.pre-container {
  display: block;
  padding: 10px;
  border: 1px solid #ccc;
  font-family: monospace;
  white-space: pre-wrap;
}

.highlight {
  background-color: yellow;
}

.tooltip {
  position: absolute;
  z-index: 1000;
  background-color: rgba(255, 255, 255, 0.9);
  border: 1px solid #333;
  padding: 10px;
  max-width: 300px;
  max-height: 300px;
  overflow-y: auto;
  word-break: break-all;
  font-family: monospace;
  font-size: .8em;
  white-space: pre-wrap;
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
        print(content, file=file)
    print('<canvas id="canvas"></canvas>', file=file)
    print("</body>", file=file)
    print("</html>", file=file)


def script_text():
    out = r"""

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


///////// highlight source

let originalSourceText;

function highlightCode(startLine, endLine, startColumn, endColumn) {
  originalSourceText = document.querySelector('body > .pre-container pre').textContent;


  // Get the pre element
  const preElement = document.querySelector('.pre-container pre');

  if (!preElement) {
    console.error('No <pre> element found');
    return;
  }
  // Split the content into lines
  const lines = preElement.textContent.split('\n');


  // Find the range of lines to highlight
  const startIndex = Math.max(0, startLine);
  const endIndex = Math.min(lines.length - 1, endLine);

  // Apply highlighting styles

  for (let i = startIndex; i <= endIndex; i++) {
    const line = lines[i];
    if (line.slice(startColumn - 1, endColumn)) {
        const highlightedLine = line.slice(0, startColumn - 1) + '<span class="highlight">' + line.slice(startColumn - 1, endColumn) + '</span>' + line.slice(endColumn);
        // Replace the original line with the highlighted version
        lines[i] = highlightedLine;
    }
  }
  // Join the modified lines back together
  preElement.innerHTML = lines.join('\n');

}


function clearHighlight() {
  // Restore the original text
  const preElement = document.querySelector('body > .pre-container pre');
  if (preElement) {
    preElement.textContent = originalSourceText;
  }
}

// Function to handle click event on source_info div
function handleSourceInfoClick(event) {
  const sourceInfoDiv = event.target.closest('.source_info');
  if (!sourceInfoDiv) {
    console.error("source info div not found");
    return;
  }

  const linestart = parseInt(sourceInfoDiv.dataset.linestart, 10) - 1;
  const colstart = parseInt(sourceInfoDiv.dataset.colstart, 10);
  const lineend = parseInt(sourceInfoDiv.dataset.lineend, 10) - 1;
  const colend = parseInt(sourceInfoDiv.dataset.colend, 10);

  if (originalSourceText) clearHighlight();


  highlightCode(linestart, lineend, colstart, colend);
}

// Function to handle hover event on source_info div
function handleSourceInfoHover(event) {
  const sourceInfoDiv = event.target.closest('.source_info');
  if (!sourceInfoDiv) return;

  const linestart = parseInt(sourceInfoDiv.dataset.linestart, 10) - 1;
  const colstart = parseInt(sourceInfoDiv.dataset.colstart, 10);
  const lineend = parseInt(sourceInfoDiv.dataset.lineend, 10) - 1;
  const colend = parseInt(sourceInfoDiv.dataset.colend, 10);

  // Create tooltip element
  const tooltip = document.createElement('div');
  tooltip.className = 'tooltip';
  tooltip.style.position = 'absolute';

  // Add tooltip to DOM
  document.body.appendChild(tooltip);

  // Populate the tooltip with the highlighted text
  const lines = document.querySelector('body > .pre-container pre').textContent.split('\n');
  line = lines[linestart]
  if (line){
    tooltip.innerHTML = line.slice(0, colstart - 1) + '<span class="highlight">' + line.slice(colstart - 1, colend) + '</span>' + line.slice(colend);
    tooltip.innerHTML = `${linestart+1} | ` + tooltip.innerHTML;
  }
  // Function to update tooltip position
  function updateTooltipPosition(e) {
    const rect = tooltip.getBoundingClientRect();
    const x = e.clientX + 10;
    const y = e.clientY + 10;

    // Adjust tooltip position based on scroll
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;

    tooltip.style.left = `${x + scrollLeft}px`;
    tooltip.style.top = `${y + scrollTop}px`;


  }
  // Add event listener for mousemove
  window.addEventListener('mousemove', updateTooltipPosition);

  // Remove event listener when leaving the element
  sourceInfoDiv.addEventListener('mouseleave', () => {
    window.removeEventListener('mousemove', updateTooltipPosition);
    tooltip.remove();
  });
}

window.addEventListener("load", function(){

    // Add click event listener to all .source_info elements
    document.querySelectorAll('.source_info').forEach(element => {
        element.addEventListener('click', handleSourceInfoClick);
    });


    // Add hover event listener to all .source_info elements
    document.querySelectorAll('.source_info').forEach(element => {
        element.addEventListener('mouseenter', handleSourceInfoHover);
    });

});


</script>
    """
    return out
