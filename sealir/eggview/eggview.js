
document.addEventListener('click', (e) => {

    if (e.target.classList.contains("term")) {
        e.stopPropagation();

        document.querySelectorAll('.activated').forEach(term => {
            term.classList.remove('activated');
        });

        e.target.classList.add('activated');
    }
});

/* JS to copy HTML element of the referenced eclass */
document.addEventListener('click', function (e) {
    if (e.target.classList.contains('eclass_name')) {
        const ec = e.target.dataset.eclass;
        const targetContent = document.querySelector(`div[data-eclass="${ec}"]`);
        if (targetContent) {
            e.target.parentElement.innerHTML = targetContent.innerHTML;
        }
        e.preventDefault();
    }
});

document.addEventListener('keydown', function (e) {
    const termDiv = document.querySelector("div.activated");
    if (!termDiv) return;

    if (e.key == 'f') {
        const mainDiv = document.getElementById('main');
        mainDiv.innerHTML = termDiv.innerHTML;
        e.preventDefault();
    } else if (e.key == 'v') {
        const op = termDiv.dataset.termOp;
        const terms = document.querySelectorAll(`div[data-term-op="${op}"] > div.content`);
        terms.forEach(term => {
            term.style.display = term.style.display === 'none' ? 'block' : 'none';
        });
        e.preventDefault();
    } else if (e.key == 's') {
        convertToDropdown(termDiv.closest('.eclass'));
    }
});


document.addEventListener('DOMContentLoaded', function (e) {
    /* copy original state */
    const mainDiv = document.getElementById('main');
    const hiddenDiv = document.getElementById('hidden');
    hiddenDiv.innerHTML = mainDiv.innerHTML;
});

function convertToDropdown(eclassDiv) {
    const terms = Array.from(eclassDiv.children).filter(el => el.classList.contains('term'));
    if (terms.length <= 1) return;

    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '10px';

    const select = document.createElement('select');
    select.className = 'term-selector';
    select.style.height = 'fit-content';

    // Find activated term index
    const activatedIndex = terms.findIndex(term => term.classList.contains('activated'));
    const defaultIndex = activatedIndex >= 0 ? activatedIndex : 0;

    terms.forEach((term, idx) => {
        const option = document.createElement('option');
        option.value = idx;
        option.textContent = `${term.dataset.termOp} ${idx + 1}`;
        option.selected = idx === defaultIndex;
        select.appendChild(option);
    });

    select.addEventListener('change', (e) => {
        terms.forEach((term, idx) => {
            term.style.display = idx === parseInt(e.target.value) ? 'block' : 'none';
        });
    });

    terms.forEach((term, idx) => {
        term.style.display = idx === defaultIndex ? 'block' : 'none';
        eclassDiv.removeChild(term);
        container.appendChild(term);
    });

    container.appendChild(select);
    eclassDiv.appendChild(container);
}

function revertDropdowns() {
    document.querySelectorAll('.eclass').forEach(eclassDiv => {
        const container = eclassDiv.querySelector('div');
        if (!container) return;

        const terms = Array.from(container.querySelectorAll('.term'));
        const select = container.querySelector('select');

        if (select && terms.length > 0) {
            // Remove container and move terms back directly under eclass
            terms.forEach(term => {
                term.style.display = 'block';
                eclassDiv.appendChild(term);
            });
            container.remove();
        }
    });
}


const filterBox = document.querySelector('.bottom-textbox');

filterBox.addEventListener('input', function () {
    const hideOps = this.value.split(',').map(s => s.trim());
    const allTerms = document.querySelectorAll('div.term[data-term-op]');

    allTerms.forEach(term => {
        const opname = term.getAttribute('data-term-op');
        if (hideOps.includes(opname)) {
            term.style.display = 'none';
        } else {
            term.style.display = 'block';
        }
    });
});
