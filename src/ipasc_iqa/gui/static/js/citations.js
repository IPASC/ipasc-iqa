// Citation popup wiring, shared by every page that includes fragments/cite_dialog.html.
//
// Uses a single capture-phase click listener so that clicking a citation icon
// inside an otherwise-clickable element (e.g. an HTMX-bound table row) is
// intercepted before that element's own handler runs.
(function () {
  document.addEventListener(
    "click",
    function (e) {
      const dlg = document.getElementById("cite-dialog");

      // Measure info: clone the adjacent <template class="measure-info"> into
      // the measure dialog and open it.
      const infoBtn = e.target.closest && e.target.closest(".info-btn");
      if (infoBtn) {
        const infoDlg = document.getElementById("measure-dialog");
        const tpl = infoBtn.parentElement.querySelector(".measure-info");
        const body = document.getElementById("measure-dialog-body");
        if (infoDlg && tpl && body) {
          e.preventDefault();
          e.stopPropagation();
          body.innerHTML = "";
          body.appendChild(tpl.content.cloneNode(true));
          infoDlg.showModal();
          return;
        }
      }

      // Open: clicking a citation icon (or the <i> inside it).
      const citeBtn = e.target.closest && e.target.closest(".cite-btn");
      if (citeBtn && dlg) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById("cite-text").textContent = citeBtn.dataset.bibtex || "";
        const copyBtn = document.getElementById("cite-copy");
        if (copyBtn) copyBtn.textContent = "Copy";
        dlg.showModal();
        return;
      }

      // Copy to clipboard.
      if (e.target && e.target.id === "cite-copy") {
        e.preventDefault();
        const pre = document.getElementById("cite-text");
        navigator.clipboard.writeText(pre.textContent).then(
          function () { e.target.textContent = "Copied ✓"; },
          function () { e.target.textContent = "Select & copy manually"; }
        );
        return;
      }

      // Click on the backdrop (the dialog element itself) closes it.
      if (dlg && e.target === dlg) dlg.close();
      const infoDlg = document.getElementById("measure-dialog");
      if (infoDlg && e.target === infoDlg) infoDlg.close();
    },
    true // capture phase
  );
})();
