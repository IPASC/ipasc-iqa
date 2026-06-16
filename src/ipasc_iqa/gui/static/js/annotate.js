// Keyboard + live-value wiring for the single-stimulus MOS rating stage.
// Delegated/global so it keeps working after each HTMX swap of the rate fragment.
(function () {
  function slider(stage) {
    return stage.querySelector(".annotate-rate__slider");
  }

  function setValue(stage, v) {
    const s = slider(stage);
    if (!s) return;
    s.value = String(v);
    const out = stage.querySelector(".annotate-rate__value");
    if (out) out.textContent = String(v);
  }

  // Keep the displayed number in sync when the slider is dragged.
  document.addEventListener("input", function (e) {
    const s = e.target;
    if (s && s.classList && s.classList.contains("annotate-rate__slider")) {
      const out = s.closest(".annotate-rate").querySelector(".annotate-rate__value");
      if (out) out.textContent = s.value;
    }
  });

  // Keys 1–4 set the score; space confirms.
  document.addEventListener("keydown", function (e) {
    const stage = document.querySelector(".annotate-rate");
    if (!stage) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" && e.target.type !== "range") return; // don't hijack text fields

    if (e.key >= "1" && e.key <= "4") {
      setValue(stage, e.key);
      e.preventDefault();
    } else if (e.key === " " || e.code === "Space") {
      const btn = stage.querySelector(".annotate-rate__confirm");
      if (btn) {
        btn.click();
        e.preventDefault();
      }
    }
  });
})();
