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

  // After each HTMX swap of the annotation area, re-anchor the freshly drawn
  // stage at the top so the viewport doesn't "jump" between pairs/ratings on
  // mobile (where the stage is sized to fit the screen).
  document.body.addEventListener("htmx:afterSwap", function (e) {
    if (e.target && e.target.id === "contribute-root" && e.target.querySelector(".annotate-stage")) {
      window.scrollTo(0, 0);
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
