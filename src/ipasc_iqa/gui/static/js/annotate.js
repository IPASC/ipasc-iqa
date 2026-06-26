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
      resetZoom(); // a fresh pair always starts at 1x
    }
  });

  // ---------------------------------------------------------------------
  // Synchronised scroll-to-zoom for the annotation image pair.
  //
  // Wheeling over either image zooms BOTH images at the same rate, anchored to
  // the location the pointer is hovering on. Scale is clamped to [1x, 10x];
  // zooming out bottoms out at the original (1x) framing. The zoom is rate-
  // limited so traversing the whole 1x→10x range takes at least MIN_FULL_SWEEP
  // seconds — a hard flick of the wheel still eases in smoothly rather than
  // snapping. Global + delegated, so it keeps working after each HTMX swap.
  // ---------------------------------------------------------------------
  const MIN_SCALE = 1;
  const MAX_SCALE = 10;
  const MIN_FULL_SWEEP = 2; // seconds for a full min→max (or max→min) sweep
  const MAX_RATE = (MAX_SCALE - MIN_SCALE) / MIN_FULL_SWEEP; // scale units / second

  let scale = MIN_SCALE; // currently rendered scale
  let target = MIN_SCALE; // scale the wheel is easing us toward
  // The content point under the cursor when a zoom gesture starts, as a fraction
  // of the image (0..1). The zoom keeps THIS point fixed as the scale changes, so
  // it stays put relative to the current (already panned/zoomed) view — no jumps.
  let pivotX = 0.5;
  let pivotY = 0.5;
  // Current translate, expressed as a fraction of each image's own size. Always
  // kept within [1 - scale, 0] so the image cannot be panned off its box.
  let ax = 0;
  let ay = 0;
  let raf = null;
  let lastT = 0;

  // Drag-to-pan state (only active while zoomed in). `suppressClick` cancels the
  // selection click that ends a genuine drag, so panning never picks an image.
  const DRAG_THRESHOLD = 5; // px of movement before a press counts as a drag
  let dragging = false;
  let dragMoved = false;
  let suppressClick = false;
  let panImg = null;
  let lastX = 0;
  let lastY = 0;
  let downX = 0;
  let downY = 0;

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function pairImages() {
    return Array.from(document.querySelectorAll(".annotate-pair img"));
  }

  function render() {
    const idle = scale > 1 ? "grab" : "";
    pairImages().forEach(function (im) {
      im.style.transformOrigin = "0 0";
      im.style.transform =
        "translate(" + ax * im.offsetWidth + "px, " + ay * im.offsetHeight + "px) scale(" + scale + ")";
      im.style.cursor = dragging ? "grabbing" : idle;
    });
  }

  function tick(now) {
    if (!lastT) lastT = now;
    const dt = (now - lastT) / 1000;
    lastT = now;
    const maxStep = MAX_RATE * dt;
    const diff = target - scale;
    const prev = scale;
    scale = Math.abs(diff) <= maxStep ? target : scale + Math.sign(diff) * maxStep;
    // Hold the pivot (the point under the cursor) fixed as the scale changes,
    // adjusting the *current* translate incrementally rather than re-deriving it.
    const d = scale - prev;
    ax = clamp(ax - d * pivotX, 1 - scale, 0);
    ay = clamp(ay - d * pivotY, 1 - scale, 0);
    render();
    if (scale !== target) {
      raf = requestAnimationFrame(tick);
    } else {
      raf = null;
      lastT = 0;
    }
  }

  function animate() {
    if (raf == null) {
      lastT = 0;
      raf = requestAnimationFrame(tick);
    }
  }

  function resetZoom() {
    if (raf != null) {
      cancelAnimationFrame(raf);
      raf = null;
    }
    scale = target = MIN_SCALE;
    pivotX = pivotY = 0.5;
    ax = ay = 0;
    dragging = false;
    suppressClick = false;
    lastT = 0;
    pairImages().forEach(function (im) {
      im.style.transform = "";
      im.style.transformOrigin = "";
      im.style.cursor = "";
    });
  }

  document.addEventListener(
    "wheel",
    function (e) {
      const pair = e.target.closest && e.target.closest(".annotate-pair");
      if (!pair) return;
      e.preventDefault(); // take over the wheel; don't scroll the page

      // Anchor on the image actually under the pointer (fractions are preserved
      // under transform, so this is correct even mid-zoom).
      const im = (e.target.closest && e.target.closest("img")) || pair.querySelector("img");
      if (!im) return;
      // Pivot = content fraction under the cursor in the CURRENT view (fractions
      // are preserved under the transform, so reading the live rect is correct
      // even after panning/zooming) — this is what keeps the zoom jump-free.
      const r = im.getBoundingClientRect();
      pivotX = clamp((e.clientX - r.left) / r.width, 0, 1);
      pivotY = clamp((e.clientY - r.top) / r.height, 0, 1);

      // Normalise line-mode deltas to pixels, then zoom multiplicatively.
      const delta = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      target = clamp(target * Math.exp(-delta * 0.0015), MIN_SCALE, MAX_SCALE);
      animate();
    },
    { passive: false }
  );

  // --- Drag-to-pan while zoomed in --------------------------------------
  document.addEventListener("pointerdown", function (e) {
    suppressClick = false;
    if (e.button !== 0) return; // left button only
    const pair = e.target.closest && e.target.closest(".annotate-pair");
    if (!pair || scale <= 1) return; // panning only makes sense once zoomed in
    const im = e.target.closest("img");
    if (!im) return;
    dragging = true;
    dragMoved = false;
    panImg = im;
    downX = lastX = e.clientX;
    downY = lastY = e.clientY;
    try {
      im.setPointerCapture(e.pointerId);
    } catch (_) {
      /* capture is best-effort */
    }
    render(); // switch the cursor to "grabbing"
  });

  document.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    const w = panImg.offsetWidth;
    const h = panImg.offsetHeight;
    ax = clamp(ax + (e.clientX - lastX) / w, 1 - scale, 0);
    ay = clamp(ay + (e.clientY - lastY) / h, 1 - scale, 0);
    lastX = e.clientX;
    lastY = e.clientY;
    if (Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > DRAG_THRESHOLD) {
      dragMoved = true;
    }
    render();
  });

  document.addEventListener("pointerup", function (e) {
    if (!dragging) return;
    dragging = false;
    try {
      panImg.releasePointerCapture(e.pointerId);
    } catch (_) {
      /* no-op */
    }
    if (dragMoved) suppressClick = true; // the trailing click is a drag, not a pick
    render();
  });

  // A drag that ends over an image must not submit the choice form. Cancel the
  // click in the capture phase, before it reaches the form's submit handler.
  document.addEventListener(
    "click",
    function (e) {
      if (suppressClick && e.target.closest && e.target.closest(".annotate-pair")) {
        e.preventDefault();
        e.stopPropagation();
      }
      suppressClick = false;
    },
    true
  );

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
