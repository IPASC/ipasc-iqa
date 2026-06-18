// Mobile nav: toggle the collapsed menu (shown as a hamburger below a breakpoint).
(function () {
  function init() {
    var btn = document.getElementById("nav-toggle");
    var nav = document.getElementById("site-nav");
    if (!btn || !nav) return;

    btn.addEventListener("click", function () {
      var open = nav.classList.toggle("nav--open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    });

    // Collapse after following a link (the page navigates anyway).
    nav.addEventListener("click", function (e) {
      if (e.target.closest("a")) {
        nav.classList.remove("nav--open");
        btn.setAttribute("aria-expanded", "false");
      }
    });
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
