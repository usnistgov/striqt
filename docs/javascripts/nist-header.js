// nist-header-footer.js fetches the NIST boilerplate header after DOMContentLoaded
// and inserts it at the top of the body. overrides/partials/header.html already
// renders a copy of its logo link in the zensical header, so the fetched link replaces
// that copy in place, and the emptied boilerplate header is removed.
(function () {
  function merge() {
    var holder = document.getElementById('nistheadergoeshere');
    var logo = holder && holder.querySelector('.nist-header__logo-link');
    var inner = document.querySelector('.md-header__inner');
    if (!logo || !inner) {
      return false;
    }
    var placed = inner.querySelector('.nist-header__logo-link');
    if (placed) {
      inner.replaceChild(logo, placed);
    } else {
      inner.insertBefore(logo, inner.firstChild);
    }
    holder.remove();
    return true;
  }

  if (merge()) {
    return;
  }
  var observer = new MutationObserver(function () {
    if (merge()) {
      observer.disconnect();
    }
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
