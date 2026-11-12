// nist-header-footer.js fetches the NIST boilerplate header after DOMContentLoaded
// and inserts it at the top of the body. The logo link of that header is moved into
// the zensical header, so the two render as one band, and the emptied boilerplate
// header is removed.
(function () {
  function merge() {
    var holder = document.getElementById('nistheadergoeshere');
    var logo = holder && holder.querySelector('.nist-header__logo-link');
    var inner = document.querySelector('.md-header__inner');
    if (!logo || !inner) {
      return false;
    }
    inner.insertBefore(logo, inner.firstChild);
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
