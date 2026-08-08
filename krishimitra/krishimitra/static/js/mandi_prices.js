(function () {
  const form = document.getElementById("mandiSearchForm");
  const overlay = document.getElementById("mandiLoadingOverlay");
  const searchBtn = document.getElementById("mandiSearchBtn");
  const stateSelect = document.getElementById("stateSelect");

  if (!form || !overlay || !searchBtn) {
    return;
  }

  form.addEventListener("submit", function () {
    overlay.classList.remove("d-none");
    searchBtn.disabled = true;

    const btnText = searchBtn.querySelector(".btn-text");
    const btnSpinner = searchBtn.querySelector(".btn-spinner");
    if (btnText) btnText.classList.add("d-none");
    if (btnSpinner) btnSpinner.classList.remove("d-none");
  });

  // Reload commodity options when state changes (without search flag)
  if (stateSelect) {
    stateSelect.addEventListener("change", function () {
      const url = new URL(window.location.href);
      url.searchParams.set("state", stateSelect.value);
      url.searchParams.delete("commodity");
      url.searchParams.delete("search");
      window.location.href = url.toString();
    });
  }
})();
