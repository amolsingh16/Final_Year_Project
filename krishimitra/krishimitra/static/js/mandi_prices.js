(function () {
  const form = document.getElementById("mandiSearchForm");
  const overlay = document.getElementById("mandiLoadingOverlay");
  const searchBtn = document.getElementById("mandiSearchBtn");
  const stateSelect = document.getElementById("stateSelect");
  const commoditySelect = document.getElementById("commoditySelect");

  if (!form || !overlay || !searchBtn) {
    return;
  }

  function showLoading() {
    overlay.classList.remove("d-none");
    searchBtn.disabled = true;

    const btnText = searchBtn.querySelector(".btn-text");
    const btnSpinner = searchBtn.querySelector(".btn-spinner");
    if (btnText) btnText.classList.add("d-none");
    if (btnSpinner) btnSpinner.classList.remove("d-none");
  }

  form.addEventListener("submit", function () {
    showLoading();
  });

  // Immediately fetch mandi prices when state changes
  if (stateSelect) {
    stateSelect.addEventListener("change", function () {
      showLoading();
      const url = new URL(window.location.href);
      url.searchParams.set("state", stateSelect.value);
      if (commoditySelect && commoditySelect.value) {
        url.searchParams.set("commodity", commoditySelect.value);
      } else {
        url.searchParams.delete("commodity");
      }
      url.searchParams.set("search", "1");
      window.location.href = url.toString();
    });
  }

  // Immediately fetch mandi prices when commodity changes
  if (commoditySelect) {
    commoditySelect.addEventListener("change", function () {
      showLoading();
      form.submit();
    });
  }
})();
