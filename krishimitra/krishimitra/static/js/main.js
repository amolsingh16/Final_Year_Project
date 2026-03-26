// Dark mode toggle with persistence
(function() {
  const toggle = document.getElementById('darkModeToggle');
  const current = localStorage.getItem('ky-theme');
  if (current === 'dark') document.body.classList.add('dark');
  if (toggle) {
    toggle.checked = document.body.classList.contains('dark');
    toggle.addEventListener('change', () => {
      document.body.classList.toggle('dark');
      localStorage.setItem('ky-theme', document.body.classList.contains('dark') ? 'dark' : 'light');
    });
  }
})();

// Simple animated counters
document.querySelectorAll('[data-counter]').forEach(el => {
  const target = parseInt(el.getAttribute('data-counter'), 10) || 0;
  let current = 0;
  const step = Math.max(1, Math.floor(target / 80));
  const tick = () => {
    current += step;
    if (current >= target) { el.textContent = target; return; }
    el.textContent = current;
    requestAnimationFrame(tick);
  };
  const observer = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) { tick(); observer.disconnect(); }
  }, { threshold: 0.3 });
  observer.observe(el);
});


