(function () {
  const root = document.documentElement;
  const toggle = document.querySelector('.theme-toggle');

  if (toggle) {
    toggle.addEventListener('click', function () {
      const current = root.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const next = current === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      localStorage.setItem('cm-theme', next);
    });
  }

  const progress = document.querySelector('.reading-progress span');
  if (progress) {
    const updateProgress = function () {
      const height = document.documentElement.scrollHeight - innerHeight;
      progress.style.transform = 'scaleX(' + (height > 0 ? scrollY / height : 0) + ')';
    };
    addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  }

  const content = document.getElementById('post-content');
  const toc = document.getElementById('toc');
  if (content && toc) {
    const headings = Array.from(content.querySelectorAll('h2, h3'));
    if (headings.length) {
      const list = document.createElement('ol');
      headings.forEach(function (heading, index) {
        if (!heading.id) heading.id = 'section-' + (index + 1);
        const item = document.createElement('li');
        if (heading.tagName === 'H3') item.classList.add('is-sub');
        const link = document.createElement('a');
        link.href = '#' + heading.id;
        link.textContent = heading.textContent;
        item.appendChild(link);
        list.appendChild(item);
      });
      toc.innerHTML = '';
      toc.appendChild(list);

      if ('IntersectionObserver' in window) {
        const links = Array.from(list.querySelectorAll('a'));
        const activate = function (id) {
          links.forEach(function (link) {
            link.parentElement.classList.toggle('is-active', link.getAttribute('href') === '#' + id);
          });
        };
        const observer = new IntersectionObserver(function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) activate(entry.target.id);
          });
        }, { rootMargin: '-12% 0px -76% 0px' });
        headings.forEach(function (heading) { observer.observe(heading); });
      }
    }
  }
}());
