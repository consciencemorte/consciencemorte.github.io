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
    const headings = Array.from(content.querySelectorAll('h2'));
    if (headings.length) {
      const list = document.createElement('ol');
      headings.forEach(function (heading, index) {
        if (!heading.id) heading.id = 'section-' + (index + 1);
        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = '#' + heading.id;
        link.textContent = heading.textContent;
        item.appendChild(link);
        list.appendChild(item);
      });
      toc.innerHTML = '';
      toc.appendChild(list);
    }
  }
}());
