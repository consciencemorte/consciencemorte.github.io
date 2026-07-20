# Conscience Morte

Site éditorial construit avec Jekyll.

## Organisation

- `_posts/` : observations et monographies datées.
- `_biosemiotique/` : essais de prospective et de biosémiotique.
- `_notes/` : notes courtes.
- `_projects/` : projets de terrain (à créer au premier projet).
- `_includes/` : fragments HTML et figures SVG intégrées dans les articles.
- `_layouts/` : gabarits de pages.
- `_templates/` : modèles de front matter pour les nouveaux contenus.
- `assets/img/` : images publiques, classées par type de contenu puis par article.
- `resources/` : sources de travail et références, exclues du site généré.
- `images_bank/` : banque d’images locale, ignorée par Git et exclue du site.
- `old_archive/` : ancienne version locale, ignorée par Git et exclue du site.

Les dossiers générés (`_site/`, `.jekyll-cache/`, `vendor/`) ne font pas partie des sources du site.

## Nommage

Les nouvelles sources publiques utilisent des noms descriptifs en minuscules, séparés par des tirets. Les contenus Jekyll conservent le préfixe de date `AAAA-MM-JJ-`. Les noms techniques imposés par Jekyll et les deux dossiers locaux historiques `images_bank/` et `old_archive/` font exception.
