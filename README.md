# Conscience Morte

Site éditorial construit avec Jekyll.

## Organisation

- `_monographies/` → `/monographies/` : études techniques longues.
- `_biosemiotique/` → `/biosemiotique/` : prospective et systèmes de signes.
- `_terrain/` → `/terrain/` : protocoles, expériences et relevés.
- `_notes/` → `/notes/` : notes courtes et revues de littérature.
- `a-propos.md` → `/a-propos/` : présentation éditoriale.
- `_includes/` : fragments HTML et figures SVG intégrées dans les articles.
- `_layouts/` : gabarits de pages.
- `_templates/` : modèles de front matter pour les nouveaux contenus.
- `assets/img/` : images publiques, classées par type de contenu puis par article.
- `resources/` : sources de travail et références, exclues du site généré.
- `images_bank/` : banque d’images locale, ignorée par Git et exclue du site.
- `_tmp/` : imports et brouillons temporaires locaux, ignorés par Git et exclus du site.
- `old_archive/` : ancienne version locale, ignorée par Git et exclue du site.

Les dossiers générés (`_site/`, `.jekyll-cache/`, `vendor/`) ne font pas partie des sources du site.

## Nommage

Les sources publiques utilisent un slug descriptif en minuscules, séparé par des tirets, sans préfixe de date. La date de publication appartient au front matter. Les médias suivent `assets/img/<collection>/<slug>/`; les figures SVG incorporées suivent `_includes/figures/<collection>/<slug>/`.
