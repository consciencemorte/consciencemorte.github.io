---
layout: page
title: Biosémiotique
permalink: /biosemiotique/
redirect_from: [/essais/]
intro: Textes prospectifs sur l’alignement, la cognition artificielle, les systèmes de signes et les trajectoires de l’intelligence non biologique.
---

{% assign sorted_biosemiotique = site.biosemiotique | sort: 'date' | reverse %}
{% if sorted_biosemiotique.size > 0 %}
<ul class="archive-list">
{% for article in sorted_biosemiotique %}
  <li>
    <span class="archive-kind">{{ article.status | default: 'ARTICLE' | upcase }}</span>
    <div class="archive-entry">
      <a href="{{ article.url | relative_url }}">{{ article.title }}</a>
      {% if article.topics %}<span>{{ article.topics | join: ' · ' }}</span>{% endif %}
    </div>
    <time datetime="{{ article.date | date_to_xmlschema }}">{{ article.date | date: '%d.%m.%Y' }}</time>
    {% if article.hero_image %}<a class="archive-thumb" href="{{ article.url | relative_url }}" tabindex="-1" aria-hidden="true"><img src="{{ article.hero_image | relative_url }}" alt="" style="object-position: {{ article.hero_position | default: 'center' }}"></a>{% endif %}
  </li>
{% endfor %}
</ul>
{% else %}
<div class="collection-empty">
  <span>BIOSÉMIOTIQUE / 000</span>
  <h2>Les premières études prospectives sont en préparation.</h2>
  <p>Cette section accueillera les textes prospectifs et philosophiques. Aucun contenu de démonstration ne sera publié à la place des textes définitifs.</p>
</div>
{% endif %}
