---
layout: page
title: Prospective & Biosémiotique
permalink: /essais/
intro: Textes prospectifs sur l’alignement, la cognition artificielle, les systèmes de signes et les trajectoires de l’intelligence non biologique.
---

{% assign all_essays = site.essays | concat: site.biosemiotique %}
{% assign sorted_essays = all_essays | sort: 'date' | reverse %}
{% if sorted_essays.size > 0 %}
<ul class="archive-list">
{% for essay in sorted_essays %}
  <li>
    <span class="archive-kind">{{ essay.status | default: 'ESSAI' | upcase }}</span>
    <div class="archive-entry">
      <a href="{{ essay.url | relative_url }}">{{ essay.title }}</a>
      {% if essay.topics %}<span>{{ essay.topics | join: ' · ' }}</span>{% endif %}
    </div>
    <time datetime="{{ essay.date | date_to_xmlschema }}">{{ essay.date | date: '%d.%m.%Y' }}</time>
    {% if essay.hero_image %}<a class="archive-thumb" href="{{ essay.url | relative_url }}" tabindex="-1" aria-hidden="true"><img src="{{ essay.hero_image | relative_url }}" alt="" style="object-position: {{ essay.hero_position | default: 'center' }}"></a>{% endif %}
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
