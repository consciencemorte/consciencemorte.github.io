---
layout: page
title: Terrain
permalink: /terrain/
redirect_from: [/projets/]
intro: Protocoles, code d’interprétabilité, frameworks expérimentaux et relevés effectués sur des systèmes réels.
---

<div class="terrain-domains" aria-label="Types de relevés">
  <span>Outils d’interprétabilité</span>
  <span>Frameworks</span>
  <span>Reproductions</span>
  <span>Tests réels</span>
</div>

{% assign sorted_terrain = site.terrain | sort: 'date' | reverse %}
{% if sorted_terrain.size > 0 %}
<div class="terrain-archive">
{% for entry in sorted_terrain %}
  <article class="terrain-row">
    <div class="terrain-state"><span>{{ entry.status | default: 'expérience' }}</span><time>{{ entry.date | date: '%Y' }}</time></div>
    <div>
      <p>{{ entry.kind | default: 'Relevé expérimental' }}{% if entry.stack %} · {{ entry.stack | join: ' / ' }}{% endif %}</p>
      <h2><a href="{{ entry.url | relative_url }}">{{ entry.title }}</a></h2>
      <p>{{ entry.description }}</p>
    </div>
    {% if entry.hero_image %}<a class="terrain-thumb" href="{{ entry.url | relative_url }}" tabindex="-1" aria-hidden="true"><img src="{{ entry.hero_image | relative_url }}" alt="" style="object-position: {{ entry.hero_position | default: 'center' }}"></a>{% endif %}
    <a class="row-arrow" href="{{ entry.url | relative_url }}" aria-label="Voir {{ entry.title }}">↗</a>
  </article>
{% endfor %}
</div>
{% else %}
<div class="collection-empty collection-empty-terrain">
  <span>TERRAIN / 000</span>
  <h2>Le terrain ouvre bientôt.</h2>
  <p>Les fiches présenteront le problème, le protocole, le code, les résultats — y compris négatifs — et les conditions de reproduction. Pas de faux relevé ni de démonstration fictive en attendant les expériences réelles.</p>
</div>
{% endif %}
