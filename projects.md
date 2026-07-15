---
layout: page
title: Protocoles & Relevés de terrain
permalink: /projets/
intro: Protocoles, code d’interprétabilité, frameworks expérimentaux et relevés effectués sur des systèmes réels.
---

<div class="project-domains" aria-label="Types de projets">
  <span>Outils d’interprétabilité</span>
  <span>Frameworks</span>
  <span>Reproductions</span>
  <span>Tests réels</span>
</div>

{% assign sorted_projects = site.projects | sort: 'date' | reverse %}
{% if sorted_projects.size > 0 %}
<div class="project-archive">
{% for project in sorted_projects %}
  <article class="project-row">
    <div class="project-state"><span>{{ project.status | default: 'expérience' }}</span><time>{{ project.date | date: '%Y' }}</time></div>
    <div>
      <p>{{ project.kind | default: 'Projet expérimental' }}{% if project.stack %} · {{ project.stack | join: ' / ' }}{% endif %}</p>
      <h2><a href="{{ project.url | relative_url }}">{{ project.title }}</a></h2>
      <p>{{ project.description }}</p>
    </div>
    <a class="row-arrow" href="{{ project.url | relative_url }}" aria-label="Voir {{ project.title }}">↗</a>
  </article>
{% endfor %}
</div>
{% else %}
<div class="collection-empty collection-empty-projects">
  <span>TERRAIN / 000</span>
  <h2>Le terrain ouvre bientôt.</h2>
  <p>Les fiches présenteront le problème, le protocole, le code, les résultats — y compris négatifs — et les conditions de reproduction. Pas de faux dépôt ni de démonstration fictive en attendant les vrais projets.</p>
</div>
{% endif %}
