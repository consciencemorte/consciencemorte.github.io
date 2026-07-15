---
layout: page
title: Publications & Monographies
permalink: /observations/
intro: Publications techniques, études longues et monographies sur les mécanismes, les représentations et la sécurité des systèmes d’intelligence artificielle.
---

<ul class="archive-list">
{% for post in site.posts %}
  <li>
    <span class="archive-kind">{{ post.status | default: post.type | default: 'observation' | upcase }}</span>
    <div class="archive-entry">
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <span>{% if post.topics %}{{ post.topics | join: ' · ' }}{% else %}{{ post.categories | join: ' · ' }}{% endif %}{% if post.level %} — {{ post.level }}{% endif %}</span>
    </div>
    <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%d.%m.%Y' }}</time>
  </li>
{% endfor %}
</ul>
