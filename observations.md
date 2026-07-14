---
layout: page
title: Observations
permalink: /observations/
intro: Études longues sur les mécanismes, les représentations et les comportements des systèmes d’intelligence artificielle.
---

<ul class="archive-list">
{% for post in site.posts %}
  <li>
    <span class="archive-kind">{{ post.categories | first | upcase }}</span>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%d.%m.%Y' }}</time>
  </li>
{% endfor %}
</ul>

