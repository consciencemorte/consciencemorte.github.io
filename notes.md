---
layout: page
title: Notes de laboratoire
permalink: /notes/
intro: Lectures, hypothèses et schémas intermédiaires. Des traces courtes, publiées avant que l’intuition ne devienne une étude.
---

<ul class="archive-list">
{% assign sorted_notes = site.notes | sort: 'date' | reverse %}
{% for note in sorted_notes %}
  <li>
    <span class="archive-kind">NOTE {{ forloop.index | prepend: '0' }}</span>
    <a href="{{ note.url | relative_url }}">{{ note.title }}</a>
    <time datetime="{{ note.date | date_to_xmlschema }}">{{ note.date | date: '%d.%m.%Y' }}</time>
  </li>
{% endfor %}
</ul>

