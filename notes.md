---
layout: page
title: Notes
permalink: /notes/
intro: Revues de littérature, lectures commentées et notes bibliographiques sur les travaux récents.
---

<ul class="archive-list">
{% assign sorted_notes = site.notes | sort: 'date' | reverse %}
{% for note in sorted_notes %}
  <li>
    <span class="archive-kind">{{ note.status | default: 'NOTE' | upcase }} {{ forloop.index | prepend: '0' }}</span>
    <div class="archive-entry">
      <a href="{{ note.url | relative_url }}">{{ note.title }}</a>
      {% if note.topics %}<span>{{ note.topics | join: ' · ' }}</span>{% endif %}
    </div>
    <time datetime="{{ note.date | date_to_xmlschema }}">{{ note.date | date: '%d.%m.%Y' }}</time>
    {% if note.hero_image %}<a class="archive-thumb" href="{{ note.url | relative_url }}" tabindex="-1" aria-hidden="true"><img src="{{ note.hero_image | relative_url }}" alt="" style="object-position: {{ note.hero_position | default: 'center' }}"></a>{% endif %}
  </li>
{% endfor %}
</ul>
