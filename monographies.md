---
layout: page
title: Monographies
permalink: /monographies/
redirect_from: [/observations/]
intro: Publications techniques, études longues et monographies sur les mécanismes, les représentations et la sécurité des systèmes d’intelligence artificielle.
---

{% assign sorted_monographies = site.monographies | sort: 'date' | reverse %}
<ul class="archive-list">
{% for monographie in sorted_monographies %}
  <li>
    <span class="archive-kind">{{ monographie.status | default: monographie.type | default: 'monographie' | upcase }}</span>
    <div class="archive-entry">
      <a href="{{ monographie.url | relative_url }}">{{ monographie.title }}</a>
      <span>{% if monographie.topics %}{{ monographie.topics | join: ' · ' }}{% else %}{{ monographie.categories | join: ' · ' }}{% endif %}{% if monographie.level %} — {{ monographie.level }}{% endif %}</span>
    </div>
    <time datetime="{{ monographie.date | date_to_xmlschema }}">{{ monographie.date | date: '%d.%m.%Y' }}</time>
    {% if monographie.hero_image %}<a class="archive-thumb" href="{{ monographie.url | relative_url }}" tabindex="-1" aria-hidden="true"><img src="{{ monographie.hero_image | relative_url }}" alt="" style="object-position: {{ monographie.hero_position | default: 'center' }}"></a>{% endif %}
  </li>
{% endfor %}
</ul>
