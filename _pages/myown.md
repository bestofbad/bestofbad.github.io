---
layout: single
title: "직접 작성한 코드들"
permalink: /myown/
post_categories: myown
author_profile: true

sidebar:
  nav: "myown"
---

필요한 코드들을 만들어서 정리하는 곳입니다.


{% for category in site.categories %}
  {% if category[0] == page.post_categories %}
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}