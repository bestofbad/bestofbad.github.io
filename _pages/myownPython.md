---
layout: single
title: "일반적인 Python 활용 코드"
permalink: /myPy/
post_categories: myPy
author_profile: true
sidebar:
  nav: "myown"

---


{% for category in site.categories %}
  {% if category[0] == page.post_categories %}
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}
