---
layout: single
title: "Helpful with diverse matters"
permalink: /bOD/
post_categories : bOD
author_profile: true
sidebar:
  nav: "Blogroll"
---

front page for Blogs about Object Detection


{% for category in site.categories %}
  {% if category[0] == page.post_categories %}
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}
