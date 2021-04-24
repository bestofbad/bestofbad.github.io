---
layout: single
title: "Basic Practice in Pytorch"
permalink: /torchbasic/
author_profile: true
sidebar:
  nav: "DL"

---

front page

Basic Practice in Pytorch


{% for category in site.categories %}
  {% if category[0] == "torchbasic" %}
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}
