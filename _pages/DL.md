---
layout: my_archive
title: "Deep Learning"
permalink: /DL/
post_categories: DL
author_profile: true
sidebar:
  nav: "DL"

---

관심있는 Deep Learning 분야를 공부하면서 정리한 포스트입니다.

This page is a notebook for self-study about Deep Learning I have been interested in.



{% for category in site.categories %}
  {% if category[0] == page.post_categories %}
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}