---
layout: my_archive
title: "02. Model Coding 기본 구조"
date: "2020-05-16"
categories: [DL, Pytorch, torchbasic]
tags: [Deep Learning, Pytorch, Basic Practice]
sidebar:
  nav: "DL"
---

## Model Coding 기본 구조

- 구조 예) Linear Regression
- Regression Analysis(회귀분석) : 독립변수와 종속변수 사이에 인과관계가 존재할 때 그 관계의 통계적 유의미성을 검증하고, 또 그 관계의 정도를 분석하는 것

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
```
## Dataset은 아래 두 가지로 구성 가능


1. 입력 Data가 torch.FloatTensor 인 경우

2. 일반 행렬 Data를 pytorch의 Tensor로 변환해야 하는 경우


** import 함수가 다르며, 아래는 2번으로 진행되고, 1번은 Markdown 처리함)

```python


```

```python


```

```python


```

```python


```

```python


```

```python


```


---
- ['PyTorch로 시작하는 딥 러닝 입문' Linear Regression](https://wikidocs.net/60036)
- ['PyTorch로 시작하는 딥 러닝 입문' Logistic Regression](https://wikidocs.net/60037)
{: .notice--info}