---
layout: my_archive
title: "01. Model을 Class로 만드는 방법"
date: "2020-05-16"
categories: [DL, Pytorch, torchbasic]
tags: [Deep Learning, Pytorch, Basic Practice]
sidebar:
  nav: "DL"
---

## Model을 Class로 만드는 방법

### 로지스틱 회귀 모델 예

```python
model = nn.Sequential(
   nn.Linear(2, 1),      # input_dim = 2, output_dim = 1
   nn.Sigmoid()          # 출력은 시그모이드 함수를 거친다
)
```

이를 Class로 바꾸는 방법
```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
```


1. 클래스(class) 형태의 모델은 nn.Module 을 상속받음.
2. init()은 모델의 구조와 동적을 정의하는 생성자를 정의 객체가 갖는 속성값을 초기화하는 역할을 하며, 객체가 생성될 때 자동으호 호출됨.
3. super() 함수는 nn.Module 클래스의 속성들을 가지고 해당 Class를 초기화
4. foward() 함수는 Model에 입력된 학습데이터를 연산을 진행시키는 함수로, 자동으로 실행됨. 즉, model이란 이름의 객체를 생성 후, model(입력 데이터)와 같은 형식으로 객체를 호출하면 자동으로 forward 연산이 수행됨.
{: .notice--info}

따라서 PyTorch의 모든 모델은 기본적으로 다음 구조를 갖으며, PyTorch 모든 모델은 반드시 다음의 정의를 따라야 한다.

```python
import torch.nn as nn
import torch.nn.functional as F

class Model_Name(nn.Module):
    def __init__(self):
        super(Model_Name, self).__init__()
        self.module1 = ...
        self.module2 = ...
        """
        ex)
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        """

    def forward(self, x):
        x = some_function1(x)
        x = some_function2(x)
        """
        ex)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        """
        return x
```

---
- [Linear Regression](https://wikidocs.net/60036)
- [Logistic Regression](https://wikidocs.net/60037)
{: .notice--info}