---
layout: my_archive
title: "03. Logistic Regression 구조"
date: "2020-05-16"
categories: [DL, Pytorch, torchbasic]
tags: [Deep Learning, Pytorch, Basic Practice]
sidebar:
  nav: "DL"
---

이 포스트는 Github 접속 제약이 있을 경우를 위한 것이며, 아래와 동일 내용을 관련 그림 및 실행 결과와 함께 [Jupyter notebook](https://github.com/bestofbad/Pytorch-Study/blob/master/3.%20Logistic%20Regression%20%EA%B5%AC%EC%A1%B0.ipynb)으로도 보실 수 있습니다.
{: .notice--warning}


## 로지스틱 회귀(Logistic Regression)

- 이진 분류(Binary Classification)
: 시험 점수가 합격인지 불합격인지, 어떤 메일을 받았을 때 이게 정상 메일인지 스팸 메일인지를 분류

H(x)=sigmoid(Wx+b)=1/(1+e −(Wx+b)) =σ(Wx+b)

cost(H(x),y)=−[ylogH(x)+(1−y)log(1−H(x))]

## Model 함수 결과 의미

여러번의 퀴즈 결과로 시험 점수를 예측했을 때, 이를 다시 시험 합격/불합격으로 예측한다면,

1. y = Wx+b : 각 퀴즈 결과 (x 입력 변수)로 시험 점수(y)의 예상치를 계산한다.
2. y (시험 점수)는 Sigmoid 함수에서 x 값이 되며, x 축의 '0' 주변쯤 어딘가 위치한다.
3. y (시험 점수)는 x축 우측으로 갈 수록 Sigmoid 계산 결과가 1에 가까워진다.
4. y가 입력된 Sigmoid 결과는 0과 1사이의 값을 가지며, 어떤 기준(예에서는 0.5쯤)을 넘으면 1에 가깝다, 그러니 True 다... 이는 시험 합격이다... 이렇게 판단한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```

```python
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```

## Modeling 방법

1. 고전적으로 함수를 일일히 정의
2. nn.Module을 써서 함수를 간략히 정의
3. class 구조까지 사용하여 정의

** 아래에서 1, 2, 3 방법 중 한 가지만 실행.


```python
# 1-1 가중치 w 와 편향 b 를 Manual로 선언하는 방법
# Variable method로 w와 b에 aurograd 가능함을 선언

import numpy as np

# numpy array로 W, b 정의
W = np.zeros((2, 1))
b = np.zeros(1)

# numpy를 torch Tensor 변환 및 aurograd 전환
W = torch.Tensor(W)
b = torch.Tensor(b)

W = torch.autograd.Variable(W, requires_grad=True)
b = torch.autograd.Variable(b, requires_grad=True)
```

```python
# 1-2 numpy없이, 가중치 w 와 편향 b 를 torch Tensor 및 aurograd option으로 선언
W = torch.zeros((2,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)
```

```python
#1 
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    hypothesis = 1/ (1+torch.exp(-(x_train.matmul(W)+b)))
    # 또는 sigmoid 내장함수 사용
    # hypothesis = torch.sigmoid(x_train.matmul(W)+b)

    losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
    cost = losses.mean()  # 전체 오파에 대한 평균
    # 또는 cross entropy 내장함수 사용
    # cost = F.binary_cross_entropy(hypothesis, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```



```python
#2
model = nn.Sequential(nn.Linear(2,1),     # nn.Linear에 '2'는 x값 2차원, 이에 맞춰 W와 b가 랜덤 초기화 됨
                      nn.Sigmoid())
```


```python
#3
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))     # nn.Sequential함수대신 Sigmoid 직접 입력

model = BinaryClassifier()
```


```python
#2, 3 공통

optimizer = optim.SGD(model.parameters(), lr=1)


nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
      # 예측값이 0.5를 넘으면 True로 간주하며 해석 논리는 아래 '#1'의 학습결과 확인 내용 참조
        correct_prediction = prediction.float() == y_train
      # True,False 값인 prediction을 1과 0인 값으로 바꾸고, y_train 실제값들과 비교하며, 일치하면 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
      # 정확도: correct_prediction은 1 또는 0을 갖는 행렬이므로, 행렬 원소 전체 합을 행렬크기로 나눔
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))
```


## Model 학습 결과 확인

현재 W와 b는 훈련 후의 값을 가지고 있으며, 학습된 W와 b 및 예측값을 출력해 봄.

```python
#1
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis, "\n", W,"\n", b)

#이제 0.5를 넘으면 True, 넘지 않으면 False로 값을 정하여 출력.

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
```


```python
#2, 3 공통 :  훈련 후의 W와 b
print(list(model.parameters()))
```


---
- [로지스틱 회귀(Logistic Regression)](https://wikidocs.net/57810)
- [PyTorch Lecture 06: Logistic Regression](https://www.youtube.com/watch?v=GAKTBQn7yKo&list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m&index=6)
{: .notice--info}