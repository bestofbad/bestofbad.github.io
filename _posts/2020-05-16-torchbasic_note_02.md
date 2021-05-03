---
layout: my_archive
title: "02. Model Coding 기본 구조"
date: "2020-05-16"
categories: [DL, Pytorch, torchbasic]
tags: [Deep Learning, Pytorch, Basic Practice]
sidebar:
  nav: "DL"
---

이 포스트는 Github 접속 제약이 있을 경우를 위한 것이며, 아래와 동일 내용을 관련 그림 및 실행 결과와 함께 [Jupyter notebook](https://github.com/bestofbad/Pytorch-Study/blob/master/2.%20Model%20Coding%20%EA%B8%B0%EB%B3%B8%20%EA%B5%AC%EC%A1%B0.ipynb)으로도 보실 수 있습니다.
{: .notice--warning}

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
{: .notice--info}

(** import 함수가 다르며, 아래는 1, 2번 중 하나 진행)

```python
#1
from torch.utils.data import TensorDataset
x_trains = torch.FloatTensor([[73, 80, 75],
                              [93, 88, 93],
                              [89, 91, 90],
                              [96, 98, 100],
                              [73, 66, 70]])
y_trains = torch.FloatTensor([[152], [185], [180], [196], [142]])
dataset = TensorDataset(x_trains, y_trains)
```

```python
#2
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()
```
## Model 정의 방법

본 예시의 경우, model = nn.Linear(3,1)로 간단히 정의 가능하나, 향후 복잡한 문제 해결을 위해 class로 구현함

```python
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
        
    def forward(self, x):
        return self.linear(x)

model = MultiVariateLinear()

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)
```

## Model 학습

```python
nb_epochs = 200
for epoch in range(nb_epochs +1):
    for batch_idx, samples in enumerate(dataloader):
#        print(batch_idx)     # enumerate 함수 작동을 보려면 주석 해제
#        print(samples)
        x_train, y_train = samples
        
        prediction = model(x_train)
        
        cost = F.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print('Epoch {:4d}/{} Batch{}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()
                ))
```
## 학습된 Model로 예측

```python
new_var = torch.FloatTensor([[73, 80,75]])
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
```
---
- [Multivariable Linear regression](https://wikidocs.net/54841)
- [클래스로 파이토치 모델 구현하기](https://wikidocs.net/60036)
{: .notice--info}