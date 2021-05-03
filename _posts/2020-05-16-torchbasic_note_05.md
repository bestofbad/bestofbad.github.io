---
layout: my_archive
title: "05. MNIST by Softmax"
date: "2020-05-16"
categories: [DL, Pytorch, torchbasic]
tags: [Deep Learning, Pytorch, Basic Practice]
sidebar:
  nav: "DL"
---

이 포스트는 Github 접속 제약이 있을 경우를 위한 것이며, 아래와 동일 내용을 관련 그림 및 실행 결과와 함께 [Jupyter notebook](https://github.com/bestofbad/Pytorch-Study/blob/master/5.%20MNIST%20by%20Softmax.ipynb)으로도 보실 수 있습니다.
{: .notice--warning}

## MNIST by Softmax

```python
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
```

```python
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", device)

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
```

### MNIST Data

```python
# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
```

### Dataset 구성 확인

```python
# MNIST dataset 크기
print("train data 크기:", len(mnist_train), "      ", "test data 크기:",len(mnist_test))

# i번째 data확인
i=5343
print(type(mnist_train[i]), len(mnist_train[i]))
```
위에서 train data가 60000개 이며, 그 중에 i번째 성분을 골라서 확인해보니, tuple로 2개의 요소가 묶여져 있음을 확인

```python
first, second= mnist_train[i]
print("first type :", type(first), "       first len :", len(first), "         first size :", first.size())
print("second type :", type(second), "      second :", second)
```
2개의 tuple을 분리하여 first 성분을 확인해보니, 1 x 28 x 28 행렬(image)인 Tensor이고, second는 이 행렬의 Label 정보(정답)을 가지고 있다.

```python
# image가 3차원이므로, 의미없는 1을 없애고 2차원으로 만들기 위해 view(28, 28) 진행하여 image확인
plt.imshow(first.view(28, 28), cmap='Greys', interpolation='nearest')
```

### Data loading

```python
data_loader = DataLoader(dataset=mnist_train, batch_size=100, shuffle=True, drop_last=True)

print(len(data_loader), type(data_loader))
```
- 전체 크기가 60000개인 data를 100개의 batch_size로 나누므로, data_loader는 600개 data로 구성됨. 이 때, data_loader의 한개 data는 자동으로 '100 x input차원'의 행렬이 된다.

### Modeling

1. Linear Classification 함수 1개 사용
2. class로 MyModel을 만들고, Linear Layer 4개, Activation Function으로 sigmoid와 softmax 사용.
3. Linear Layer 4개, Activation Function으로 Relu 사용.

```python
# 1.
# softmax의 Linear 함수를 위해 28 x 28 image를 1차원으로 바꾸고, 10개 숫자 분류를 해야하므로,
mymodel = nn.Linear(784, 10, bias=True).to(device)
```

```python

# 2. 결과적으로 Linear Layer, Sigmoid 수가 증가할 수록, 계산 속도가 느려지고,
#    1개 비해 3개 정도에서  cost가 0.02 가량 감소했으나, 4개층에서는 오히려 1.5배가량 증가됨

import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.myL1 = nn.Linear(784, 580, bias=True).to(device)   # 580,380,180은 임의 분류 수
        self.myL2 = nn.Linear(580, 380, bias=True).to(device) 
        self.myL3 = nn.Linear(380, 180, bias=True).to(device)       
        self.myL4 = nn.Linear(180, 10, bias=True).to(device)       
        
        self.sigmoid = torch.nn.Sigmoid().to(device)

           
    def forward(self, x):
        x = self.sigmoid (self.myL1(x))           # 뷴류기로 softmax를 2회 이상쓰면, cost 감소 안함.
        x = self.sigmoid (self.myL2(x))           # softmax의 output이 1이하의 소수라서 backward 불리한 듯.
        x = self.sigmoid (self.myL3(x))           
        prediction = torch.log(F.softmax(self.myL4(x), dim=1))   # cost 계산을 위해 log 취함
        
        return prediction

mymodel = MyModel()
```

```python
# 3. Activation Funciotn : Relu , 결과적으로 학습이 확연히 뛰어남.

import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.myL1 = nn.Linear(784, 580, bias=True).to(device)   # 380은 임의 분류 수
        self.myL2 = nn.Linear(580, 380, bias=True).to(device)   # 100은 임의 분류 수
        self.myL3 = nn.Linear(380, 180, bias=True).to(device)       
        self.myL4 = nn.Linear(180, 10, bias=True).to(device)       
        
           
    def forward(self, x):
        x = F.relu(self.myL1(x))
        x = F.relu(self.myL2(x))
        x = F.relu(self.myL3(x))
        prediction = self.myL4(x)         # 마지막엔 activation 필요없음, cost Function에 softmax 포함됨.
        
        return prediction

mymodel = MyModel()
```

### 학습

```python
# optimizer 정의
optimizer = torch.optim.SGD(mymodel.parameters(), lr=0.1)

# Model 별 Cost 정의
#1, 3
criterion = nn.CrossEntropyLoss().to(device)   # 소프트맥스 함수 포함됨
#2. 아래 epoch for문에서 # 주석 삭제
```

```python
training_epochs = 15

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # 배치 크기가 100이므로 X는 (100, 784)
        X = X.view(-1, 28 * 28).to(device)
        # Y는 0 ~ 9의 정수, cost 함수를 원-핫 인코딩이 필요없는 Activation Function를 사용함.
        Y = Y.to(device)

        hypothesis = mymodel(X)
        #1, 3
        cost = criterion(hypothesis, Y)
        #2
        #cost = F.nll_loss(hypothesis, Y).to(device)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
```

### Test Data

```python
# 서두에서 mnist_test가 10000개임을 확인했고, j번째 data확인
j=540
print(type(mnist_test[j]), len(mnist_test[j]))
```

```python
test_first, test_second= mnist_test[j]
print("first type :", type(test_first), "       first len :", len(test_first), "         first size :", test_first.size())
print("second type :", type(test_second), "      second :", test_second)

plt.imshow(test_first.view(28, 28), cmap='Greys', interpolation='nearest')
```
### Model 검증
- 위에서 mnist_test[j]의 image값인 test_first와 Label인 test_second로 model 검증

```python
# 우선 위 mnist_test[j]를 검증
X_test1 = test_first.view(-1, 28 * 28).float().to(device)     # model 함수 input 차원과 맞춤
Y_test1 = test_second

prediction = torch.argmax(mymodel(X_test1),1)
print(prediction)

print(prediction == Y_test1)
```

```python
# minist_test 모든 data에 대해 정확도 통계 계산
# test_data, test_labels method로 전체 data 계산

with torch.no_grad():             # gradient 계산 않음.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = mymodel(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
```

---
- [소프트맥스 회귀로 MNIST 데이터 분류하기](https://wikidocs.net/60324)
- [PyTorch Lecture 09: Softmax Classifier](https://www.youtube.com/watch?v=lvNdl7yg4Pg&list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m&index=10)
{: .notice--info}