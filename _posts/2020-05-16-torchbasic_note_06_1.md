---
layout: my_archive
title: "Pytorch 06-1. Convolutional Neural Network 개념"
date: "2020-05-16"
categories: [DL, Pytorch, torchbasic]
tags: [Deep Learning, Pytorch, Basic Practice, CNN, Convolutional Neural Network]
sidebar:
  nav: "DL"
---

## Convolutional Neural Network

![image-center]({{ "/assets/images/Fig_CNN6_1.png" | relative_url }}){: .align-center}

### 기존 신경망과 CNN의 Parameter 수 비교
- 기존은 입력 이미지가 Parameter 크기에 직접 영향을 미쳤으나, CNN은 훨씬 작은 Kernel 크기가 영향을 준다.  
  단, 은닉층의 갯수에 의한 비례 증가는 동일.

![image-center]({{ "/assets/images/Fig_CNN6_2.png" | relative_url }}){: .align-center}

### Feature Map 크기
![image-center]({{ "/assets/images/Fig_CNN6_3.png" | relative_url }}){: .align-center}  

### Channel 수
- k 개 Channel의 Input에는 k Channel의 1개 Kernel이 필요하고, Output인 Feature mape은 1개의 Channel을 가진다.
![image-center]({{ "/assets/images/Fig_CNN6_4.png" | relative_url }}){: .align-center}

따라서 Feature Map의 Channel 수는 Kernel 갯수로 결정된다.
![image-center]({{ "/assets/images/Fig_CNN6_5.png" | relative_url }}){: .align-center}

![image-center]({{ "/assets/images/Fig_CNN6_6.png" | relative_url }}){: .align-center}


### Pooling
- 특성 맵을 다운샘플링하여 특성 맵의 크기를 줄이는 연산
![image-center]({{ "/assets/images/Fig_CNN6_7.png" | relative_url }}){: .align-center}

- Pooling 연산에도 Kernel과 Stride가 필요하나, 학습해야하는 가중치가 없고, Channel 수가 유지됨.
- Pooling에는 위 그림처럼 최대값을 추출하는 max pooling과 평균값을 추출하는 average pooling이 있음.

### Subsampling (downsampling)
![image-center]({{ "/assets/images/Fig_CNN6_8.png" | relative_url }}){: .align-center}

- Convolutional 층과 subsample 층 교대로 반복됨.
![image-center]({{ "/assets/images/Fig_CNN6_9.png" | relative_url }}){: .align-center}

- Input Image에 Convolution 과정을 수행하는 것은 단지 수치적 연산이다.  
  이 numerical result에 Pooling을 하는 것은 특성 추출 과정인데, linear Model과 Activation Function을 통해 확률값(probabilities)으로 변환한 것이 바로 subsample이다.   
  따라서 Convolution 층은 numerical matrices인 반면, Subsample은 항상 featured pictures이다.


---
- [Deep Learning 05: Talk about Convolutional Neural Networks（CNN）](https://ireneli.eu/2016/02/03/deep-learning-05-talk-about-convolutional-neural-network%EF%BC%88cnn%EF%BC%89/)
- [합성곱과 풀링(Convolution and Pooling)](https://wikidocs.net/62306)
{: .notice--info}