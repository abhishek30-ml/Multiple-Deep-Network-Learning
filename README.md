# Multiple-Deep-Network-Learning-for-Emotion-Recognition
## Introduction
This project aims to classify emotion into 7 different classes using Multiple Deep Network Learning. This model is heavily inspired from [this paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/icmi2015_ChaZhang.pdf)
authored by Zhiding Yu and Cha Zhang

## Dependencies
* Python3
* Pytorch
* Numpy and Pandas

## script
### Stochastic Pooling
As suggested in the paper, I have created a class which performs Stochastic Pooling in pytorch. Unlike
max pooling which chooses the maximum response, stochastic pooling randomly samples a response based on the probability distribution obtained by normalizing the responses.
This helps us greatly to reduce overfitting

### Multiple Network
This file contains the code implementing Multiple Deep Network. As stated in the paper, The network contains five convolutional layers, three
stochastic pooling layers and three fully connected layers.The fully connected layers contains dropout , another
mechanism for randomization. These statistical randomness
reduces the risk of network overfitting.


The input to the network are the preprocessed 48 × 48
faces. Both the second and the third stochastic pooling layers include two convolutional layers prior to pooling. The
filter step height and width for all convolutional layers are
both set to 1. The nonlinear mapping functions for all convolutional layers and fully connected layers are set as rectified
linear unit (ReLU) . For stochastic pooling layers, the
window sizes are set to 3 × 3 and the strides are both set
to 2. This makes the sizes of response maps reduce to half
after each pooling layer.
The last stage of the network includes a softmax layer,
followed by a negative log likelihood loss.

On top of the single CNN model, we present a multiple
network learning framework to further enhance the performance. A common way to ensemble multiple networks is
to simply average the output responses. But a better way is to adaptively assign
different weights to each network such that the ensembled
network responses complement each other.


## Dataset
As of now, fer2013 dataset is used. The future aim of this project is to train it on fer2013 and fine tune on SFEW dataset
