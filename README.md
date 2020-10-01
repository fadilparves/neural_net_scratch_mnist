# Neural Network From Scratch

TL;DR build ANN from scratch and use the algorithm to train and test on mnist fashion data.

### Introduction
In the era of unstructured data such as images being one of the most collected data through applications such as facebook, instagram, pinterest and other such apps has really caused more images data to be available for DS to utilize for AI purpose.

### Neural Network
In this repo, the repo has 50 hidden layers, with 28 * 28 input layer and 10 output layer as the mnist has 10 to be predicted classes

### Basic neural network architecture
![ann_basic](https://miro.medium.com/max/1000/1*ub-ifcgdi9xgryqvo0_GRA.png)

*Image courtesy of [link](https://mc.ai/my-notes-on-neural-networks-2/)*

* __Input layer__ is number of neurons as what we are feeding in into the neural network - in mnist cases the number of features of each image 28 * 28 = 784
* __Hidden layer__ is where the weight and bias is applied to train the model in learning and predicting the class for classification or value for regression
* __Output layer__ is to return the predicted classes and the probability of each class to input, and take the highes proba value as final output

### Use case
Predict MNIST fashion classes

[Neural Network Code](https://github.com/fadilparves/neural_net_scratch_mnist/blob/master/nueral_net.py)

[Mnist predictor](https://github.com/fadilparves/neural_net_scratch_mnist/blob/master/mnist_nn_classifier.py)

### Model Evaluation
Without scaling
``` Training accuracy: 54% | Test accuracy: 46% ```

![Without_scaled](https://github.com/fadilparves/neural_net_scratch_mnist/blob/master/output/without_scaled.png)

- We can see that at one point the model is not getting better and the error is increasing and decreasing, the loss is not flattening

With scaling (to flatten the loss)
``` Training accuracy: 96% | Test accuracy: 86% ```

![With_scaled](https://github.com/fadilparves/neural_net_scratch_mnist/blob/master/output/with_scaled.png)

Orange line is with scaled input data, and we can see the training and test accuracy increased a lot and the loss is flatten now!

### How to use
1. Clone this repo
2. Make sure you have all the libs install such as pandas, numpy, matplotlib, seaborn, scikit-learn
3. Run `mnist_nn_classifier.py`
4. Enjoy the result

## Contributor
<a href="https://github.com/fadilparves/neural_net_scratch_mnist/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=fadilparves/neural_net_scratch_mnist" />
</a>
