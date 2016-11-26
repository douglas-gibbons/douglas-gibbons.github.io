---
layout: post
title:  "Machine Learning Terminology"
date:   2016-10-17 13:00:00 -0700
categories: machine-learning
---

An incomplete summary of some of the many terms used in the trade.

### Convolutional Neural Networks (CNNs / ConvNets)

CNNs are neural networks, where the design of the various layers of the network are based around the desire to gain insight from the structure of the incoming data. CNNs are used primarily for images, allowing the image to be split up, so that the system can more easily recognise patterns in the sections.

### Deep Learning

"Deep Learning" is a subset of machine learning. A neural network used for machine learning has "layers"; an input layer, one or more hidden layers, and an output layer. Each "layer" is really just a set of parameters that are tuned during the training phase of the system. The more "layers", the more complex the decisions that the system can achieve. "Deep Learning" is the term used when more than one hidden layer is employed.

### Machine Learning

Using data to feed a set of algorithms, automatically tweaking the functions of those algorithms to best fit the data. Once the system has "learned" to fit the data, it can be used on unseen data to make predictions or classify that data. With __"supervised learning"__ the data classifications are already known and are part of the training.  With __"unsupervised learning"__ it is up to the machine to classify the data.

### Softmax Regression

Logistic regression, capable of handling multiple classes (not just binary/boolean). See the [UFLDL Tutorial from Stanford](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/).  Each output is the probability of that value being the truth state, so the sum of the values add up to one. i.e. 100% probability that it has to be _one_ of the outputs!

"Softmax" can be used as the output layer of a neural network.

### Stochastic Training

Using small batches of data, usually picked from a set, where the order of the data has been randomised. This is quicker than using the whole dataset at a time for training.

### TensorFlow

A Deep Learning) library with Python and C++ APIs.  [TensoreFlow](https://www.tensorflow.org/)
