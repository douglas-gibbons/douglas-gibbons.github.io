---
layout: post
title:  "AWS vs Google Machine Learning"
date:   2016-11-27 10:00:00 -0700
categories: machine-learning
---

Previously I [reviewed Amazon's Machine Learning platform](http://www.douglas-gibbons.com/machine-learning/2016/11/26/aws-machine-learning.html) by testing it against the [MNIST Database](http://yann.lecun.com/exdb/mnist/).

Google also offers [machine learning for their cloud platform](https://cloud.google.com/products/machine-learning/). Their [quickstart tutorial](https://cloud.google.com/ml/docs/quickstarts/) focuses on the MNIST dataset like my Amazon review, so it's easy enough to run their code and directly compare the AWS and Google machine learning offerings.

I picked up their tutorial, and tweaked the parameters slightly, so that the results would reach the same accuracy as my test of the AWS platform.  This was achieved simply by increasing the max_steps on to 4000. The [code is here](https://github.com/douglas-gibbons/gcloud_ml_mnist).

Now we have a direct comparison.

## Running the Google Code

Google's machine learning offering the user to be proficient in [TensorFlow](https://www.tensorflow.org/). Their system expects the code to be written a certain way too, so it's not always as simple as taking existing TensorFlow code and "putting it in the cloud". The TensorFlow library runs C++ behind the scenes. It requires the user to first define a data flow graph of all the steps to run, and only when that map is defined are the variable place-holders filled with data, so the map can be run.  This makes for some "interesting" and very un-pythonic code.

I could go on criticizing TensorFlow, believe me. I've only just got started. The problem is though, that it's not just good, it's excellent. Sure, it'll give you a few evenings of swearing at the computer, but once you get the concept, there's nothing quite like it for building complex models. And it's fast. Really fast!  This goes some way to explaining the results below.


## Test Results of MNIST Database machine learning problem

## AWS ML Platform

* __Cost__ About $2.80 (USD) for AWS ML charges
* __Computation Time__ 2.5 hours
* __Easy of use__ Easy
* __Accuracy__ 91.6%

## Google Cloud ML Platform

* __Cost__ $0.10 (USD)
* __Computation Time__ 7 minutes, 20 seconds
* __Easy of use__ Difficult
* __Accuracy__ 91.9%

## Was this a fair test?

We are not quite comparing apples with apples here. The Amazon test relied on my code, which I knocked up in half a day with very little tuning. However, the Google test relied on Google's own code, with a deep neural network, tuned for the MNIST problem. Google was going to win here!

It should also be noted that the TensorFlow code could have been optimized further to produce far greater accuracy on the dataset. Instead of tuning the code further, I tried to keep the accuracy of both tests the same, so I could more directly compare run times and costs.

## Review

Looking at the results, one might think that Google is the clear winner here. It costs less and the run time is shorter. Google is the racing car, and AWS is the tractor.  However, sometimes you don't want a team of engineers to tweak your system, just so it can race a few laps around a track. Sometimes, you want to just get in, start it up and go plough that machine learning field.

Possibly I've over-stretched the analogy here. The point is, that the costs above don't take into account the engineering time required to build the model, which on the Google system may be far greater.  If your needs are simple, AWS will get you there, and the journey will be simple and straightforward. Anything beyond simple, and Google Cloud is your friend.

