---
layout: post
title:  "AWS Machine Learning"
date:   2016-11-05 13:00:00 -0700
categories: machine-learning
---

Amazon launched their machine learning platform in April 2015 - [Amazon Machine Learning](https://aws.amazon.com/machine-learning/).  In typical AWS style it provides a point-and-click web interface as well as an API, which has been integrated into several libraries already, including Python's [boto3](https://boto3.readthedocs.io/en/latest/reference/services/machinelearning.html).

They have made some interesting design choices, choosing to treat Machine Learning as a kind of shrink-wrapped product that you just use, rather than giving the user full control. This means that it's easy to get started, when compared to building models in something like TensorFlow. However, it's not as customizable, so don't expect to build complicated convolutional neural networks. This is big computing, but simple.

Amazon give some examples of using the product focusing on marketing data, but how does it compare when faced with the technical challenge of handwriting recognition?  I tested it out on the [MNIST Database](http://yann.lecun.com/exdb/mnist/), which is a common data set, often used to test out machine learning systems.

## The Code

I opted to build the model using Python, which is the most common language for playing around with machine learning. Probably. The code can be found in [GitHub](https://github.com/douglas-gibbons/aws-ml-mnist), and can be run using the instructions below.

The files are:

* download_mnist.py - Script to download MNIST data locally and save the data in a python pickle file
* aws_upload.py - Script to upload MNIST data to Amazon S3
* aws_model.py - Script to build to the AWS ML model
* aws_use_model.py - Script to run predictions
* aws_check_results.py - Script to download a results file and measure its performance
* data.py - class file for storing MNIST data
* image.schema - Data schema for MNIST images
* bucket_policy.json - Example policy file for an S3 bucket
* config.py - Configurations variables for the user to change before running the scripts
* recipe.json - AWS ML recipe file


## Running the application

### Pre-requisites

The code uses python v3, and will need the following libraries:

* boto3
* numpy

You'll also need full access to an AWS account and some access keys

It was tested on a Ubuntu 16.04 machine, but it should run on most systems with Python 3.

### Setting Up

Clone the code from [github.com/douglas-gibbons/aws-ml-mnist](https://github.com/douglas-gibbons/aws-ml-mnist)

Create and AWS S3 bucket and set up a bucket policy. The policy should allow access to AWS ML components and to the scripts to upload and download files. An example policy (which will need editing) can be found in the code directory, and is called "bucket_policy.json".

Create directories "data" and "output" inside the bucket

Set up ~/.aws/credentials with credentials and a region suitable for ML. For example:

```
[default]
aws_access_key_id = YOUR_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
region=us-east-1
```

Edit ```config.py``` and change the data_bucket name and check the bucket region is correct.

### Running the Code

The various python files are designed to run in order:

1. Download the MNIST data locally by running ```./download_mnist.py```
1. Format the data and upload to S3 ```./aws_upload.py```
1. Build and run the AWS ML model ```./aws_model.py```
1. Wait for the model to complete building by checking the [web interface](https://console.aws.amazon.com/machinelearning/home?region=us-east-1#/).
1. Run ```./aws_use_model.py``` to use the model to make predictions on the test images
1. Wait for the predictions to complete (again, check the [web interface](https://console.aws.amazon.com/machinelearning/home?region=us-east-1#/)).
1. Download the prediction output and evaluate the predictions ```./aws_check_results.py```

The output should show the accuracy of the machine learning predictions to be about 91%.

## The Results

For the MNIST trial:

Cost: About $2.80
Time: About 3 hours to run
Easy of use: Very
Accuracy: 90%

### Disadvantages

The AWS ML system does not offer much in the way of tweaking the machine learning. Don't expect the level of recognition performance you might get with custom convolutional neural networks. However, for most tasks it should perform well enough.

### Advantages

AWS ML integrates well with other AWS products. It can take its data from Redshift, RDS, S3 or EMR, as well as real time inputs from Kinesis and of course Lambda and EC2 instances.  If you're an AWS shop and want to integrate machine learning into an existing structure, this is certainly worth a try.

The simplicity of the product makes it easy to get started and quick to get results.


## In Conclusion

AWS ML provides a very simple and easy-to-use interface into the world of machine learning. It also plays well with all those other AWS products that we've grown to depend on. It's out-of-the box performance is enough to satisfy most common uses, but it is limited in terms of tuning more complex problems.

