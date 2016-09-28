---
layout: post
title:  "Market Data Categorization with Machine Learning"
date:   2016-09-28 11:00:00 -0700
categories: machine-learning
---

_The use of machine learning in financial trading is gaining traction. One problem
of using computers to trade is in the recognition of trading patterns in price data. This
paper gives a practical solution in the form of Python code, using existing machine learning
libraries. The example includes data smoothing, which is seen as an important step in
improving the accuracy of trading systems. The source code is provided to allow the
reader to experiment further._

# Introduction

If a trader can make profitable trading decisions simply by looking at trading data, then it is
reasonable to expect a computer to be able to match, or better the performance of a human
at the same task. The use of computers to automated the trading of stocks, shares and
currencies is an ongoing task, with many separate entities competing to produce favourable
results.

This guide takes the reader though the process of using the Python programming language
and the "sklearn" libraries to build an unsupervised machine learning system which can categorise
samples of market data. In doing so, the process isolates the volatility problem seen in trading
systems, and uses data smoothing to manage this problem.

## Supervised and unsupervised learning

Machine learning can be categorised into two forms; "supervised" and "unsupervised" learning.
In the former, a sample set of data is used with known outcomes. This data is used to "train"
the machine learning system. An example of supervised learning might be the use of news feeds, where some
might have a positive affect on a stock, while others may have a negative affect. Training
data would consist of sample news feeds in a suitable format, as well as the outcomes of these
feeds on the stock. The training system inputs and outputs are then used to train the system
to identify positive and negative news articles.

[Gong, Si, Fong, and Biuk-Aghai (2016)](http://www.sciencedirect.com/science/article/pii/S0957417416300483) described in detail a system for identifying typical
market patterns in data, using a supervised learning approach in their paper; ["Financial timeseries pattern matching with extended UCR Suite and Support Vector Machine"](http://www.sciencedirect.com/science/article/pii/S0957417416300483).

Unsupervised machine learning, as opposed to supervised machine learning is a method of
"training" a system with data that has not been categorised, allowing the system to group
the data as best it can. For example, given the data in figure 1, and a request to classify
this data into two discrete categories, the system would attempt to find two reasonable data
sets within the data, such as A and B shown in figure 2. Once the system has "learned" its
categories, the parameters used to define them can be used to group new data into the same
two categories.

Financial data categories, used by many traders, such as "head and shoulders", "double bottom/top" and "saucers", although [commonly referred to](https://www.stocktrader.com/2009/05/18/best-stock-chart-patterns-investing-technical-analysis/), are not
defined to the level required for computational systems. Instead of using these categories,
we shall use an unsupervised learning algorithm to let the system categorise sample financial
data.

### Figure 1: Example of unsupervised training data

![figure 1](/assets/marketdata/unsup1.png)

### Figure 2: Example of categorised training data

![figure 2](/assets/marketdata/unsup2.png)


## K-means clustering

"k-means" is a relatively simple clustering algorithm, commonly used in unsupervised machine learning.

For a given number of clusters, k, the algorithm tries to minimise the total distance between
each data point and their nearest cluster centre (see figure 2). To begin the process, k random points are
chosen. Using an iterative process, the system moves the cluster centroids to reduce the total
distance.

The nature of the random first choice and the ambiguity in some groups may sometimes lead
to different results if the algorithm is re-run.

## Data features

Machine learning data may consist of two dimensional data; rows of discrete sets of "features",
where each feature is a measurable quantity that describes the entity. Each row, or set of
features is called a "feature vector".

In our example we are interested in the movement of a particular trading symbol. Each entity
must therefore have features that describe the movement. To achieve this, we simply split the data into time slices of trading, with each feature being a price of the symbol within the time slice.

To further improve the pattern matching, the data is smoothed. The mean is calculated over
several trades, and it is these mean values which are time sliced. Figure 3 shows an example
where the mean of three trading values are calculated to form each feature. In the example, there are also three features per feature vector.

In the example code shown later, 20 features are used per feature vector, with each feature being a mean of 10 trading prices.

### Figure 3: Data Features

![figure 3](/assets/marketdata/slice.png)

# Clustering program

## Requirements

To run the code, Python 3, and the following Python packages are required:

* numpy
* matplotlib
* sklearn

## Data


A [sample of forex trading data](https://raw.githubusercontent.com/zenly/ml_samples/master/GBPUSD_asks.txt) is read in by the program. This is a list
of 20,000 consecutive GBPUSD asking prices taken from a typical trading time period.

The code takes this training data and splits it into two sections; training data and test
data. The training data is used to train the unsupervised learning algorithm and create the
parameters necessary to group the data. The training data is then grouped by the system so
that the different groupings can be shown.

The test data is then used to prove that new data, unseen by the system can be correctly
grouped by the system.

## Program overview

The computer program takes training data samples and groups them into 6 categories, choosing the data grouping criteria by looking for common data shapes. It then uses the same grouping criteria on test data samples to show that it can correctly identify the data shapes.

The program output is a set of graphs showing:

1. Raw training data
1. Training data split into discrete samples and then categorised by the system
1. Test data split into samples and catagorised by the same system


# Data Model

First, the required packages are imported. The [sklearn](http://scikit-learn.org/stable/) package
contains a varied selection of common machine learning tools to vastly reduce the implementation time of common machine learning algorithms in Python.


{% highlight python %}
import urllib.request
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
{% endhighlight %}


The program settings can be altered to modify the various settings by changing the following
variables:


{% highlight python %}
numGroups=6
numFeatures=20
dataPointsPerFeature=10
dataPointsPerRow=numFeatures*dataPointsPerFeature
dataSource = (
    'https://raw.githubusercontent.com/zenly/ml_samples/master/GBPUSD_asks.txt'
)
{% endhighlight %}


Data is taken from the dataSource URL into the "data" variable as a numpy array.


{% highlight python %}
data = np.array(
    urllib.request.urlopen(dataSource).readlines()
).astype(float)
{% endhighlight %}


The data will be divided into samples, but first it must be resized to ensure that the data exactly divides into the given sample size.


{% highlight python %}
samples = data[:len(data) - (len(data) % dataPointsPerRow)]
{% endhighlight %}


Samples may be of different orders of magnitude, but show a similar trading pattern. To
eliminate problems of magnitude in pattern matching each sample is scaled, so that they all
have zero mean and unit variance.

Market data can be choppy, especially at a macro scale. To smooth the data, it is averaged
over every ten trades (or "ticks"). The smoothing can be modified by changing the value of
the "dataPointsPerFeature" variable. The data is them split into samples of 20 smoothed data
points (controlled by the "numFeatures" variable). Each sample therefore forms a view of 200
trading ticks, feature reduced to make 20 data points.


{% highlight python %}
samples = samples.reshape(-1,dataPointsPerRow)
means = []
for sample in samples:
    means.append(preprocessing.scale(
        np.mean(
            sample.reshape(numFeatures,dataPointsPerFeature),
            axis = 1
        )
    ))
data_normalised = np.array(means)
{% endhighlight %}


The data is then split into samples of training and testing data.


{% highlight python %}
X_train, X_test = train_test_split(
    data_normalised,test_size=0.33, random_state=42
)
{% endhighlight %}


The training data is used to create the k-means clustering parameters.


{% highlight python %}
kmeans = KMeans(
    init='k-means++', n_clusters=numGroups, n_init=10
)
kmeans.fit(X_train)
{% endhighlight %}




The clustering indexes for the training data are then predicted, so that we can display the
training data samples later to give examples of the data shapes that the algorithm has found.


{% highlight python %}
Z_train = kmeans.predict(X_train)
{% endhighlight %}


The testing data is then used to test the k-means clustering. Later this can be visually
compared to the clustering of the training data to ensure the clustering algorithm is operating
correctly.


{% highlight python %}
Z_test = kmeans.predict(X_test)
{% endhighlight %}


# Results Plots

The remainder of the code is dedicated to plotting the results.

## Raw Data

The graph below shows the raw training data; asking price plotted against time. This is the
original input data before being chunked into samples, and split into training and test data.


{% highlight python %}
plt.plot(data)
plt.show()
{% endhighlight %}



![png](/assets/marketdata/output_21_0.png)


## Training Features By Group

The training data has been used to train the k-means algorithm, which created the data categories based on similarities found in the data.

The training data was then categorized into the groups that the k-means algorithm found. This was to allow us to plot the training data, grouped by category, so we can see the shapes of the various categories.


{% highlight python %}
for g in range(0,numGroups):
    print('Group',g)
    
    for i in range(0,len(X_train)):
        if Z_train[i] == g:
            
            plt.plot(X_train[i])
        
    plt.show()
{% endhighlight %}


    Group 0



![png](/assets/marketdata/output_23_1.png)


    Group 1



![png](/assets/marketdata/output_23_3.png)


    Group 2



![png](/assets/marketdata/output_23_5.png)


    Group 3



![png](/assets/marketdata/output_23_7.png)


    Group 4



![png](/assets/marketdata/output_23_9.png)


    Group 5



![png](/assets/marketdata/output_23_11.png)


As can be seen from the training data, the 6 groups do indeed show different trends.

Finally, samples from the test data are shown. These have been run through the k-means
fitting algorithm to categorise the data into the groups created during training.


{% highlight python %}
numSamples = 5

for i in range(0, numSamples):
    print('Test data',i,'categorized as group',Z_test[i])
    plt.plot(X_test[i])
    plt.show()
{% endhighlight %}


    Test data 0 categorized as group 4



![png](/assets/marketdata/output_25_1.png)


    Test data 1 categorized as group 3



![png](/assets/marketdata/output_25_3.png)


    Test data 2 categorized as group 1



![png](/assets/marketdata/output_25_5.png)


    Test data 3 categorized as group 0



![png](/assets/marketdata/output_25_7.png)


    Test data 4 categorized as group 0



![png](/assets/marketdata/output_25_9.png)


By looking at the test data plots, a visual inspection of the shapes of the graphs and their grouping, compared to the training data plots shows that the system has correctly grouped this data. 

# Conclusion

Python’s sklearn libraries provide a quick and easy way to test out machine learning on market
data. The speed at which financial market data flows, far exceeds the ability of a human to
fully register all of it. Categorising such data computationally may offer significant savings
over manual methods of data recognition, thus providing greater opportunities for trading.

This program isolates a single problem in using machine learning to predict market trends
and make successful trades; that of identifying time bound patterns in price data. The test
program allows us to explore the use of data smoothing to more accurately predict patterns.
In isolating the problem we are better able to understand how smoothing can improve
the efficiency of a trading system.

It may be that an efficient trading system could be build of machine learning "layers"; with each layer solving a discrete problem.

# The Complete Code

Shown below is the complete code. The plot output has been modified slightly, just to show the plots in one, rather than many windows.

{% highlight python %}
#!/usr/bin/env python3
#
# Example kmeans data categorisation
# Douglas Gibbons, 2016
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import urllib.request
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


# Program settings
numGroups=6
numFeatures=20
dataPointsPerFeature=10
dataPointsPerRow=numFeatures*dataPointsPerFeature
dataSource = (
  'https://raw.githubusercontent.com/zenly/ml_samples/master/GBPUSD_asks.txt'
)

# Load data from URL
data = np.array(urllib.request.urlopen(dataSource).readlines()).astype(float)


# Chop off any data that won't exactly fit a row
samples = data[:len(data) - (len(data) % dataPointsPerRow)]

# one row for all data that makes up a sample
samples = samples.reshape(-1,dataPointsPerRow)

means  = []
for sample in samples:
  means.append(preprocessing.scale(
    np.mean(
      sample.reshape(numFeatures,dataPointsPerFeature),
      axis = 1
    )
  ))  
data_normalised = np.array(means)

# Split data into training and testing data
X_train, X_test = train_test_split(
  data_normalised,test_size=0.33, random_state=42
)

# Compute k-means clustering
kmeans = KMeans(init='k-means++', n_clusters=numGroups, n_init=10)
kmeans.fit(X_train)

# Compute clustering indexes of training data
Z_train = kmeans.predict(X_train)
print(
  "Average score for k-means prediction of X_train:",
  kmeans.score(X_test)/len(X_train)
)

# Compute clustering indexes of testing data
Z_test = kmeans.predict(X_test)
print(
  "Average score for k-means prediction of X_test: ",
  kmeans.score(X_test)/len(X_test)
)

# -- Plots --

# Compute maximum number of samples in a group to correctly align data plots
maxGroupSize = max([ sum(Z_train == i) for i in range(0,numGroups) ])

cols = maxGroupSize + 1
rows = numGroups + 6

fig = plt.figure(0)
fig.canvas.set_window_title('Market Data Grouping')

# Raw training data
ax = plt.subplot2grid((rows,cols), (0,0), colspan=cols)
ax.text(
  0.5,0.5,"Raw Data",horizontalalignment='center',verticalalignment='center'
)
ax.tick_params(
  labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off'
)
ax._frameon = False

ax = plt.subplot2grid((rows,cols), (1,0), colspan=cols)
plt.plot(data)

# Training samples, sorted into their k-means clustering indexes
ax = plt.subplot2grid((rows,cols), (2,0), colspan=cols)
ax.text(
  0.5,0.5,
  "Training features by group",
  horizontalalignment='center',
  verticalalignment='center'
)
ax.tick_params(
  labelcolor=(1.,1.,1., 0.0),
  top='off', bottom='off', left='off', right='off'
)
ax._frameon = False


for g in range(0,numGroups):

  # Group header at start of row
  ax = plt.subplot2grid((rows,cols), (g+3,0))
  ax.text(
    0.5,0.5,str(g),
    horizontalalignment='center',
    verticalalignment='center'
  )
  ax.tick_params(
    labelcolor=(1.,1.,1., 0.0),
    top='off', bottom='off', left='off', right='off'
  )
  ax._frameon = False
  
  col = 1
  for i in range(0,len(X_train)):
    if Z_train[i] == g:
      plt.subplot2grid((rows,cols), (g+3,col))
      col = col + 1
      plt.plot(X_train[i])
      

# A selection of testing samples, each annotated with its given clustering index
ax = plt.subplot2grid((rows,cols), (g+4,0), colspan=cols)
# ax.set_title("Test Data Features")
ax.text(
  0.5,0,
  "Test features labeled with predicted group",
  horizontalalignment='center',verticalalignment='center'
)
ax.tick_params(
  labelcolor=(1.,1.,1., 0.0),
  top='off', bottom='off', left='off', right='off'
)
ax._frameon = False


# Show a row of test samples, as many as we can fit up to maxGroupSize to fill
testSamples = min(maxGroupSize,len(Z_test))

for i in range(0, testSamples):
  ax = plt.subplot2grid((rows,cols), (numGroups + 5,i))
  plt.plot(X_test[i])
  ax.set_title(str(Z_test[i]))
  

# Plot formating
for ax in fig.axes:
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.subplots_adjust(wspace=0, hspace=0)

fig.set_facecolor('w')
plt.show()
{% endhighlight %}
