---
layout: post
title:  "Summations, Matrices and Python"
date:   2016-10-01 13:00:00 -0700
categories: machine-learning
---

Matrix mathematics is used to great affect in evaluating machine learning algorithms.  Some of the built-in libraries in many programming languages automatically run such calculations over multiple CPUs or GPUs to reduce the computation time.

Below is a quick introduction to matrix multiplication with the _Python_ programming language, and an example of how it's used in machine learning.

We're going to use Python's [numpy](http://www.numpy.org/) library:


{% highlight python %}
import numpy as np
{% endhighlight %}


First we can create some numpy arrays. Here are several ways of creating them:


{% highlight python %}
a = np.arange(12).reshape((4, 3))
print("The shape of matrix a is",a.shape)
print(a)
{% endhighlight %}


    The shape of matrix a is (4, 3)
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]



{% highlight python %}
b = np.array([[1,2,3,4],[5,6,7,8]])
print("The shape of matrix b is",b.shape)
print(b)
{% endhighlight %}


    The shape of matrix b is (2, 4)
    [[1 2 3 4]
     [5 6 7 8]]



{% highlight python %}
c = np.array([range(9)]).reshape(3,3)
print("The shape of matrix c is",c.shape)
print(c)
{% endhighlight %}

    The shape of matrix c is (3, 3)
    [[0 1 2]
     [3 4 5]
     [6 7 8]]


## Matrix Transpose

The __transpose__ of a matrix turns rows to columns, and columns to rows. Written as \\(A^T\\). Think of it as a reflection of the matrix along it's largest diagonal axis. This can be done quite simply with a numpy array:


{% highlight python %}
a.T
{% endhighlight %}





    array([[ 0,  3,  6,  9],
           [ 1,  4,  7, 10],
           [ 2,  5,  8, 11]])



## Hadamard product
__Hadamard product__, also known as the Schur product, entrywise, or element wise multiplication, multiplies matrices together on an element-by-element basis.

e.g.

\begin{equation}
\begin{bmatrix}
 1 & 2 & 3 &#92;&#92;
 4 & 5 & 6 &#92;&#92;
 7 & 8 & 9 &#92;&#92;
\end{bmatrix} \times \begin{bmatrix}
 a & b & c &#92;&#92;
 d & e & f &#92;&#92;
 g & h & i &#92;&#92;
\end{bmatrix} = \begin{bmatrix}
 1a & 2b & 3c &#92;&#92;
 4d & 5e & 6f &#92;&#92;
 7g & 8h & 9i &#92;&#92;
\end{bmatrix}
\end{equation}

Element wise multiplication also works for scalers and matrices:

\begin{equation}
2 \times \begin{bmatrix}
 1 & 2 & 3 &#92;&#92;
 4 & 5 & 6 &#92;&#92;
 7 & 8 & 9 &#92;&#92;
\end{bmatrix} = \begin{bmatrix}
 2 & 4 & 6 &#92;&#92;
 8 & 10 & 12 &#92;&#92;
 14 & 16 & 18 &#92;&#92;
\end{bmatrix}
\end{equation}

## Dot Product
The matrix __dot product__ can be thought of as a system for calculating distances of vectors, but we're going to use it for more general summations.

To calculate the dot product for two matrices (A and B), calculate the answer a row at a time. For the first cell, move across the first row of A and down the first column of B, adding the products of each pair.  For the _second_ cell in the answer row, move across the first row of A and the _second_ column of B.  Do this for every column of B.  Now repeat for the rest of the rows in A to make more rows in the answer matrices.

This is much clearer with some examples:

\begin{equation}
\begin{bmatrix}
 1 & 2 &#92;&#92;
 3 & 4 &#92;&#92;
\end{bmatrix} \cdot \begin{bmatrix}
 a & b &#92;&#92;
 c & d &#92;&#92;
\end{bmatrix} = \begin{bmatrix}
 1a+2c & 1b+2d &#92;&#92;
 3a+4c & 3b+4d &#92;&#92;
\end{bmatrix} 
\end{equation}

\begin{equation}
\begin{bmatrix}
 1 &#92;&#92;
 2 &#92;&#92;
 3 &#92;&#92;
\end{bmatrix} \cdot \begin{bmatrix}
 a & b & c &#92;&#92;
\end{bmatrix} = \begin{bmatrix}
  1a & 1b & 1c &#92;&#92;
  2a & 2b & 2c &#92;&#92;
  3a & 3b & 3c
\end{bmatrix} 
\end{equation}

\begin{equation}
\begin{bmatrix}
 1 & 2 & 3 &#92;&#92;
\end{bmatrix} \cdot \begin{bmatrix}
 a &#92;&#92;
 b &#92;&#92;
 c &#92;&#92;
\end{bmatrix} = \begin{bmatrix}
  1a + 2b + 3c &#92;&#92;
\end{bmatrix} 
\end{equation}



The result has the same number of rows as the first matrix, and the number of columns of the second. Dot product multiplication is only valid when the number of columns in the first matrix is equal to the number of rows in the second matrix


Suppose we want to add together the natural numbers from 1 to 4.  We could write this in many ways:

\begin{equation}
1 + 2 + 3 + 4 = \frac{4(4+1)}{2} = \sum\limits_{i=1}^{4} i = \begin{bmatrix}1&2&3&4\end{bmatrix} \cdot  \begin{bmatrix}
1 &#92;&#92;
1 &#92;&#92;
1 &#92;&#92;
1 &#92;&#92;
\end{bmatrix}
\end{equation}

Looking at the last method, we have one column in the first matrix multiplied by one row in the second matrix.  That would give is a 1 x 1 matrix as the answer.

Working through, it would be \begin{equation} 1 \times 1 + 2 \times 1 + 3 \times 1 + 4 \times 1 \end{equation}

In python we can check the answer:


{% highlight python %}
print("1+2+3+4:",1+2+3+4)
{% endhighlight %}

    1+2+3+4: 10



{% highlight python %}
a = np.array([[1,2,3,4]])
a
{% endhighlight %}





    array([[1, 2, 3, 4]])




{% highlight python %}
b = np.ones((4,1))
b
{% endhighlight %}





    array([[ 1.],
           [ 1.],
           [ 1.],
           [ 1.]])




{% highlight python %}
a.dot(b)
{% endhighlight %}





    array([[ 10.]])



### Order matters

For matrices \\(A \cdot B \neq B \cdot A\\) but \\(A \cdot B = ( B^T \cdot A^T ) ^T \\)

Let's prove this by first creating two square matrices:


{% highlight python %}
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a
{% endhighlight %}





    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




{% highlight python %}
b = np.array([[10,11,12],[13,14,15],[16,17,18]])
b
{% endhighlight %}





    array([[10, 11, 12],
           [13, 14, 15],
           [16, 17, 18]])



\\(A \cdot B \\):


{% highlight python %}
a.dot(b)
{% endhighlight %}





    array([[ 84,  90,  96],
           [201, 216, 231],
           [318, 342, 366]])

\\(B \cdot A \\) is different to \\(A \cdot B\\):


{% highlight python %}
b.dot(a)
{% endhighlight %}





    array([[138, 171, 204],
           [174, 216, 258],
           [210, 261, 312]])



\\(( B^T \cdot A^T )^T\\) is the same as \\(A \cdot B\\) :


{% highlight python %}
b.T.dot(a.T).T
{% endhighlight %}





    array([[ 84,  90,  96],
           [201, 216, 231],
           [318, 342, 366]])



## Practical Example with the  McCulloch-Pitts Neuron

Time for a practical example!

In the McCulloch-Pitts artificial neuron, a single neuron transforms the weighted sum of its inputs:

\\( x_j = \sum\limits_{k \in K_j} w_{kj}y_k \\)

Where:

* \\(j\\) is the neuron index
* \\(x_J\\) is the output (which might be subject to an activation function)
* \\(w\\) is a weighting factor
* \\(k\\) is the input index

There are several inputs to each neuron, so \\(x_j\\) is a sum of these weighted inputs.


Let's consider one neuron, \\(j\\), for now, and assume it has three inputs (k = 1 to 3) with values:

\\( x_j = \sum\limits_{k = 1}^{3} w_{kj}y_k \\)

\begin{equation}
y = \begin{bmatrix}
 0.1 & 0.2 & 0.3 &#92;&#92;
\end{bmatrix}
\end{equation}

It would require a weight for each input:

\begin{equation}
w_k = \begin{bmatrix}
 0.4 & 0.5 & 0.6 &#92;&#92;
\end{bmatrix}
\end{equation}

In matrix form: \\( x_j = y \cdot w_j^T\\) (the matrices are one dimensional, so are called vectors and written in lower case)

We would expect a single output figure, the sumation of the products of all the weights and inputs, for \\(x_j\\), so:

\begin{equation}
\begin{bmatrix}
 0.1 & 0.2 & 0.3 &#92;&#92;
\end{bmatrix}
\cdot \begin{bmatrix}
 0.4 & 0.5 & 0.6 &#92;&#92;
\end{bmatrix}^T = \begin{bmatrix}
 0.1 & 0.2 & 0.3 &#92;&#92;
\end{bmatrix}
\cdot \begin{bmatrix}
 0.4 &#92;&#92;
 0.5 &#92;&#92;
 0.6 &#92;&#92;
\end{bmatrix} = 0.1 \times 0.4 + 0.2 \times 0.5 + 0.3 \times 0.6 = 0.32
\end{equation}

In _Python_:


{% highlight python %}
y   = np.array([[0.1,0.2,0.3]])
w_k = np.array([[0.4,0.5,0.6]])
y.dot(w_k.T)
{% endhighlight %}





    array([[ 0.32]])



Now suppose we have not one, but four neurons (numbered 1 - 4) in our layer, each with the same inputs. Each neuron would have its own set of weights:

\\( x_1 = \sum\limits_{k \in K_j} w_{1k}y_k \\)

\\( x_2 = \sum\limits_{k \in K_j} w_{2k}y_k \\)

\\( x_3 = \sum\limits_{k \in K_j} w_{3k}y_k \\)

\\( x_4 = \sum\limits_{k \in K_j} w_{4k}y_k \\)


All the weights together might now look like this:

\begin{equation}
w_k = \begin{bmatrix}
 0.4 & 0.5 & 0.6 &#92;&#92;
 -0.1 & -0.2 & -0.3 &#92;&#92;
 0.7 & 0.8 & 0.9 &#92;&#92;
 -0.4 & -0.5 & -0.6 &#92;&#92;
\end{bmatrix}
\end{equation}


In _Python_:


{% highlight python %}
w = np.array([[0.4,0.5,0.6],[-0.1,-0.2,-0.3],[0.7,0.8,0.9],[-0.4,-0.5,-0.6]])
{% endhighlight %}


We can use exactly the same equation, \\( x_j = y \cdot w_j^T\\), but this time, we should get an output for every neuron:


{% highlight python %}
y.dot(w.T)
{% endhighlight %}





    array([[ 0.32, -0.14,  0.5 , -0.32]])



This is great, but we can go one step further. What if we had several data samples, so we wanted to find the output with different values of the input?  We could look through a _for_ loop to find the values, or put all the inputs into a matrix. Let's say we have just two samples:

\begin{equation}
y = \begin{bmatrix}
 0.1 & 0.2 & 0.3 &#92;&#92;
 0.4 & 0.6 & 0.7 &#92;&#92;
\end{bmatrix}
\end{equation}



{% highlight python %}
y   = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
{% endhighlight %}


The code looks the same, only now we have an output row for every input row:


{% highlight python %}
y.dot(w.T)
{% endhighlight %}





    array([[ 0.32, -0.14,  0.5 , -0.32],
           [ 0.77, -0.32,  1.22, -0.77]])



## In Summary

Matrices can be confusing, hopefully this has demystified them a little. They're used heavily in machine learning, so it's good to build strong foundations first, rather than spend hours of trial and error, trying to rotate them around until the outputs just sort of _look_ right. Enjoy!
