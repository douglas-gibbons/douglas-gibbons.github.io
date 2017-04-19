---
layout: post
title:  "Python functions"
date:   2017-04-18 18:00:00 -0700
categories: ["Python Course"]
---

Here are some course notes for getting started with Python. In this article,
we'll look at python functions.

--------------

## Functions

Want to use the same piece of code several times, without having to rewrite it each time? This is where "functions" come in!

Here's a simple example:

{% highlight python %}
# The function
def sayHello(name):
    print("Hello",name)

# Using the function
sayHello("Doug")
{% endhighlight %}

First, we define a function called ```sayHello```.  The function takes in a variable called ```name``` and uses it to print out a greeting.

We then use the function by calling it with the values we want to give it within parenthesis.

We could feed in several variables to a function too, like this:

{% highlight python %}
# The function
def saySomething(greeting, name):
    print(greeting,name)

# Using the function
saySomething("Hello","Doug")
{% endhighlight %}

Functions can also return values too. For example:

{% highlight python %}
# The function
def addNumbers(a, b):
    return a + b

# Using the function to set a variable called
# "answer" to the return value of the function
answer = addNumbers(1,2)
print("Our answer is",answer)
{% endhighlight %}

We're going to use functions later...

## Mortgage Interest - Is It Interesting?

If you've bought a house, or are thinking about it, you've probably spent more time than you wanted, thinking about mortgage payments.

Calculating mortgage payments isn't something I personally like doing, so it would be nice if it was easy. [wikiHow](http://www.wikihow.com/Calculate-Mortgage-Payments) has a pretty good description of the maths, but I wouldn't want to go through the calculation every time.  Perhaps some Python could help.

Below is that calculation in Python, calculating a 25 year mortgage on $200,000 at 5%.

First we import the ```math``` library, which we need for ```math.pow``` later, then we set some values.  We then convert these to "floating point" values. These are numbers that may have a decimal point in them, such as 12.99. There could be many, few, or no digits after the decimal point, hence the term "floating".

Next, we do the calculation. No need to worry about this too much, but if you want to, feel free to look at the equation on the [wikiHow](http://www.wikihow.com/Calculate-Mortgage-Payments) page, and compare it to the Python code. Don't be put off if you need to write it down and take some time over it.  It took me a while too!

Last, we just print out the answer from the calculation, which is the amount of monthly payments required to pay off the mortgage.

{% highlight python %}
# import math module
import math

# Set some values
years = 25
interest = 5
principle = 200000

# Convert the values from integer numbers to floating point
years = float(years)
interest = float(interest)
principle = float(principle)

# Calculate the result
monthly_interest=interest/100/12
payments = years * float(12)

monthly_payment = principle * (
    monthly_interest * math.pow((1 + monthly_interest ), payments) /
    ( math.pow((1+monthly_interest),payments) -1 )
)

# Print the answer
print("Monthly payment:",monthly_payment)
{% endhighlight %}


## Now The Tricky Part

Can you take the mortgage calculation code and turn it into a function, and use it in some code?  It would need to accept values for ```years```, ```interest```, and ```principle```. What value would it return?

Don't forget to ```import math``` too. Imports should always go at the top of your code.
