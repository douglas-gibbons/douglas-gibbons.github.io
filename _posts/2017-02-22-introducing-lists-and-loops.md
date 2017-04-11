---
layout: post
title:  "Python - Introducing lists and for loops"
date:   2017-02-22 19:00:00 -0700
categories: ["Python Course"]
---

Here are some course notes for getting started with Python. In this article,
we'll look at "lists" and "for loops".

--------------

## Lists

We've seen variables before which can be used to store information, such as strings or numbers, but what do we do if we want to store lots of different items, for example a list of animals?

We could use one variable for each, but there's an easier way with __lists__!


To go back a little, here's how we set a simple variable (not a list):


{% highlight python %}
a = "dog"
{% endhighlight %}


Now, here's how we create a __list__ of animals. See how it's different to setting just a single variable?  We put square brackets around the elements of the list and separate them with commas:


{% highlight python %}
b = [ "dog", "moose", "mouse", "hedgehog", "elephant"]
{% endhighlight %}

Lists are objects with some some useful methods.  For example, we can append to the end of a list with the ```append``` method:


{% highlight python %}
b.append("donkey")
print(b)
{% endhighlight %}

    ['dog', 'moose', 'mouse', 'hedgehog', 'elephant', 'donkey']


The [Python documentation](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) has lots more information on lists, and on the methods that you can use. The documentation may seem a little confusing at first, but it's worth looking at the examples, and then scrolling up to read the different methods of the ```list``` object. 

## For Loops

For loops can be used to loop over lists (among other things). We'll briefly use them here to loop through each element of a list and print out that element:


{% highlight python %}
for c in b:
    print(c)
{% endhighlight %}

    dog
    moose
    mouse
    hedgehog
    elephant
    donkey


What did we do there?  Seems odd, doesn't it?  To translate that to english we did this:

For each element in the list called ```b```, assign the value of that element to the variable ```c```, then print out the value of the variable ```c```.

Do you see that the print statement is indented? We can add more code to that indented block of code, to run for each iteration of the ```for``` loop. Here's a more complicated example. Can you see what it's doing, and how it's doing it?


{% highlight python %}
i = 0
for c in b:
    i = i + 1
    print("Value of i:",i,"Value of c:",c)
{% endhighlight %}

    Value of i: 1 Value of c: dog
    Value of i: 2 Value of c: moose
    Value of i: 3 Value of c: mouse
    Value of i: 4 Value of c: hedgehog
    Value of i: 5 Value of c: elephant
    Value of i: 6 Value of c: donkey


It's ok if that seems a bit confusing at the moment. There's a lot going on there that we'll learn more about later!
