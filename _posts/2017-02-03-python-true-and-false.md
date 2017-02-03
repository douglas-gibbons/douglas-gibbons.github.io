---
layout: post
title:  "True and False with Python"
date:   2017-02-03 10:00:00 -0700
categories: python
---

Here are some course notes for getting started with Python. In this article, we'll look at "True", "False", variables, strings, and a quick glimpse at making decisions.

--------------

## Variables and Strings

"variables" are a way of storing something we might need later. Variable names can start with an underscore, or a letter and can contain numbers, letters and underscores.

We can assign a variable a value, using the "=" sign.

Here are some examples of assigning values to variables:


{% highlight python %}
badger_age = 2
badger_name = "Gertrude"
possom1 = "small"
possom2 = "large"

print("Age of badger is")
print(badger_age)
{% endhighlight %}


    Age of badger is
    2


We can also assign the value of one variable to another. For example:


{% highlight python %}
possom1_friend = badger_name
print(possom1_friend)
{% endhighlight %}


    Gertrude


See we've used quotation marks around some of the text?  This is how we create a "string". Strings can also contain spaces in them. The quotation marks are necessary, otherwise, Python may think we mean a variable instead of a string.  Here are some examples of strings and variables that might be _really_ confusing, if you didn't know about strings and quotation marks!


{% highlight python %}
sentence = "The quick brown fox jumps over the lazy dog"
word = "sentence"
{% endhighlight %}



{% highlight python %}
print(sentence)
{% endhighlight %}


    The quick brown fox jumps over the lazy dog



{% highlight python %}
print("sentence")
{% endhighlight %}


    sentence



{% highlight python %}
print(word)
{% endhighlight %}


    sentence



{% highlight python %}
print("word")
{% endhighlight %}


    word


## True and False

You can assign variables to be either ```True``` or ```False``` too. For example:


{% highlight python %}
sun_today = True
rain_today = False

print(sun_today)
{% endhighlight %}


    True


 The first letter of True or False _must_ be capitalized. If we try ```true``` instead of ```True``` we get an error:


{% highlight python %}
sun_today = true
{% endhighlight %}



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-9-7400b3cce556> in <module>()
    ----> 1 sun_today = true
    

    NameError: name 'true' is not defined


See that we've always used a single ```=``` sign to assign values to variables?  Two equals signs ```===``` have a different and special meaning.  It allows us to check and see if two values are the same.  If they are, it will return ```True``` and if not, ```False```. For example:


{% highlight python %}
badger_name = "Gertrude"

print( badger_name == "Gertrude")
{% endhighlight %}

    True



{% highlight python %}
print( badger_name == "Tony")
{% endhighlight %}


    False


What if we want to check that two things are _not_ equal? Then we use ```!=``` instead. For example:


{% highlight python %}
print( badger_name != "Gertrude")
{% endhighlight %}


    False



{% highlight python %}
print( badger_name != "Tony")
{% endhighlight %}


    True


...so, it is ```True``` that our badger is not called Tony.

Here is a strannge example.  Does it make sense?


{% highlight python %}
true = True
print(true == True)
{% endhighlight %}


    True


We have created a variable called "true" and assigned it a value of ```True```!  Computer programs should be written so others can understand what they do.  For example, this would be rather evil, because it would really confuse anyone trying to read the code:


{% highlight python %}
true = False

the_sky_is_blue = true
print(the_sky_is_blue == True)
{% endhighlight %}


    False


I hope it made sense what happened there! This is bad code, but if it still made sense to you, you're doing brilliantly!

