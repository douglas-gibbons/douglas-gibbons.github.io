---
layout: post
title:  "Python - Writing code for a quiz - 2"
date:   2017-03-13 19:00:00 -0700
categories: ["Python Course"]
---

Here are some course notes for getting started with Python. In this article, I'll
show some more code to play with.

--------------


## The Code

Here's some code, written for Python 3 (if you try and run it using Python 2, it
will cause an error because the "input" statement for Python 2 is called 
"raw_input" instead).

This is suspiciously similar to the code in <a href="{% post_url 2017-03-07-python-making-a-quiz %}">the previous post</a>, however I've modified it, so instead of reading from a list, it reads from a "csv" file called quiz.csv.

"csv" stands for "comma-separated-values", it loosely defines a file format where the fields are separated by commas, and can be enclosed in quotes. Sometimes the fields are not separated by quotes, sometimes they are. It can all get rather confusing, so rather than try and write the code to read CSV files ourselves, we just rely on a library to do it.

New things:

* [csv library](https://docs.python.org/3/library/csv.html)
* [Reading files](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)

Below is the code, that you should copy and save into a file called "quiz.py", and also a CSV file that you should save as "quiz.csv" in the same directory.  Once you have both, you should be able to run the code with ```python3 quiz.py```.

### The Code

{% highlight python %}
# Imports are always first. This one loads in the CSV library
import csv

# Now we open the file
csvfile = open('quiz.csv','r')

# Now we use the CSV library to create an iterator that we can use a "for
# loop" on. Each iteration returns a new line of the CSV file as a list of
# the fields from that line in the file.
lines = csv.reader(csvfile, delimiter=',', quotechar='"')

# Now we iterate over our iterator
for q in lines:
  question = q[0]
  answer = q[1]
  response = input(question +"? ")
  print("You responded with",response,"the correct answer is",answer)

{% endhighlight %}


### quiz.csv

{% highlight csv %}
"What is the capital of Norway","Oslo"
"Entomology is the science that studies","Insects"
"What is 9 x 7","63"
{% endhighlight %}



## Next Steps

What can you do to improve the code?  If you tried the previous example, may I suggest copying and pasting some of your modifications from that?
