---
layout: post
title:  "Python - Writing code for a quiz"
date:   2017-03-07 18:00:00 -0700
categories: python
---

Here are some course notes for getting started with Python. In this article, I'll
show some code to play with.

--------------


## The Code

Here's some code, written for Python 3 (if you try and run it using Python 2, it
will cause an error because the "input" statement for Python 2 is called 
"raw_input" instead).


There are a couple of new things in the code.  Firstly, we discussed lists previously,
but this uses a list of lists. A list can have a number of objects. The "quiz" list
in this case has a list of objects; each object being a list!

Secondly, the "input" command is new. This command takes an argument of what it should
ask the user, and returns the answer the user gave.


{% highlight python %}
quiz = [
	[ "What is the capital of Norway" , "Oslo" ],
	[ "Entomology is the science that studies", "Insects" ],
	[ "What is 9 x 7", "63" ]
]

for q in quiz:
  question = q[0]
  answer = q[1]
  response = input(question +"? ")
  print("You responded with ",response,"the correct answer is",answer)

{% endhighlight %}


Can you improve this code?  Things to try:

* Add some new questions
* Use ```if``` to see if the user was correct
* Add a variable to keep score and use it to tell the user how well they did
* When checking answers, it should not matter if the user uses lower case or upper case letters. How could you code for this?



