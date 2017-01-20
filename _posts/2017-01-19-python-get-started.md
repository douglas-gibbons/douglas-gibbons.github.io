---
layout: post
title:  "Getting Started With Python"
date:   2017-01-19 14:00:00 -0700
categories: python
---

Here are some course notes for getting started with Python. In this article, we'll open a terminal window and run some Python commands, just to prove that we can!

--------------

# Using the terminal prompt

Called the "terminal window" on Mac and the "command prompt" on Linux, this is your _command line interface_ to another world!

__To open on Windows:__ Press the start button and type "command". You should see the "command prompt" application for you to open ([more info](https://www.lifewire.com/how-to-open-command-prompt-2618089)).

__To open on a Mac:__ [Here's a quick video](https://www.youtube.com/watch?v=zw7Nd67_aFw)


## Commands

#### Listing files and directories

On windows the command is: ```dir```<br />
On a mac the command is: ```ls```

#### Creating a directory

On windows and mac the command is: ```mkdir```

For example ```mkdir "python 101"``` creates a directory called ```python 101``` under the current directory.  Note the quotation marks around the name of the directory. You only need those if the directory has funny characters, such as spaces in the name.


#### Changing directory

On windows and mac: ```cd```

For example ```cd "python 101``` changes directory to the "```python 101```" directory.

## Running Python from the terminal prompt

We're going to assume you have [python installed](https://www.python.org/downloads/) for this!

There are a few ways to run python. For now, we'll look at the interactive mode, and running python files from the terminal prompt.

#### Python Interactive Mode

From a terminal window, just type ```python```. You'll see some text come up, that no one ever reads, but it should look a little like this:


    Python 3.5.2 (default, Nov 17 2016, 17:05:23)
    [GCC 5.4.0 20160609] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>>

Now you're ready to type python commands.  For example, entering ```print("Hello world")``` does what you might imagine and ```exit()``` exits back to your Windows or Mac command prompt.

#### Running Python Scripts

A python script is a file with python commands in it. You can run them straight from your Windows or Mac command prompt.

For example, to run a program called ```hello world.py``` just type ```python "hello world.py"```.  Note the quotation marks around the name of the file, because we wanted to be difficult and put a space in the filename.

# Frequently Asked Questions

#### What's the difference between an interactive python session, and a command prompt?

The command prompt is used for typing commands for the operating system.  Your operating system is there to manage files, and run programs for you, so you'll typically use it for creating directories,  or running over programs.

The "interactive python session" is a python interpretor, where you can type python commands.

If that wasn't confusing enough, we can open a terminal window, and run the python command in the terminal window, to issue python commands!

Below shows an example of this, where I create a directory called ```test```, show the directory contents (to show my new directory), then change into that directory.  I then run an interactive python session to print ```Hello World```.  The commands I typed are underlined in red.

![screenshot](/assets/commands-screenshot.png)



Thanks for reading. In my next article in the series, we'll use what we've learned to build a rocket capable of reaching Mars, just using freely available Python libraries.



