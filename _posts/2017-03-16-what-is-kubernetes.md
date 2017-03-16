---
layout: post
title:  "What is Kubernetes?"
date:   2017-03-16 15:00:00 -0700
categories: main
---

_From the [kubernetes website](https://kubernetes.io/); "Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications."_

This article explores what that means, in simple terms, and why Kubernetes is so important.

--------------

Suppose we have one computer, and on that computer we wanted to put some applications, such as components for a web site:

![png](/assets/kubernetes_simple_1.png)

Our web site becomes popular, so we buy another server to cope with the increased load. We then copy our applications onto the new server, so now we have two identical servers, running all the applications that make up our website:

![png](/assets/kubernetes_simple_2.png)

Our websites gets even more popular, so we buy another server.  App 1 needs to talk to App 2, but App 3 can run on its own. There are a few ways to split this, but we decide on the following:

![png](/assets/kubernetes_simple_3.png)

Suppose we expand; now we have more servers of different sizes, and lots of different types of applications with different needs:

![png](/assets/kubernetes_simple_4.png)


There are several problems here:

* How do we decide where to put the applications, so each has the resources it needs?
* How do we connect the applications together, so the right applications can talk to each other, but block the ones that should not have access to each other?
* If a server fails, how to do we automatically move the applications to a new server so that the failure does not affect the live running of our website?
* When a service receives more load, how do we add more instances of the application, so that the service can cope with the additional load?
* How do we update an application while ensuring the website is continuously available to our users?

Kubernetes provides answers for all these questions. It is a layer between the servers and the applications, able to make decisions on how best to distribute and update those applications. It can scale to 1000 servers, and run 10s of thousands of applications at a time.

This is especially important for large users of computing power, such as Google, Ebay, and the Wikimedia foundation, who all use Kubernetes to help them keep their services reliable and secure.
