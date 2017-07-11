---
layout: post
title:  "Starting a Company: What Software Do I Need?"
date:   2017-07-11 11:00:00 -0700
categories: ["Startups"]
---

I recently helped start a company called [Trafero](https://trafero.io) and had to put together all the software I thought we needed. As an advocate of Open Source, I tried to use as much open source software as was sanely possible.

Here's what I learned.

----------------------------------------------------

## It Starts With a Name

Naming a company should be easy enough, right?  It turns out that finding the right domain name for your company is actually fiendishly difficult, because they _all_ seem to be taken already.

If you don't believe me, [give it a go](https://www.godaddy.com/domains/domain-name-search).

There are a few other places you might want to register your brand. For me it was:

* [Twitter](https://twitter.com/)
* [Github](https://github.com/)
* [hub.docker.com](https://hub.docker.com/)

Github and docker may not be your thing, but that's ok.

Once you find that domain name, register it with a general purpose registration service, such as [godaddy.com](https://www.godaddy.com/domains/domain-name-search) or (my favourite) [123-Reg](https://www.123-reg.co.uk/).


## Email and Calendars

I have used [mailu](https://github.com/Mailu/Mailu) before for email.  It takes quite a bit of setting up, but there is a bigger problem: Calendars.

For "home use" I use Google for calendars, and something different for email.  This is bad. Every time someone emails me a meeting invite, there's a merry dance I have to go through to get that invite in the right calendar and send them back a meeting acceptance.  This is not how to run a business.

There are a few Open Source Groupware products out there, and I hope that at least one of them is good. Sadly, I don't know, as I did a rare thing in the world of computing and thought of my users.  They're used to running Gmail, so I took the easy route and went for [G Suite](https://gsuite.google.com/).

## The Website

Of course you want a website! Who wouldn't? What is the first thing that strikes you, when you see my website?  Probably that you shouldn't ask me for advice on designing one.

Webites need content. Your users need to know who you are and what you do, so get your story straight first.  Have you done a [Three-Hour Brand Sprint](http://www.brandknewmag.com/the-three-hour-brand-sprint/)? Do you know what your [brand colors](https://www.bigbrandsystem.com/how-to-choose-brand-colors/) are?  Do you even have a [brand font](https://fontlibrary.org/)?  It's good to get these questions right first, otherwise, when it comes to building a website, it somehow just won't _feel_ like you want it to. Without the right questions being asked, you may well waste valuable time trying to get it to feel right by trial and error.

Website content is important to. It's good to have real content ready to go before the website design starts.  An empty website looks, well, empty.

#### Website Software

[Wordpress](https://wordpress.org/) is an old favourite. There's a [hosted solution](https://wordpress.com/) for it, if you don't want to run it yourself.  It is powerful and used by many large companies. Wordpress is written in PHP behind the scenes. Luckily, no coding is required to use the product, because coding in PHP is like trying to eat a bowl full of bees; it only makes sense if you have no pain receptors, and even then, it's a questionable act.

A lot of the startups these days use [Wix](https://www.wix.com/) instead, however it's not Open Source. I favour [jekyll](https://jekyllrb.com/). You might even see the similarity between [this jekyll theme](https://jekyll-demos.github.io/Arcana-Jekyll-Theme/) and [my company website](https://trafero.io) if you look hard enough.

Jekyll is open source and free, but will probably require you to do some HTML editing, as well as learning [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).  The silver lining to this clolud is that you might be able to [host your jekyll site on Github](https://help.github.com/articles/using-jekyll-as-a-static-site-generator-with-github-pages/) for free.

## Contact Us!

[Mautic](https://www.mautic.org/) belongs to a group of software that you may not have considered, but I love it. It will allow you to create contact forms that you can incorporate right into your website [such as this](https://trafero.io/contact.html), or any other forms, such as surveys or sign-ups. Once you have that data, it can automatically send out email responses. You can also generate email campaigns right from Mautic, using all those email addresses you harvested. There's also a [hosted solution](https://mautic.com/) if you don't want to install it yourself.

Mautic's flexibility is a strength, but its complexity is a weakness. Prepare to spend a few hours getting to grips with it.  It will be a love/hate relationship at first.

## CRM

I still think I need a CRM for all my customers, even though I don't. If you're like me, you might want to try [SuiteCRM](https://suitecrm.com/), which is an offshoot of [SugarCRM](https://www.sugarcrm.com/). SuiteCRM has a slightly more up-to-date user interface (only slightly).  There are hosted options available if you Google hard enough, otherwise there's always the ever popular (and Closed Source) [Salesforce](https://www.salesforce.com/).

## Wiki

I didn't think I needed a wiki, but now I have it, I use it a lot.  There's so much information and research when creating a company. It's all inter-connected, but sometimes it's hard to know how to make those connections. Wikis are amazing, in that you can throw information in there in a random way, but, by linking documents together, you can still get all that value out again.


[Mediawiki](https://www.mediawiki.org/wiki/MediaWiki) is an old favourite. Wikipedia uses it. The one disadvantage is that you have to learn how to [format documents](https://www.mediawiki.org/wiki/Help:Formatting) using special notation, but it doesn't take long to get the hang of it.

There are hosted options for MediaWiki too if you google for them.

## Wrapping it Up

All of the open source software at Trafero is hosted in AWS and installed in Docker containers, using docker compose.

Please [let me know](https://trafero.io/contact.html) if this page could be improved, if you want copies of our docker compose files for building your own, or if you want Trafero to build any of this for you.


