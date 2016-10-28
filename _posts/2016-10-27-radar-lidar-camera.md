---
layout: post
title:  "Radar, LIDAR and Cameras for Automotive Applications"
date:   2016-10-28 13:00:00 -0700
categories: machine-learning
---

Here's a quick breakdown of the types of sensors used for feature detection in automotive applications.

### LIDAR
_Light detection and ranging_

Uses a pulsed laser to measure distances to objects. Very good for accurate three dimensional shape information. Not so good for detecting white lines on a road.

* Range: 100 meters
* Cost: High
* Amount of data: High
* Good for: 3D object mapping

### Radar
_Radio detection and ranging_

Sends out radio waves and detects them bouncing back to determine range, angle and velocity of objects.

* Range: 100 meters
* Cost: Medium
* Amount of data: Low
* Good for: Object detection in poor weather and light conditions

### Camera
Cameras can be used to detect visible light, as well as infra-red for night vision.

* Range: 100 meters
* Cost: Low
* Amount of data: High
* Good for: Detection of road marking, signs and other traffic.
* Not good for: Low visibility conditions, such as fog


### Ultrasonic
Ultrasonic senders and receivers are used to measure the echos from the transmitted high-frequency sound. This technology is often used for parking sensors

* Range: 1 meter
* Cost: Low
* Amount of data: Low
* Good for: Detection of close objects


### Electro-magnetic
An electromagnetic field is generated around the front and rear bumpers to sense objects in the vicinity. Unlike ultrasonic sensors, these do not have to be mounted on the surface of the vehicle; they can be mounted behind existing panels.

* Range: 1 meter
* Cost: Low
* Amount of data: Low
* Good for: Detection of close objects
