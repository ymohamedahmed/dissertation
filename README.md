# Non-contact heart-rate estimation from video

It turns out that by pointing a simple smartphone camera at a person's face, you can get a good idea of their heart-rate, so good that it can often work about as well as a smartwatch.

I've spent [~11000 words talking about it](report/diss.pdf) if you want to read about how I got skin detection running in real-time or the nitty-gritty of when it works and doesn't. It's [implemented in Python](https://github.com/ymohamedahmed/dissertation/tree/master/code/rPPG/python/core), with heavy reliance on OpenCV, NumPy and SciPy amongst a few other libraries.

<p align="center">
<img src="output.gif" width="600" height="382"/>
</p>
