# Fast Sampling from Categorical Distributions on the GPU using PyTorch

Currently, the `pytorch.distributions.Categorical` is a bit slow if you
need to draw a large number of samples from a static categorical distribution.
Also, you are limited to having no more than 2^24 different outcomes.

If you need categorical distributions with really large support, and/or
you need to quickly draw millions of samples, then this is for you.

At the time I made this, there was an open issue to incorporate a more rapid
sampler based on the alias method (which I use here).  Hopefully that will
get into a release soon!  For now, use this!

I've backed this by a few simple tests, including a benchmark against torch.
This implementation takes about 100X longer to construct a sampler, but 
after this up-front cost, it yields samples about 3500X faster.  So the main
usecase is when you have to draw many samples from a stable distribution.

Enjoy!

