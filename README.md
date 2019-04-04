# Fast Sampling from Categorical Distributions on the GPU using PyTorch

Currently, the `pytorch.distributions.Categorical` is a bit slow if you
need to draw a large number of samples from a static categorical distribution.
Also, you are limited to having no more than 2^24 different outcomes.

The 
["alias method"](http://cgi.cs.mcgill.ca/~enewel3/posts/alias-method/index.html)
let's you sample very quickly from distributions with large support, and this
implementation in PyTorch let's you have more than 2^24 outcomes.

I needed this for rapid generation of word embeddings in
[hilbert](https://github.com/enewe101/hilbert).

# Install

``pip install pytorch-categorical``

# Use

    import pytorch_categorical
	import torch

	num_outcomes = int(1e6)
    probs = torch.random(num_outcomes)
	probs /= probs.sum()

	sampler = pytorch_categorical.Categorical(probs)

	num_samples = int(1e6)
	samples = sampler.sample((num_samples,))

The constructor also takes a `dtype` and a `device` if you want to specify 
them.  By default

# Posterity
At the time I made this, there was an open issue to incorporate a more rapid
sampler based on the alias method, but nothing was released yet.  Hopefully
that will get into a release soon!  For now, use this!

# Tested.  It's Correct and Fast.
I've backed this by a few simple tests, including a benchmark against torch.
This implementation takes about 175X longer to construct a sampler with one
million outcomes, but after this up-front cost, it yields (draws of ten
thousand) samples about 3000X faster (with greater advantage the more samples
that are eventually drawn).  So the main usecase is when you have to draw many
samples from a stable distribution.

Run the correctness and benchmark tests: ``python test.py``.

Enjoy!

