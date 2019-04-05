import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-categorical",
    version="0.0.3",
    author="Edward Newell",
    author_email="edward.newell@gmail.com",
    description=(
	"Draw a large number of samples from a categorical distribution with "
	"large support on the GPU using Pytorch."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enewe101/pytorch-categorical",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

print(
    "You'll need PyTorch to use this.  Get it from https://pytorch.org/ ."
)
