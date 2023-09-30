# PINNa colada 🍹 

*Where Deep Learning Shakes up Physics 🧪🤖*
![Pinnacolada](pinacolada.jpg)


## Dependencies (for development, i.e. after cloning the repo)

1. Optionally: create a virtual environment, e.g. with conda:

```shell
conda create -n pinnacolada python=3.10
conda activate pinnacolada
```

If you don't create a virtual environment with conda, poetry should create one.


2. Use [poetry](https://python-poetry.org/) to install all dependencies:

```shell
poetry install
```

3. Install pytorch (see below)


## Dependencies (for package)
All dependencies are installed with the package, except of `pytorch`. 
You have to install it manually following instructions from 
[here](https://pytorch.org/get-started/locally/), e.g. like this:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

This is caused by limitation of poetry which does not allow to specify 
`--index-url` flag. Without it, CPU only version would be installed, which 
usually is not what you want.


