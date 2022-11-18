# AdvancedRegression

This repository contains the course material for 
[Advanced Regression Course](https://www.intuitivebayes.com/advanced-regression).

## What is contained in this repository
### Lesson Code
All slides and code that you see in the video are present in the `lesson_code` directory.
This is a mix of Google slides or Jupyter notebook depending on the lesson

### Exercises
The lesson exercises are also contained in the `lesson_code` directory.   


## Setup

Most of the course material is provided as Jupyter notebooks. We provide two sets of instructions to install the required
packages needed to run these notebooks. The first, that we strongly recommend, will rely on [Anaconda](hhttps://www.anaconda.com), and the second
will only use the standard [pip python package](https://packaging.python.org/tutorials/installing-packages/).

To see a live video example of how to install the virtual environment for this course, watch the [dedicated video in the Resources Library](https://www.intuitivebayes.com/view/courses/advanced-regression/1615884-resource-library/5071356-environment-installation-with-anaconda).

To install PyMC specifically, you'll find updated instructions on the official repository -- [MacOS](https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(MacOS)), [Windows](https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Windows)), [Linux](https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Linux)).

### Instructions for Anaconda (recommended)

[Anaconda](https://www.anaconda.com/) is a convenient package manager that can create isolated python environments,
install python and non-python packages and libraries, and greatly simplify the process of setting up reproducible
working environments. To continue with these instructions, please make sure to
install [Anaconda](https://www.anaconda.com/products/individual#download-section) on your system.

#### 1. Clone locally

The next step is to clone or download the course material on your computer. Just run the following command where you want the material to be downloaded:

```bash
git clone git@github.com:IntuitiveBayes/AdvancedRegression.git
```

#### 2. Create virtual environment
Once the material is on your computer, you'll see that the repository for this course has a file called `environment.yml` that includes a list of all the packages you need
to install. If you run:

```bash
cd AdvancedRegression
conda env create -f environment.yml
```

from the root course directory, it will create the environment for you and install all the packages listed. This
environment can be enabled using:

```bash
conda activate ib_advanced_regression
```

#### 3. Setup and launch Jupyter

Last step, make Jupyter aware of this new virtual environment. With the `ib_advanced_regression` environment activated, run:

```bash
python -m ipykernel install --user --name ib_advanced_regression
```

That will create what's called a *kernel* in Jupyter linguo (this is just a mirror of your virtual environment).
Then, you can start JupyterLab to access the materials:

```bash
jupyter lab
```

**When you open any notebook, make sure it's using the right kernel, which will be named `ib_advanced_regression`**. You can check this at the top-right of the Jupyter page (refer to the [dedicated video in the resources lesson](https://www.intuitivebayes.com/view/courses/advanced-regression/1615884-resource-library/5071356-environment-installation-with-anaconda)).

### Instructions for pip (only if necessary)

Sometimes, using Anaconda is not an option. Don't worry, to run these notebooks you won't need much. You will have to
first [install python and pip](https://www.python.org/downloads/) on your system. Then, you will need to
install [graphviz](https://www.graphviz.org) on your system.

You can work in your system level python, but we strongly recommend that you isolate your environment
by [creating a new virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) using
the python provided `venv` tool.

```bash
python -m venv /path/to/new/virtual/environment
```

You will then have to activate your environment to install contents into it by running:

```bash
source <venv>/bin/activate
```

Refer
to [this link for instructions for Windows users](https://docs.python.org/3/library/venv.html#creating-virtual-environments).

Then, to install the required packages, run:

```bash
python -m pip install -r requirements.txt
```

You're almost there! Once this is done, refer to the third section of the Anaconda instructions (above) to setup and launch Jupyter Lab.
