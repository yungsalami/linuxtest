# SMDA Project


## Install conda or miniconda

Download from https://docs.conda.io/en/latest/miniconda.html

Download the python 3.8 version for your OS, on Linux and Mac run:

```
$ bash Miniconda-<vesion>.sh -p $HOME/.local/miniconda -b
```

Add the following lines to your `$HOME/.bashrc`
```
source $HOME/.local/miniconda/etc/profile.d/conda.sh
```
and start a new terminal.


## Create a new environment for SMD

In this repository, run:

```
$ conda env create -f environment.yaml
```

To use this environment, use
```
$ conda activate smd
```


Install the python module in this repository in developement mode:
```
$ pip install -e .
```

Test if you can import it everywhere:
```
$ cd $HOME
$ python -c 'import project_a5'
```

## Run the unit tests

We provide some unit tests, for our implementations and
some to check if you implemented the exercises correctly,
feel free to add more tests yourself.

Run them using
```
$ python -m pytest
```


