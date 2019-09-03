# Advanced Message Passing Algorithms

## Project tree
.
 * [bin](./bin)
   * [alpha_compare_mmse.py](./bin/alpha_compare_mmse.py)
   * [alpha_compare.py](./bin/alpha_compare.py)
   * [ep.py](./bin/ep.py)
   * [marginal_loopy.py](./bin/marginal_loopy.py)
   * [plot_save.py](./bin/plot_save.py)
   * [varying_loopy.py](./bin/varying_loopy.py)
 * [manuscript](./manuscript)
 * [README.md](./README.md)
 * [requirements.txt](./requirements.txt)
 * [src](./src)
   * [alphaBP.py](./src/alphaBP.py)
   * [factorgraph.py](./src/factorgraph.py)
   * [loopy_modules.py](./src/loopy_modules.py)
   * [maxsum.py](./src/maxsum.py)
   * [modules.py](./src/modules.py)
   * [utils.py](./src/utils.py)
   * [variationalBP.py](./src/variationalBP.py)

## Installation

First clone the repository:
```bash
$ git clone https://github.com/FirstHandScientist/expectation_propagation.git
```
## Environment
Create a virtual environment with a python2 interpreter at 'path/to/your/evn/'
```bash
$ virtualenv -p python2.7 pyenv27
```
Then activate your environment:

``` bash
$ source path/to/your/evn/pyenv27/bin/activate
```
and install the requirement file:

``` bash
$ pip install -r requirements.txt
```

# To-do list:
- [ ] More explanation in readme
- [ ] boosting subsection in manuscript
- [ ] One round of writing check
