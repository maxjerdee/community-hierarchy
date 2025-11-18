# community-hierarchy

[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/maxjerdee/community-hierarchy/workflows/CI/badge.svg
[actions-link]:             https://github.com/maxjerdee/community-hierarchy/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/community-hierarchy
[conda-link]:               https://github.com/conda-forge/community-hierarchy-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/maxjerdee/community-hierarchy/discussions
[pypi-link]:                https://pypi.org/project/community-hierarchy/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/community-hierarchy
[pypi-version]:             https://img.shields.io/pypi/v/community-hierarchy
[rtd-badge]:                https://readthedocs.org/projects/community-hierarchy/badge/?version=latest
[rtd-link]:                 https://community-hierarchy.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->


# community-hierarchy

### Model the interaction between community and hierarchy in directed graphs

##### Maximilian Jerdee, Elizabeth Bruch, Mark Newman

`community-hierarchy` is a python package that implements a Bayesian model for the interaction between community and hierarchy structure in directed networks. We particularly focus on applications to social settings, for example networks of friendship nominations within schools. This model allows us to measure not only the social groups and social hierarchy within the school, but also the status associated with particular groups: the cool group. 

We model the group structure as a Stochastic Block Model (SBM) and social hierarchy as a Bradley-Terry (BT) model. 

Examples of data may be found in the `\data` directory in `.gml` file formats. If a community structure is given (as a "community" attribute), it can either be taken as the fixed community structure or as the initial partition for Monte Carlo sampling over the space of possible community structure partitions. 


## Installation

`community-hierarchy` can be installed through pip:

```bash
pip install community-hierarchy
```

or built locally by cloning [this repository](https://github.com/maxjerdee/community-hierarchy) and running

```bash
pip install .
```

in the base directory.

## Typical usage

Once installed, the package can be imported as

```python
import community_hierarchy as ch
```

Note that this is not `community-hierarchy`.

To load network data, 