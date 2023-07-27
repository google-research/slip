This is not an officially supported Google product.

# SLIP - Synthetic Landscape Inference for Proteins
![](https://github.com/google-research/slip/workflows/Build/badge.svg)

SLIP is a sandbox environment for engineering protein sequences with
synthetic fitness functions. See our [preprint](https://www.biorxiv.org/content/10.1101/2022.10.28.514293v1)

## Installation instructions

Tested on python >= 3.7

We recommend installing into a [virtual environment](https://docs.python.org/3/library/venv.html) to isolate dependencies.

```
python3 -m venv env
source env/bin/activate
```

To install:
```
pip3 install -q -r requirements.txt
```

To run the unit tests:
```
bash -c 'for f in *_test.py; do python3 $f || exit 1; done'
```

## Example landscape usage

See this [colab](https://colab.research.google.com/drive/1BkR2KvvjgzUTJg5VO3BsuTPSDjQisnbJ) for an example of using a landscape.

## Constructing a new landscape

All landscapes were constructed using [Mogwai](https://github.com/songlab-cal/mogwai). See that repo's [example](https://github.com/songlab-cal/mogwai/blob/main/examples/gremlin_train.ipynb), which shows how to train a new Potts model and how to (optionally) examine contact accuracy after training. All that is required is an alignment in .a3m format, true contacts are not required (e.g. as in this [colab](https://github.com/songlab-cal/slc22a5/blob/main/slc22a5_train_potts.ipynb)). 
