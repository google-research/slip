This is not an officially supported Google product.

# SLIP - Synthetic Landscape Inference for Proteins
![](https://github.com/google-research/slip/workflows/Build/badge.svg)

SLIP is a sandbox environment for engineering protein sequences with
synthetic fitness functions.

## Status

In progress

## Instructions

Tested on python >= 3.7

We recommend installing into a virtual environment to isolate dependencies (see https://docs.python.org/3/library/venv.html).

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

