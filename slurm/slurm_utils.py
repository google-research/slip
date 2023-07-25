from datetime import datetime
import itertools
import json
from typing import Dict, Sequence


def product_dict(**kwargs):
    """Returns the dictionaries representing the cartesian product over all keyword arguments.

    Example usage:
    >>> list(product_dict(num=[1, 2], alpha=['a', 'b']))
    [{'num': 1, 'alpha': 'a'},
     {'num': 2, 'alpha': 'a'},
     {'num': 1, 'alpha': 'b'},
     {'num': 2, 'alpha': 'b'},]
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def update_dict(input_dict: Dict, update_dict: Dict) -> Dict:
    """Returns a dict including (key: value) pairs in both `input_dict` and `update_dict`."""
    return dict(input_dict, **update_dict)


def get_params_json(global_defaults: Dict,
                    global_options: Dict,
                    local_defaults_list: Sequence[Dict],
                    local_options_list: Sequence[Dict]) -> Sequence[str]:
    assert len(local_defaults_list) == len(local_options_list)
    params_json_list = []
    for local_defaults, local_options in zip(local_defaults_list, local_options_list):
        all_options = update_dict(global_options, local_options)
        for option in product_dict(**all_options):
            d = update_dict(option, local_defaults)
            d = update_dict(d, global_defaults)
            params_json_list.append(json.dumps(d))
    return params_json_list
