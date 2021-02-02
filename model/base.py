from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaseException(Exception):
    def __init__(
            self,
            parameter,
            types: List):
        message = '{} type must be one of {}'.format(parameter, types)
        super().__init__(message)


class Base(pl.LightningModule):
    def __init__(self):
        super(Base, self).__init__()

    def migrate(
            self,
            state_dict: Dict
    ):
        state_dict_keys = state_dict.keys()
        with torch.no_grad():
            for name, p in self.state_dict().items():
                if name in state_dict_keys:
                    p_ = state_dict[name]
                    if p.data.shape == p_.shape:
                        p.copy_(p_)

    def remove_prefix_state_dict(
            self,
            state_dict: Dict,
            prefix: Union[str, int]
    ):
        result_state_dict = {}
        if isinstance(prefix, int):
            # TODO
            return state_dict
        elif isinstance(prefix, str):
            len_prefix_remove = len(prefix) + 1
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    result_state_dict[key[len_prefix_remove:]
                                      ] = state_dict[key]
                else:
                    result_state_dict[key] = state_dict[key]
            return result_state_dict
        else:
            raise BaseException('prefix', [str, int])

    def filter_prefix_state_dict(
        self,
        state_dict: Dict,
        prefix: str
    ):
        if not isinstance(prefix, str):
            raise BaseException('prefix', [str])


class BaseSequential(nn.Sequential, Base):
    pass
