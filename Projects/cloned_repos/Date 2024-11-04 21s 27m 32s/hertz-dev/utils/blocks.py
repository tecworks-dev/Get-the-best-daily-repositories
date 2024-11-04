from dataclasses import dataclass
from typing import TypeVar, Generic, Type, Optional
from functools import wraps
import time
import random

import torch as T
import torch.nn as nn

# @TODO: remove si_module from codebase
# we use this in our research codebase to make modules from callable configs
si_module_TpV = TypeVar('si_module_TpV')
def si_module(cls: Type[si_module_TpV]) -> Type[si_module_TpV]:
    if not hasattr(cls, 'Config') or not isinstance(cls.Config, type):
        class Config:
            pass
        cls.Config = Config
    
    cls.Config = dataclass(cls.Config)
    
    class ConfigWrapper(cls.Config, Generic[si_module_TpV]):
        def __call__(self, *args, **kwargs) -> si_module_TpV:
            if len(kwargs) > 0:
                config_dict = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
                config_dict.update(kwargs)
                new_config = type(self)(**config_dict)
                return cls(new_config)
            else:
                return cls(self, *args)
    
    ConfigWrapper.__module__ = cls.__module__
    ConfigWrapper.__name__ = f"{cls.__name__}Config"
    ConfigWrapper.__qualname__ = f"{cls.__qualname__}.Config"
    
    cls.Config = ConfigWrapper
    
    original_init = cls.__init__
    def new_init(self, *args, **kwargs):
        self.c = next((arg for arg in args if isinstance(arg, cls.Config)), None) or next((arg for arg in kwargs.values() if isinstance(arg, cls.Config)), None)
        original_init(self, *args, **kwargs)
        self.register_buffer('_device_tracker', T.Tensor(), persistent=False)
    
    cls.__init__ = new_init
    
    @property
    def device(self):
        return self._device_tracker.device
    
    @property
    def dtype(self):
        return self._device_tracker.dtype
    
    cls.device = device
    cls.dtype = dtype
    
    return cls


def get_activation(nonlinear_activation, nonlinear_activation_params={}):
    if hasattr(nn, nonlinear_activation):
        return getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
    else:
        raise NotImplementedError(f"Activation {nonlinear_activation} not found in torch.nn")


def exists(v):
    return v is not None

def isnt(v):
    return not exists(v)

def truthyexists(v):
    return exists(v) and v is not False

def truthyattr(obj, attr):
    return hasattr(obj, attr) and truthyexists(getattr(obj, attr))

defaultT = TypeVar('defaultT')

def default(*args: Optional[defaultT]) -> Optional[defaultT]:
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner
