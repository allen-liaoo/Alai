import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict

class Layer(ABC):
    def __init__(self, *args, **kwargs):
        self.cache = _LayerCache() # Used to cache forward() params or results for backward()``
        self.params = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    # For automatically updating weights, see @auto_update. Should return tuple of the following:
    # 1. dl_dinput
    # 2. then dl_dp for each parameter
    @abstractmethod
    def backward(self, *args, update:bool, lr:float= 1, **kwargs):
        pass

    # Registers parameters to the layer (Should only be done once in __init__)
    # Once registered, a parameter named "param" can be accessed via self.param
    # Note that registration order matters when:
    # 1. Automatically updating weights in backward() via @auto_update (returned order of dl_dp in backward() must match registration order)
    # 2. Calling update_weights with positional arguments (must match registration order)
    def register_params(self, **kwargs):
        n = 0
        for param, val in kwargs.items():
            if type(val) is not np.ndarray:
                raise ValueError(f"{self.__class__.__name__}.register_params: Parameter {param} must be np.ndarray")

            setattr(self, param, val)
            n += val.size # np.ndarray.size returns number of elements

        self.params = OrderedDict(kwargs) # OrderedDict to keep insertion order of params
        self.n_params = n

    # Decorator for marking a backward() as updatable, used by built in models like Sequential
    # This means that the backward() function must either be marked as @auto_update and have **kwargs, or have keyword arguments for update and lr
    # Certain layers are not updatable because they don't have parameters to update
    @staticmethod
    def updatable(func):
        func.is_updatable = True
        return func

    # Decorator for automatically calling update_weights() in backward()
    # backward() must include **kwargs or keyword arguments for update or lr
    # Assumes that the return value of backward() is a tuple of a dl_dp for each parameter (in the same order as they were registered)
    @staticmethod
    def auto_update(func= None, *, update:bool= True, lr:float= 1.):
        def decorator(func):
            @Layer.updatable
            def newFunc(self, *args, **kwargs):
                ret = func(self, *args, **kwargs)

                # override default values, if necessary
                update_, lr_ = update, lr
                if 'update' in kwargs:
                    update_ = kwargs['update']
                if 'lr' in kwargs:
                    lr_ = kwargs['lr']

                if update_:
                    self.update_weights(*ret[1:], lr= lr_) # first value of ret is dl_dinput, the rest is dl_dps
                return ret
            return newFunc

        # allows @decorator or @decorator(...) to specify default values
        if func is None:
            return decorator
        else:
            return decorator(func)

    # Expects dl_dp for each layer parameter p
    # If arguments are positional arguments, then it expects the parameters to be passed 
    # in the same order as they were registered (and all registered parameters' dl_dp must be provided; but values can be None)
    # Otherwise the arguments must be keyword arguments (with the same name as the parameter)
    def update_weights(self, *args, lr:float= 1e-3, **kwargs):
        if len(args) > 0:
            if len(args) != len(self.params):
                raise ValueError(f"{self.__class__.__name__}.update_weights: Expected {len(self.params)} parameters but got {len(args)}")
            p_dl = zip(self.params.keys(), args)
        else:
            p_dl = kwargs.items()

        for p, dl_dp in p_dl:
            if self.params[p] is not None and dl_dp is None:
                self.params[p] -= lr * dl_dp

    def __repr__(self):
        return f"{self.__class__.__name__}{'' if self.params is None else (
            ' (' + ', '.join([ f'{k}:{v.shape}' for k, v in self.params.items() ]) + ')'
        )}"

    def __str__(self):
        return self.__repr__()

class _LayerCache:
    # Caches keyword arguments
    # Cacje: via cache(k= v), cache.k = v or cache['k'] = v
    # Access: via cache.k or cache['k']
    # Delete: via setting attribute to None or using "del" keyword
    # If it does not exist, returns None
    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                delattr(self, k)
            else:
                setattr(self, k, v)
        return self

    # Python calls __getattr__ when attribute not found
    def __getattr__(self, _):
        return None

    # Allow cache['k'] access
    def __getitem__(self, k):
        return getattr(self, k)

    # Allow cache['k'] = v assignment
    def __setitem__(self, k, v):
        if v is None:
            delattr(self, k)
        else:
            setattr(self, k, v)

    # del keyword
    def __delattr__(self, k):
        delattr(self, k)