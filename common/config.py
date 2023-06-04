import importlib
import yaml
import os
from os.path import join
from shutil import copyfile
import logging

class Config(dict):
    """Config class that allows for dot notation."""
    def __init__(self, dictionary=None):
        super(Config, self).__init__()
        if dictionary:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = Config(value)
                elif isinstance(value, str):
                    value = self.str_to_num(value)
                self[key] = value
                setattr(self, key, value)

    def __setattr__(self, key, value):
        if isinstance(value, str):
            value = self.str_to_num(value)
        super(Config, self).__setattr__(key, value)
        super(Config, self).__setitem__(key, value)

    def str_to_num(self, s):
        """Converts a string to a float or int if possible."""
        try:
            return float(s)
        except ValueError:
            return s

    def __setitem__(self, key, value):
        if isinstance(value, str):
            value = self.str_to_num(value)
        super(Config, self).__setitem__(key, value)
        super(Config, self).__setattr__(key, value)

    def __delattr__(self, name):
        if name in self:
            del self[name]
        if hasattr(self, name):
            super(Config, self).__delattr__(name)

    def __delitem__(self, name):
        if name in self:
            del self[name]
        if hasattr(self, name):
            super(Config, self).__delattr__(name)
    
    def yaml_repr(self, dumper):
        return dumper.represent_dict(self.to_dict())
    
    def to_dict(self):
        """Converts the object to a dictionary, including any attributes."""
        result = dict(self)  # Start with the underlying dictionary
        result.update(self.__dict__)  # Add the attributes
        return result


def instantiate(config):
    """Instantiates a class from a config object."""
    module_path, class_name = config._target_.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    kwargs = {k: v for k, v in config.items() if k != "_target_"}
    instance = class_(**kwargs)
    return instance

def get_function(config):
    """Gets a function from a config object."""
    module_path, function_name = config._target_.rsplit(".", 1)
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function

def load_config(config_file):
    """Loads a yaml config file."""
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    cfg = Config(cfg)
    return cfg

