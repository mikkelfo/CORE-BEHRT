from typing import Any
from unittest.mock import Mock

class ConfigMock(Mock):
    def __contains__(self, item):
        return item in self.__dict__
        
    def keys(self):
        return [attr for attr in self.__dict__ if self._user_attr(attr)]

    def __iter__(self):
        for attr in self.keys():
            if self._user_attr(attr):
                yield attr

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, item):
        return self.__getattribute__(item) if self.__contains__(item) else None
    
    def _user_attr(self, attr):
        return not attr.startswith('_') and not callable(getattr(self, attr)) and attr != 'method_calls'
    
    def get(self, item, default=None):
        return self.__getitem__(item) if self.__contains__(item) else default