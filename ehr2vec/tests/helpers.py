class FeaturesConfig:
    def __init__(self):
        self.age = {'round': 2}
        self.abspos = {'year': 2020, 'month': 1, 'day': 26}
        self.segment = True
        self.background = ['GENDER']

    def __contains__(self, item):
        return self.__getattribute__(item)
        
    def keys(self):
        return [attr for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))]

    def __iter__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not callable(getattr(self, attr)):
                yield attr

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, item):
        return self.__getattribute__(item)