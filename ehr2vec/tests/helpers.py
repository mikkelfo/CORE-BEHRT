class FeaturesConfig:
    def __init__(self):
        self.age = {'round': 2}
        self.abspos = {'year': 2020, 'month': 1, 'day': 26}
        self.segment = True
        self.background = ['GENDER']

    def __contains__(self, item):
        return self.__getattribute__(item)
        
