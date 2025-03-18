import random
import numpy as np

class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]

    def __call__(self, sample):
        return [transform(sample) for transform in self.transforms]


class MultiSimData(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.normal_data = args[1]
        self.args = args[2]

    def __call__(self, sample):
        return [self.transforms[0](sample), self.transforms[1](self.random_slice_data().reshape(1,-1).astype(np.float32))]
    
    def random_slice_data(self):
        key = random.choice(list(self.normal_data.keys()))
        length = len(self.normal_data[key])
        start = random.randint(0, length-self.args.length)
        return self.normal_data[key][start:start+self.args.length]


class DisSimData(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.normal_data = args[1]
        self.args = args[2]

    def __call__(self, sample):
        return [self.transforms[0](sample), self.transforms[1](sample), self.transforms[2](self.random_slice_data().reshape(1,-1).astype(np.float32))]
    
    def random_slice_data(self):
        key = random.choice(list(self.normal_data.keys()))
        length = len(self.normal_data[key])
        start = random.randint(0, length-self.args.length)
        return self.normal_data[key][start:start+self.args.length]
