import numpy as np
from .yee import *



class PML(YeeLattice):
    def __init__(self, depth, sigma, kappa=1, a=0):
        self._depth = depth
        self._sigma = sigma
        self._kappa = kappa
        self._a = a
        

    @property
    def mask(self):
        return self._mask
    @mask.setter
    def mask(self, val):
        self._mask = val


    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, fn):
        self._sigma = fn

    
    

    
        