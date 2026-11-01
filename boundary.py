from dataclasses import dataclass
import numpy as np
from .yee import *


@dataclass
class PMLParams:
    depth : float
    sigma : float = 0.0
    kappa : float = 1.0
    alpha : float = 0.0
    polydeg : int = 3




class PML:
    def __init__(self, depth, sigma, kappa=1, alpha=0.05, polydeg=3):
        self._depth = depth
        self.sigmamax = sigma
        self.kappamax = kappa
        self.alphamax = alpha
        self._polydeg = polydeg
        self.params = PMLParams(depth, sigma, kappa, alpha, polydeg)
        

    @property
    def sigmamax(self):
        return self._sigmamax
    @sigmamax.setter
    def sigmamax(self, val):
        self._sigmamax = val
        self.sigma = lambda z : self._sigmamax * np.abs(z/self._depth)**self._polydeg # polynomial scaling for sigma. Min at interface, max at edge


    @property
    def kappamax(self):
        return self._kappamax
    @kappamax.setter
    def kappamax(self, val):
        if val is None:
            val = (self._polydeg+1) / (150*np.pi*self.dx)
        self._kappamax = val
        self.kappa = lambda z : self._kappamax * np.abs(z/self._depth)**self._polydeg # polynomial scaling for kappa. Min at interface, max at edge


    @property
    def alphamax(self):
        return self._alphamax
    @alphamax.setter
    def alphamax(self, val):
        self._alphamax = val
        self.alpha = lambda z : self._sigmamax * (1 - np.abs(z/self._depth)) # linear scaling for alpha. Max at interface, min at edge


    def attach(self, yeelattice : YeeLattice):
        """
        Attach the PML to a given YeeLattice instance. Extends the domain to include the PML on all sides
        """
        newsize = [L + 2*self.depth for L in yeelattice.size]
        yeelattice.update_dims(newsize, yeelattice.dx)
        self.lattice = yeelattice

        self._QE = np.zeros((*self.lattice.dims, self.lattice.D, self.lattice.D), self.lattice.field_dtype)
        self._QH = np.zeros((*self.lattice.dims, self.lattice.D, self.lattice.D), self.lattice.field_dtype)



    def _update_QE(self):
        """
        Update the auxiliary fields Q^E 
        """
        self._QE[:,:,:,0,1] = np.multiply(self.params.QEcoef1[:,:,:,0], self._QE[:,:,:,0,1])
        self._QE[:,:,:,0,2] = np.multiply(self.params.QEcoef1[:,:,:,0], self._QE[:,:,:,0,2])
        self._QE[:,:,:,1,2] = np.multiply(self.params.QEcoef1[:,:,:,1], self._QE[:,:,:,1,2])
        self._QE[:,:,:,1,0] = np.multiply(self.params.QEcoef1[:,:,:,1], self._QE[:,:,:,1,0])
        self._QE[:,:,:,2,0] = np.multiply(self.params.QEcoef1[:,:,:,2], self._QE[:,:,:,2,0])
        self._QE[:,:,:,2,1] = np.multiply(self.params.QEcoef1[:,:,:,2], self._QE[:,:,:,2,1])

        self._QE[1:,:,:,0,1] -= np.multiply(self.params.QEcoef2[1:,:,:,0], np.diff(self.E[:,:,:,1], axis=0))
        self._QE[1:,:,:,0,2] -= np.multiply(self.params.QEcoef2[1:,:,:,0], np.diff(self.E[:,:,:,2], axis=0))
        self._QE[:,1:,:,1,2] -= np.multiply(self.params.QEcoef2[:,1:,:,1], np.diff(self.E[:,:,:,2], axis=1))
        self._QE[:,1:,:,1,0] -= np.multiply(self.params.QEcoef2[:,1:,:,1], np.diff(self.E[:,:,:,0], axis=1))
        self._QE[:,:,1:,2,0] -= np.multiply(self.params.QEcoef2[:,:,1:,2], np.diff(self.E[:,:,:,0], axis=2))
        self._QE[:,:,1:,2,1] -= np.multiply(self.params.QEcoef2[:,:,1:,2], np.diff(self.E[:,:,:,1], axis=2))


    def _update_QH(self):
        """
        Update the auxiliary fields Q^H
        """
        self._QH[:,:,:,0,1] = np.multiply(self.pmlparams.QHcoef1[:,:,:,0], self._QH[:,:,:,0,1])
        self._QH[:,:,:,0,2] = np.multiply(self.pmlparams.QHcoef1[:,:,:,0], self._QH[:,:,:,0,2])
        self._QH[:,:,:,1,2] = np.multiply(self.pmlparams.QHcoef1[:,:,:,1], self._QH[:,:,:,1,2])
        self._QH[:,:,:,1,0] = np.multiply(self.pmlparams.QHcoef1[:,:,:,1], self._QH[:,:,:,1,0])
        self._QH[:,:,:,2,0] = np.multiply(self.pmlparams.QHcoef1[:,:,:,2], self._QH[:,:,:,2,0])
        self._QH[:,:,:,2,1] = np.multiply(self.pmlparams.QHcoef1[:,:,:,2], self._QH[:,:,:,2,1])

        self._QH[:-1,:,:,0,1] -= np.multiply(self.pmlparams.QHcoef2[:-1,:,:,0], np.diff(self.H[:,:,:,1], axis=0))
        self._QH[:-1,:,:,0,2] -= np.multiply(self.pmlparams.QHcoef2[:-1,:,:,0], np.diff(self.H[:,:,:,2], axis=0))
        self._QH[:,:-1,:,1,2] -= np.multiply(self.pmlparams.QHcoef2[:,:-1,:,1], np.diff(self.H[:,:,:,2], axis=1))
        self._QH[:,:-1,:,1,0] -= np.multiply(self.pmlparams.QHcoef2[:,:-1,:,1], np.diff(self.H[:,:,:,0], axis=1))
        self._QH[:,:,:-1,2,0] -= np.multiply(self.pmlparams.QHcoef2[:,:,:-1,2], np.diff(self.H[:,:,:,0], axis=2))
        self._QH[:,:,:-1,2,1] -= np.multiply(self.pmlparams.QHcoef2[:,:,:-1,2], np.diff(self.H[:,:,:,1], axis=2))

    
    

    
        