import numpy as np


class YeeLattice:
    """
    Yee lattice class

    Parameters
    ----------
    dims : list, ndarray
        Length of each dimension in the domain
    dx : float
        Lattice spacing
    courant : float (default: None)
        Courant number for the simulation. Default is the maximum stable value.
    field_dtype : dtype (default: np.double)
        Data type for the E and H fields
    """
    MAX_COURANT_NUMBERS = [1/np.sqrt(d) for d in [1,2,3]]
    VALID_DTYPES = [np.single, np.double, np.longdouble, np.csingle, np.cdouble, np.clongdouble]
    ETA0 = 376.730313668 # Impedance of free space (ohms)
    C0 = 299792458 # speed of light in vacuum (m/s)


    def __init__(self, size, dx, courant=None, field_dtype=np.double):
        self.size = np.array(size)
        self.dims = np.ceil(self.size/dx).astype(int)
        self.D = len(self.dims)
        self.dx = dx
        self.courant = courant
        self.dt = self._courant * self.dx / self.C0

        # Vector fields
        self.field_dtype = field_dtype
        self.E = np.zeros((*self.dims, self.D), dtype=self.field_dtype)
        self.H = np.zeros((*self.dims, self.D), dtype=self.field_dtype)
        self.J = np.zeros((*self.dims, self.D), dtype=self.field_dtype)

        # Scalar fields
        self._eps = 1
        self._mu = 1
        self._sigma = 0

        self._t = 0

        self._pml = False
        self._Phi = 0
        self._Psi = 0

        self._update_coefs()





    @property
    def D(self):
        return self._D
    @D.setter
    def D(self, val):
        if val in [1,2,3]:
            self._D = val
        else:
            raise ValueError("Dimension must be 1,2, or 3")


    @property
    def courant(self):
        """
        Courant number
        """
        return self._courant
    @courant.setter
    def courant(self, val):
        max_courant = self.MAX_COURANT_NUMBERS[self.D-1]
        if val is None:
            val = max_courant - 1e-9
        if val <= max_courant:
            self._courant = val - 1e-9
        else:
            raise ValueError(f"Unstable Courant number. Max value in {self.D} " \
                             +"dimensions is {max_courant:.3f}")
        

    @property
    def field_dtype(self):
        """
        Data type for the fields (real/complex, single/double)
        """
        return self._field_dtype
    @field_dtype.setter
    def field_dtype(self, val):
        if val in self.VALID_DTYPES:
            self._field_dtype = val
        else:
            raise ValueError("Unsupported datatype")
        
    
    @property
    def eps(self):
        """ 
        Relative permittivity of the domain
        """
        return self._eps
    @eps.setter
    def eps(self, val):
        if not isinstance(val, np.ndarray): val = np.array(val)
        if np.all(val==1):
            self._eps = 1
        elif np.all(val.shape == (*self.dims,1)):
            self._eps = val.astype(self.field_dtype)
        else:
            raise TypeError(f"Dimensions must match the underlying lattice: {self.dims}")
        self._update_coefs()


    @property
    def mu(self):
        """ 
        Relative permeability of the domain
        """
        return self._mu
    @mu.setter
    def mu(self, val):
        if not isinstance(val, np.ndarray): val = np.array(val)
        if np.all(val==1):
            self._mu = 1
        elif np.all(val.shape == (*self.dims,1)):
            self._mu = val.astype(self.field_dtype)
        else:
            raise TypeError(f"Dimensions must match the underlying lattice: {self.dims}")
        self._update_coefs()


    @property
    def sigma(self):
        """ 
        Conductivity of the domain
        """
        return self._mu
    @sigma.setter
    def sigma(self, val):
        if not isinstance(val, np.ndarray): val = np.array(val)
        if np.all(val==1):
            self._sigma = 1
        elif np.all(val.shape == (*self.dims,1)):
            self._sigma = val.astype(self.field_dtype)
        else:
            raise TypeError(f"Dimensions must match the underlying lattice: {self.dims}")
        self._update_coefs()


    @property
    def pml(self):
        """
        Flag for Perfectly Matched Layer (PML)
        """
        return self._pml
    @pml.setter
    def pml(self, val):
        if val in [0,1,True,False]:
            self._pml = val
            self._update_coefs()
        else:
            raise ValueError("pml attribute must be boolean")
        

    def _update_coefs(self):
        c = np.divide(self.dt*self._sigma, 2*self._eps)
        self._htoe = np.divide(  np.divide(self._courant * self.ETA0 , self._eps) , 1+c)
        self._etoe = np.divide( 1-c, 1+c )
        self._etoh = self._courant * np.divide(1, self._mu*self.ETA0)

        if self._pml:
            L = 50
            self.boundary_e = np.s_[-L:,:,:]
            self.boundary_w = np.s_[:L,:,:]
            self.boundary_n = np.s_[:,-L:,:]
            self.boundary_s = np.s_[:,:L,:]
            self.boundary_u = np.s_[:,:,-L:]
            self.boundary_d = np.s_[:,:,:L]

            sigma = 1e9*(np.tanh( 5*np.linspace(-1,1,L) ) + 1)/2
            kappa = 1
            alpha = 2e9
            self.pmlsigma = sigma
            a = np.exp(-(sigma/kappa + alpha)*self.dt)
            a = np.reshape(a, (L,1,1))
            self.pmla = a

            cfactor = np.divide(sigma , sigma*kappa + kappa**2*alpha)
            cfactor = np.reshape(cfactor, (L,1,1))

            self._ax = np.ones(self.dims, self.field_dtype)
            self._cx = np.zeros(self.dims, self.field_dtype)
            tiling_x = (1, self.dims[1], self.dims[2])
            self._ax[self.boundary_e] = np.tile(a, tiling_x)
            self._ax[self.boundary_w] = np.tile(np.flip(a), tiling_x)
            self._cx[self.boundary_e] = np.multiply(np.tile(cfactor, tiling_x), self._ax[self.boundary_e] - 1) 
            self._cx[self.boundary_w] = np.multiply(np.tile(np.flip(cfactor), tiling_x), self._ax[self.boundary_w] - 1)

            self._ay = np.ones(self.dims, self.field_dtype)
            self._cy = np.zeros(self.dims, self.field_dtype)
            tiling_y = (self.dims[0], 1, self.dims[2])
            self._ay[self.boundary_n] = np.tile(np.reshape(a, (1,L,1)), tiling_y)
            self._ay[self.boundary_s] = np.tile(np.reshape(np.flip(a), (1,L,1)), tiling_y)
            self._cy[self.boundary_n] = np.multiply(np.tile(np.reshape(cfactor, (1,L,1)), tiling_y), self._ay[self.boundary_n] - 1) 
            self._cy[self.boundary_s] = np.multiply(np.tile(np.reshape(np.flip(cfactor), (1,L,1)), tiling_y), self._ay[self.boundary_s] - 1)

            
            self._az = np.ones(self.dims, self.field_dtype)
            self._cz = np.zeros(self.dims, self.field_dtype)#np.multiply(cfactor, self._az - 1) 

            # tiling_z = (self.dims[0], self.dims[1], 1)
            # self._az[self.boundary_u] = np.tile(np.reshape(a, (1,1,L)), tiling_z)
            # self._az[self.boundary_d] = np.tile(np.reshape(np.flip(a), (1,1,L)), tiling_z)

            self._a = np.zeros((*self.dims, self.D), self.field_dtype)
            self._a[:,:,:,0] = self._ax
            self._a[:,:,:,1] = self._ay
            self._a[:,:,:,2] = self._az

            self._Phi = np.zeros((*self.dims, self.D), self.field_dtype)
            self._Psi = np.zeros((*self.dims, self.D), self.field_dtype)





    def curl(self, F : np.ndarray, put_on_half_grid):
        """
        Compute the curl of a vector field over R3
        Parameters
        ----------
        F : ndarray
            Vector field stored in ndarray of size [Lx,Ly,Lz,3]
        put_on_half_grid : bool
            Put output on half-integer positions in grid
        """
        curlF = np.zeros(F.shape)
        Fx = F[:,:,:,0]
        Fy = F[:,:,:,1]
        Fz = F[:,:,:,2]
        if put_on_half_grid:
            # curlF_x = dFz/dy - dFy/dz
            curlF[:, 1:, :, 0] += np.diff(Fz, axis=1) # dFz/dy
            curlF[:, :, 1:, 0] -= np.diff(Fy, axis=2) # dFy/dz 
            # curlF_y = dFx/dz - dFz/dx
            curlF[:, :, 1:, 1] += np.diff(Fx, axis=2) # dFx/dz
            curlF[1:, :, :, 1] -= np.diff(Fz, axis=0) # dFz/dx
            # curlF_z = dFy/dx - dFx/dy
            curlF[1:, :, :, 2] += np.diff(Fy, axis=0) # dFy/dx
            curlF[:, 1:, :, 2] -= np.diff(Fx, axis=1) # dFx/dy
        else:
            # curlF_x = dFz/dy - dFy/dz
            curlF[:, :-1, :, 0] += np.diff(Fz, axis=1) # dFz/dy
            curlF[:, :, :-1, 0] -= np.diff(Fy, axis=2) # dFy/dz 
            # curlF_y = dFx/dz - dFz/dx
            curlF[:, :, :-1, 1] += np.diff(Fx, axis=2) # dFx/dz
            curlF[:-1, :, :, 1] -= np.diff(Fz, axis=0) # dFz/dx
            # curlF_z = dFy/dx - dFx/dy
            curlF[:-1, :, :, 2] += np.diff(Fy, axis=0) # dFy/dx
            curlF[:, :-1, :, 2] -= np.diff(Fx, axis=1) # dFx/dy

        return curlF
    

    def run(self, ttot):
        """
        Run simulation forward in time by a given duration
        Parameters
        ----------
        ttot : float
            Duration to propagate fields forward in time
        """
        time_steps = np.floor(ttot/self.dt).astype(int)
        for _ in range(time_steps):
            self._update_source(self._t)

            if self._pml:
                self._Phi = self._update_convpml(self._Phi, self.E, True)

            # Take curl of E(n) and update H(n-1/2) to H(n+1/2)
            curlE = self.curl(self.E, True) + self._Phi
            self.H = self.H - np.multiply(self._etoh, curlE)

            if self._pml:
                self._Psi = self._update_convpml(self._Psi, self.H, False)

            # Take curl of H(n+1/2) and update E(n) to E(n+1)
            curlH = self.curl(self.H, False) + self._Psi
            self.E = np.multiply(self._etoe, self.E) + np.multiply(self._htoe, curlH) -  self.J


            self._t += self.dt


    def reset(self):
        """
        Reset the domain to initial conditions
        """
        self._t = 0
        self.E = np.zeros((*self.dims, 3), dtype=self.field_dtype)
        self.H = np.zeros((*self.dims, 3), dtype=self.field_dtype)
        self._update_source(0)
            

    def _update_source(self, t):
        if hasattr(self, "update_source"):
            self.J = self.update_source(t)


    def _update_convpml(self, psi : np.ndarray, F : np.ndarray, put_on_half_grid):
        """
        Compute the curl of a vector field over R3 for convolutional PML
        Parameters
        ----------
        psi : ndarray
            Convolution vector field in PML
        F : ndarray
            Vector field stored in ndarray of size [Lx,Ly,Lz,3]
        put_on_half_grid : bool
            Put output on half-integer positions in grid
        """
        newpsi = np.zeros_like(psi)
        Fx = F[:,:,:,0]
        Fy = F[:,:,:,1]
        Fz = F[:,:,:,2]
        psi_x = psi[:,:,:,0]
        psi_y = psi[:,:,:,1]
        psi_z = psi[:,:,:,2]
        
        if put_on_half_grid:            
            # curlF_x = dFz/dy - dFy/dz
            newpsi[:, 1:, :, 0] += np.multiply(self._cy[:,1:,:], np.diff(Fz, axis=1)) + np.multiply(self._ay[:,1:,:], psi_x[:,1:,:]) # dFz/dy
            newpsi[:, :, 1:, 0] -= np.multiply(self._cz[:,:,1:], np.diff(Fy, axis=2)) + np.multiply(self._az[:,:,1:], psi_x[:,:,1:]) # dFy/dz 
            # curlF_y = dFx/dz - dFz/dx
            newpsi[:, :, 1:, 1] += np.multiply(self._cz[:,:,1:], np.diff(Fx, axis=2)) + np.multiply(self._az[:,:,1:], psi_y[:,:,1:]) # dFx/dz
            newpsi[1:, :, :, 1] -= np.multiply(self._cx[1:,:,:], np.diff(Fz, axis=0)) + np.multiply(self._ax[1:,:,:], psi_y[1:,:,:]) # dFz/dx
            # curlF_z = dFy/dx - dFx/dy
            newpsi[1:, :, :, 2] += np.multiply(self._cx[1:,:,:], np.diff(Fy, axis=0)) + np.multiply(self._ax[1:,:,:], psi_z[1:,:,:]) # dFy/dx
            newpsi[:, 1:, :, 2] -= np.multiply(self._cy[:,1:,:], np.diff(Fx, axis=1)) + np.multiply(self._ay[:,1:,:], psi_z[:,1:,:]) # dFx/dy
        else:
            # curlF_x = dFz/dy - dFy/dz
            newpsi[:, :-1, :, 0] += np.multiply(self._cy[:,:-1,:], np.diff(Fz, axis=1)) + np.multiply(self._ay[:,:-1,:], psi_x[:,:-1,:]) # dFz/dy
            newpsi[:, :, :-1, 0] -= np.multiply(self._cz[:,:,:-1], np.diff(Fy, axis=2)) + np.multiply(self._az[:,:,:-1], psi_x[:,:,:-1]) # dFy/dz 
            # curlF_y = dFx/dz - dFz/dx
            newpsi[:, :, :-1, 1] += np.multiply(self._cz[:,:,:-1], np.diff(Fx, axis=2)) + np.multiply(self._az[:,:,:-1], psi_y[:,:,:-1]) # dFx/dz
            newpsi[:-1, :, :, 1] -= np.multiply(self._cx[:-1,:,:], np.diff(Fz, axis=0)) + np.multiply(self._ax[:-1,:,:], psi_y[:-1,:,:]) # dFz/dx
            # curlF_z = dFy/dx - dFx/dy
            newpsi[:-1, :, :, 2] += np.multiply(self._cx[:-1,:,:], np.diff(Fy, axis=0)) + np.multiply(self._ax[:-1,:,:], psi_z[:-1,:,:]) # dFy/dx
            newpsi[:, :-1, :, 2] -= np.multiply(self._cy[:,:-1,:], np.diff(Fx, axis=1)) + np.multiply(self._ay[:,:-1,:], psi_z[:,:-1,:]) # dFx/dy

        return newpsi
    

    def space2grid(self, x):
        """
        Convert spatial coordinates to corresponding Yee cell indices
        """
        x_dx = x/self.dx # position in units of lattice distance dx
        xidx = np.floor(x_dx).astype(int)
        return xidx
    
    def grid2space(self, xidx):
        """
        Convert given Yee cell indices into real spatial coordinates
        """
        x = xidx * self.dx
        if not self.inbounds(x):
            raise ValueError("Input is outside of grid bounds")
        return x
    
    def inbounds(self, x):
        """
        Check if spatial position is within domain
        """
        return np.all(x <= self.size)
    

    def gridloc(self, fieldcomp):
        """
        Get Yee grid locations where the given field component is computed.
        For example, gridloc("Ex") returns the coordinates within each
        Yee cell where Ex is computed.

        Parameters
        ----------
        fieldcomp : str
            Field component. Must be "Ex", "Ey", "Ez", "Hx", "Hy", or "Hz". Case insensitive.
        """
        mx = np.arange(self.dims[0])
        my = np.arange(self.dims[1])
        mz = np.arange(self.dims[2])

        Mx,My,Mz = np.meshgrid(mx,my,mz, indexing='ij')
        loc = np.zeros([*self.dims,self.D], dtype=np.single)

        comp = fieldcomp.lower()
        if comp == "ex":
            loc[:,:,:,0] = Mx + 1/2 
            loc[:,:,:,1] = My
            loc[:,:,:,2] = Mz
        elif comp == "ey":
            loc[:,:,:,0] = Mx
            loc[:,:,:,1] = My + 1/2
            loc[:,:,:,2] = Mz
        elif comp ==  "ez":
            loc[:,:,:,0] = Mx
            loc[:,:,:,1] = My
            loc[:,:,:,2] = Mz + 1/2
        elif comp == "hx":
            loc[:,:,:,0] = Mx
            loc[:,:,:,1] = My + 1/2
            loc[:,:,:,2] = Mz + 1/2
        elif comp == "hy":
            loc[:,:,:,0] = Mx + 1/2
            loc[:,:,:,1] = My
            loc[:,:,:,2] = Mz + 1/2
        elif comp == "hz":
            loc[:,:,:,0] = Mx + 1/2
            loc[:,:,:,1] = My + 1/2
            loc[:,:,:,2] = Mz

        return loc





