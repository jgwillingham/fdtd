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
        self.update_dims(size, dx, courant)

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

        self._update_coefs()
        self.kappa= np.ones((*self.dims,self.D), dtype=self.field_dtype)


    def update_dims(self, size, dx, courant=None):
        self.size = np.array(size)
        self.dims = np.ceil(self.size/dx).astype(int)
        self.D = len(self.dims)
        self.dx = dx
        self.courant = courant
        self.dt = self._courant * self.dx / self.C0

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
        if val:
            self._pml = True
            self._update_coefs()
        else:
            self._pml = False
        

    def _update_coefs(self):
        c = np.divide(self.dt*self._sigma, 2*self._eps)
        self._htoe = np.divide(  np.divide(self._courant * self.ETA0 , self._eps) , 1+c)
        self._etoe = np.divide( 1-c, 1+c )
        self._etoh = self._courant * np.divide(1, self._mu*self.ETA0)

        if self._pml:

            depth = 200

            L = np.ceil(depth/self.dx).astype(int)
            self.boundary_e = np.s_[-L:,:,:,0]
            self.boundary_w = np.s_[:L,:,:,0]
            self.boundary_n = np.s_[:,-L:,:,1]
            self.boundary_s = np.s_[:,:L,:,1]
            self.boundary_u = np.s_[:,:,-L:,2]
            self.boundary_d = np.s_[:,:,:L,2]

            EPS0 = 8.85418782e-12
            m = 3
            sigmamax = (m+1)/(150*np.pi*self.dx)/EPS0
            kappamax = 15
            alphamax = 0.05/EPS0

            sigma_fn = lambda z : sigmamax * np.abs(z/depth)**m
            kappa_fn = lambda z : kappamax * (1 - np.abs(z/depth)**m)
            alpha_fn = lambda z : alphamax * np.abs(z/depth)

            dist_from_boundary = np.arange(L)*self.dx

            tilex = (1, self.dims[1], self.dims[2])
            tiley = (self.dims[0], 1, self.dims[2])
            tilez = (self.dims[0], self.dims[1], 1)

            s = sigma_fn(dist_from_boundary)  
            sx = np.reshape( s, (L,1,1) )
            sy = np.reshape( s, (1,L,1) )
            sz = np.reshape( s, (1,1,L) )
            sigma = np.zeros((*self.dims, self.D), self.field_dtype)
            sigma[self.boundary_e] = np.tile(sx, tilex)
            sigma[self.boundary_w] = np.tile(np.flip(sx), tilex)
            sigma[self.boundary_n] = np.tile(sy, tiley)
            sigma[self.boundary_s] = np.tile(np.flip(sy), tiley)
            #sigma[self.boundary_u] = np.tile(sz, tilez)
            #sigma[self.boundary_d] = np.tile(np.flip(sz), tilez)

            k = kappa_fn(dist_from_boundary)            
            kx = np.reshape( k, (L,1,1) )
            ky = np.reshape( k, (1,L,1) )
            kz = np.reshape( k, (1,1,L) )
            kappa = np.ones((*self.dims, self.D), self.field_dtype) # kappa = 1 in free space
            kappa[self.boundary_e] = np.tile(kx, tilex)
            kappa[self.boundary_w] = np.tile(np.flip(kx), tilex)
            kappa[self.boundary_n] = np.tile(ky, tiley)
            kappa[self.boundary_s] = np.tile(np.flip(ky), tiley)
            #kappa[self.boundary_u] = np.tile(kz, tilez)
            #kappa[self.boundary_d] = np.tile(np.flip(kz), tilez)

            a = alpha_fn(dist_from_boundary)            
            ax = np.reshape( a, (L,1,1) )
            ay = np.reshape( a, (1,L,1) )
            az = np.reshape( a, (1,1,L) )
            alpha = np.zeros((*self.dims, self.D), self.field_dtype)
            alpha[self.boundary_e] = np.tile(ax, tilex)
            alpha[self.boundary_w] = np.tile(np.flip(ax), tilex)
            alpha[self.boundary_n] = np.tile(ay, tiley)
            alpha[self.boundary_s] = np.tile(np.flip(ay), tiley)
            #alpha[self.boundary_u] = np.tile(az, tilez)
            #alpha[self.boundary_d] = np.tile(np.flip(az), tilez)

            xi = np.add(alpha, np.divide(sigma, kappa))

            #self.QEcoef1 = np.divide( 2-xi*self.dt, 2+xi*self.dt )
            self.QEcoef1 = np.divide( 1, 1 + xi*self.dt)
            #self.QEcoef2 = np.divide( 2*sigma*self.dt, np.multiply( np.power(kappa,2), 2+xi*self.dt ) )
            self.QEcoef2 = np.divide( sigma*self.dt, np.multiply( np.power(kappa,2), 1+xi*self.dt ) )
            self.QHcoef1 = self.QEcoef1
            self.QHcoef2 = self.QEcoef2

            self._QE = np.zeros((*self.dims, 3, 3), self.field_dtype)
            self._QH = np.zeros((*self.dims, 3, 3), self.field_dtype)

            self.kappa = kappa
            self.pmlsigma = sigma
            self.alpha = alpha
            self.xi = xi





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
            kx = self.kappa[1:,:,:,0]
            ky = self.kappa[:,1:,:,1]
            kz = self.kappa[:,:,1:,2]   

            # curlF_x = dFz/dy - dFy/dz
            curlF[:, 1:, :, 0] += np.divide( np.diff(Fz, axis=1), ky )# dFz/dy
            curlF[:, :, 1:, 0] -= np.divide( np.diff(Fy, axis=2), kz ) # dFy/dz 
            # curlF_y = dFx/dz - dFz/dx
            curlF[:, :, 1:, 1] += np.divide( np.diff(Fx, axis=2), kz ) # dFx/dz
            curlF[1:, :, :, 1] -= np.divide( np.diff(Fz, axis=0), kx ) # dFz/dx
            # curlF_z = dFy/dx - dFx/dy
            curlF[1:, :, :, 2] += np.divide( np.diff(Fy, axis=0), kx ) # dFy/dx
            curlF[:, 1:, :, 2] -= np.divide( np.diff(Fx, axis=1), ky ) # dFx/dy
        else:
            kx = self.kappa[:-1,:,:,0]
            ky = self.kappa[:,:-1,:,1]
            kz = self.kappa[:,:,:-1,2]   

            # curlF_x = dFz/dy - dFy/dz
            curlF[:, :-1, :, 0] += np.divide( np.diff(Fz, axis=1), ky ) # dFz/dy
            curlF[:, :, :-1, 0] -= np.divide( np.diff(Fy, axis=2), kz )# dFy/dz 
            # curlF_y = dFx/dz - dFz/dx
            curlF[:, :, :-1, 1] += np.divide( np.diff(Fx, axis=2), kz ) # dFx/dz
            curlF[:-1, :, :, 1] -= np.divide( np.diff(Fz, axis=0), kx ) # dFz/dx
            # curlF_z = dFy/dx - dFx/dy
            curlF[:-1, :, :, 2] += np.divide( np.diff(Fy, axis=0), kx ) # dFy/dx
            curlF[:, :-1, :, 2] -= np.divide( np.diff(Fx, axis=1), ky ) # dFx/dy

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
                self._update_QE()

            # Take curl of E(n) and update H(n-1/2) to H(n+1/2)
            curlE = self.curl(self.E, True)

            if self._pml:
                curlE[:,:,:,0] += self._QE[:,:,:,1,2] - self._QE[:,:,:,2,1]
                curlE[:,:,:,1] += self._QE[:,:,:,2,0] - self._QE[:,:,:,0,2]
                curlE[:,:,:,2] += self._QE[:,:,:,0,1] - self._QE[:,:,:,1,0]

            self.H = self.H - np.multiply(self._etoh, curlE)

            if self._pml:
                self._update_QH()

            # Take curl of H(n+1/2) and update E(n) to E(n+1)
            curlH = self.curl(self.H, False) 

            if self._pml:
                curlH[:,:,:,0] += self._QH[:,:,:,1,2] - self._QH[:,:,:,2,1]
                curlH[:,:,:,1] += self._QH[:,:,:,2,0] - self._QH[:,:,:,0,2]
                curlH[:,:,:,2] += self._QH[:,:,:,0,1] - self._QH[:,:,:,1,0]

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


    def _update_QE(self):
        """
        Update the auxiliary fields Q^E 
        """
        self._QE[:,:,:,0,1] = np.multiply(self.QEcoef1[:,:,:,0], self._QE[:,:,:,0,1])
        self._QE[:,:,:,0,2] = np.multiply(self.QEcoef1[:,:,:,0], self._QE[:,:,:,0,2])
        self._QE[:,:,:,1,2] = np.multiply(self.QEcoef1[:,:,:,1], self._QE[:,:,:,1,2])
        self._QE[:,:,:,1,0] = np.multiply(self.QEcoef1[:,:,:,1], self._QE[:,:,:,1,0])
        self._QE[:,:,:,2,0] = np.multiply(self.QEcoef1[:,:,:,2], self._QE[:,:,:,2,0])
        self._QE[:,:,:,2,1] = np.multiply(self.QEcoef1[:,:,:,2], self._QE[:,:,:,2,1])

        self._QE[:-1,:,:,0,1] -= np.multiply(self.QEcoef2[:-1,:,:,0], np.diff(self.E[:,:,:,1], axis=0))
        self._QE[:-1,:,:,0,2] -= np.multiply(self.QEcoef2[:-1,:,:,0], np.diff(self.E[:,:,:,2], axis=0))
        self._QE[:,:-1,:,1,2] -= np.multiply(self.QEcoef2[:,:-1,:,1], np.diff(self.E[:,:,:,2], axis=1))
        self._QE[:,:-1,:,1,0] -= np.multiply(self.QEcoef2[:,:-1,:,1], np.diff(self.E[:,:,:,0], axis=1))
        self._QE[:,:,:-1,2,0] -= np.multiply(self.QEcoef2[:,:,:-1,2], np.diff(self.E[:,:,:,0], axis=2))
        self._QE[:,:,:-1,2,1] -= np.multiply(self.QEcoef2[:,:,:-1,2], np.diff(self.E[:,:,:,1], axis=2))


    def _update_QH(self):
        """
        Update the auxiliary fields Q^H
        """
        self._QH[:,:,:,0,1] = np.multiply(self.QHcoef1[:,:,:,0], self._QH[:,:,:,0,1])
        self._QH[:,:,:,0,2] = np.multiply(self.QHcoef1[:,:,:,0], self._QH[:,:,:,0,2])
        self._QH[:,:,:,1,2] = np.multiply(self.QHcoef1[:,:,:,1], self._QH[:,:,:,1,2])
        self._QH[:,:,:,1,0] = np.multiply(self.QHcoef1[:,:,:,1], self._QH[:,:,:,1,0])
        self._QH[:,:,:,2,0] = np.multiply(self.QHcoef1[:,:,:,2], self._QH[:,:,:,2,0])
        self._QH[:,:,:,2,1] = np.multiply(self.QHcoef1[:,:,:,2], self._QH[:,:,:,2,1])

        self._QH[1:,:,:,0,1] -= np.multiply(self.QHcoef2[1:,:,:,0], np.diff(self.H[:,:,:,1], axis=0))
        self._QH[1:,:,:,0,2] -= np.multiply(self.QHcoef2[1:,:,:,0], np.diff(self.H[:,:,:,2], axis=0))
        self._QH[:,1:,:,1,2] -= np.multiply(self.QHcoef2[:,1:,:,1], np.diff(self.H[:,:,:,2], axis=1))
        self._QH[:,1:,:,1,0] -= np.multiply(self.QHcoef2[:,1:,:,1], np.diff(self.H[:,:,:,0], axis=1))
        self._QH[:,:,1:,2,0] -= np.multiply(self.QHcoef2[:,:,1:,2], np.diff(self.H[:,:,:,0], axis=2))
        self._QH[:,:,1:,2,1] -= np.multiply(self.QHcoef2[:,:,1:,2], np.diff(self.H[:,:,:,1], axis=2))


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





