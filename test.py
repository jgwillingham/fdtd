from yee import YeeLattice
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import logging
logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
EPS0 = 8.85418782e-12

# dimensions in meters
Lx = 1e3
Ly = 1e3
Lz = 1

L = 1
locx = 500
locy = 500
locz = 0

wavelen = 245 # in meters
pts_per_wavelen = 50

pow = True

pml = False

#------------------------------------------------
dx = wavelen/pts_per_wavelen
logging.info(f"Lattice spacing: {dx:.3f} m")

locxIdx = int(locx//dx)
locyIdx = int(locy//dx)
loczIdx = int(locz//dx)
L = int(L//dx)
logging.info(f"Emitter loc: ({locxIdx}, {locyIdx}, {loczIdx})")
logging.info(f"Emitter len: {L} ")


F = YeeLattice([Lx,Ly,Lz], dx, field_dtype=np.double)
logging.info(f"Time step: {F.dt/1e-9:.3f} ns")

F.pml = pml

J = np.zeros((*F.dims,3), dtype=np.float64)
J[locxIdx,(locyIdx-L//2):(locyIdx+L//2+1),loczIdx,1] = np.ones((L+1,1))#*F.dt/EPS0

freq = YeeLattice.C0 / wavelen
logging.info(f"Frequency: {freq/1e6:.3f} MHz")
logging.info(f"Period: {1/freq*1e6:.3f} us")

F.update_source = lambda t: np.sin(2*np.pi*freq*t)*J

#sigma = np.zeros((*F.dims,1))
#sigma[:,:5,:,:] = 1e12
#sigma[:,-6:-1,:,:] = 1e12
#sigma[:5,:,:,:] = 1e12
#sigma[-6:-1,:,:,:] = 1e12


#F.sigma = sigma

steptot = 200
stepsize = 5

exampleJ = [F.update_source(n*F.dt)[locxIdx,locyIdx,loczIdx,1] for n in range(steptot*stepsize)]
plt.plot(F.dt*np.arange(steptot*stepsize)/1e-9, exampleJ)
plt.xlabel("Time (ns)")
plt.ylabel("Jy")
plt.show()

F.reset()


if pml:
    plt.imshow(F.kappa[:,:,0,0])
    plt.title('Kappa')
    plt.show()

    plt.plot(F.kappa[:,0,0,0])
    plt.show()

    plt.imshow(F.pmlsigma[...,0,0])
    plt.title('Sigma')
    plt.show()

    plt.plot(F.pmlsigma[:,0,0,0])
    plt.show()

    plt.imshow(F.alpha[...,0,0])
    plt.title('alpha')
    plt.show()
    plt.plot(F.alpha[:,0,0,0])
    plt.title('alpha')
    plt.show()

    plt.imshow(F.xi[...,0,0])
    plt.title('xi')
    plt.show()

    plt.plot(F.xi[:,0,0,0])
    plt.title('xi')
    plt.show()

    plt.imshow(F.QEcoef1[...,0,0])
    plt.title('QEcoef1')
    plt.show()

    plt.plot(F.QEcoef1[:,0,0,0])
    plt.title('QEcoef1')
    plt.show()


    plt.imshow(F.QEcoef2[...,0,0])
    plt.title('QEcoef2')
    plt.show()

    plt.plot(F.QEcoef2[:,0,0,0])
    plt.title('QEcoef2')
    plt.show()

frames = []
fig = plt.figure()
for i in range(steptot):
    F.run(F.dt*stepsize)
    if not pow:
        frames.append( [plt.imshow(np.transpose(F.E[:,:,loczIdx,0]), cmap='seismic', animated=True, \
                                   origin='lower', extent=[0,Lx,0,Ly])] )
        minval,maxval= plt.gci().get_clim()
        plt.clim(-maxval/20, maxval/20)
    else:
        frames.append( [plt.imshow( np.transpose(20*np.log10(np.abs(F.E[:,:,loczIdx,0])+np.finfo(float).eps)), cmap='turbo',\
                                    animated=True, origin='lower', extent=[0,Lx,0,Ly])])
        #plt.clim(0, 80)
        plt.clim(-70,70)
    print(f"Progress: {(i+1)/steptot*100:.2f}%", end='\r')

if pow:
    plt.colorbar(label="dB (V/m)")
else:
    plt.colorbar(label="V/m")
plt.title(f"Ex: f={freq/1e6:.3f} MHz")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

vid = animation.ArtistAnimation(fig, frames, interval=10, blit=True, repeat_delay=0)
plt.show()
