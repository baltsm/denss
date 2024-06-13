import numpy as np
import matplotlib.pyplot as plt #remember exit()
from mpl_toolkits.mplot3d import Axes3D
import saxstats.saxstats as saxs

DENSS_GPU = False

## Fake variables 
rho = np.random.rand(64, 64, 64)
qdata_file = saxs.loadDatFile('/Users/maijabalts/Desktop/BioXFEL/denss-master/6lyz.dat')
q = qdata_file[0]
I = qdata_file[1]
sigq = qdata_file[2]

## Variable presets
side = 3
D = 50 #max density
voxel = 5
steps=2001
shrinkwrap_sigma_start=3
shrinkwrap_threshold_fraction = 0.2
oversampling = 3
shrinkwrap_sigma_end = 1.5
shrinkwrap_sigma_decay=0.99
shrinkwrap_iter=20
shrinkwrap_minstep=100
chi_end_fraction=0.01
write_freq=100
enforce_connectivity_steps=[500]
enforce_connectivity_max_features=1
ncs=0
ncs_steps=[500]
ncs_axis=1

#creating a box for the protein
side = oversampling*D #creating box size larger than protein
halfside = side/2
n = int(side/voxel) 
#want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
if n%2==1:
    n += 1
#store n for later use if needed
nbox = n

dx = side/n #individual size of voxel
dV = dx**3 
V = side**3
x_ = np.linspace(-halfside,halfside,n)
x,y,z = np.meshgrid(x_,x_,x_,indexing='ij') #creating cartesian coordinates
r = np.sqrt(x**2 + y**2 + z**2)

df = 1/side
qx_ = np.fft.fftfreq(x_.size)*n*df*2*np.pi #converting to frequency space
qz_ = np.fft.rfftfreq(x_.size)*n*df*2*np.pi
# qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
qx, qy, qz = np.meshgrid(qx_,qx_,qz_,indexing='ij') 
qr = np.sqrt(qx**2+qy**2+qz**2) #radius from reciprocal box
qmax = np.max(qr)
qstep = np.min(qr[qr>0]) - 1e-8 #subtract a tiny bit to deal with floating point error
nbins = int(qmax/qstep)
qbins = np.linspace(0,nbins*qstep,nbins+1) #setting up for where you're scattering

#create an array labeling each voxel according to which qbin it belongs
qbin_labels = np.searchsorted(qbins,qr,"right") #labeling voxels
qbin_labels -= 1
qblravel = qbin_labels.ravel()
xcount = np.bincount(qblravel)

#calculate qbinsc as average of q values in shell
qbinsc = saxs.mybinmean(qr.ravel(), qblravel, xcount) 
len(qbinsc)

#allow for any range of q data
qdata = qbinsc[np.where((qbinsc>=q.min()) & (qbinsc<=q.max()) )] #cuts qdata so that it's datapoints only fall within qmax and qmin. 
len(qdata)
Idata = np.interp(qdata,q,I)
len(Idata)

#create list of qbin indices just in region of data for later F scaling
qbin_args = np.in1d(qbinsc,qdata,assume_unique=True)
qba = qbin_args #just for brevity when using it later

sigqdata = np.interp(qdata,q,sigq)

Iq = np.column_stack((q, I, sigq)).T
# Iq = np.vstack((self.q,self.I,self.Ierr)).T
ne = Iq.shape[0] #number of electrons

scale_factor = ne**2 / Idata[0]
Idata *= scale_factor
sigqdata *= scale_factor
I *= scale_factor
sigq *= scale_factor

stepsarr = np.concatenate((enforce_connectivity_steps,[shrinkwrap_minstep]))
maxec = np.max(stepsarr)
steps = int(shrinkwrap_iter * (np.log(shrinkwrap_sigma_end/shrinkwrap_sigma_start)/np.log(shrinkwrap_sigma_decay)) + maxec)
#add enough steps for convergence after shrinkwrap is finished
#something like 7000 seems reasonable, likely will finish before that on its own
#then just make a round number when using defaults
steps += 7621

Imean = np.zeros((len(qbins)))
chi = np.zeros((steps+1))
rg = np.zeros((steps+1))
supportV = np.zeros((steps+1))

support = np.ones(x.shape,dtype=bool) #protein specific
prng = np.random.RandomState()
seed = prng.randint(2**31-1)

prng = np.random.RandomState(seed)

rho = prng.random_sample(size=x.shape) #- 0.5
newrho = np.zeros_like(rho)
sigma = shrinkwrap_sigma_start

swbyvol = True
swV = V/2.0
Vsphere_Dover2 = 4./3 * np.pi * (D/2.)**3
swVend = Vsphere_Dover2
swV_decay = 0.9
first_time_swdensity = True
threshold = shrinkwrap_threshold_fraction
erosion_width = int(20/dx) #this is in pixels

# if erosion_width ==0:
#     #make minimum of one pixel
#     erosion_width = 1

##Run through F transformation
F = saxs.myrfftn(rho, DENSS_GPU=DENSS_GPU)
F[np.abs(F)==0] = 1e-16 #setting anything that's 0 to almost 0

#Create F profile to compare
I3D = saxs.abs2(F) #calculating intensity (magnitude squared)
Imean = saxs.mybinmean(I3D.ravel(), qblravel) #creates profile THIS IS CORRECT
#scale Fs to match data
factors = saxs.mysqrt(Idata/Imean) ##issue with Idata and Imean matching

#do not scale bins outside of desired range
#so set those factors to 1.0
Imean[~qba] = 1.0
F *= factors[qbin_labels] 

#This is the start of the new iteration
rho_new = saxs.myirfftn(F)
rho_new = rho_new.real

#compare profiles
chi= saxs.mysum(((Imean[qba]-Idata[qba])/sigqdata[qba])**2)/Idata[qba].size
print(chi)


#checking code shape
print(np.shape(rho))
print(np.shape(rho_new))
print(np.sum(rho - rho_new)) #Why are they the exact same?
x, y, z = rho_new[:, :, 0].flatten(), rho_new[:, :, 1].flatten(), rho_new[:, :, 2].flatten()
# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='o')
plt.show()