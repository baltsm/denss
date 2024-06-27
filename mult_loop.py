import numpy as np
import saxstats.saxstats as saxs

DENSS_GPU = False

qdata_file = saxs.loadDatFile('model1.dat') 
#Null = a, One = b
qdata_null = saxs.loadDatFile("chainA.dat")
qdata_one = saxs.loadDatFile("chainB.dat")

q = qdata_file[0]
I = qdata_file[1] 
sigq = qdata_file[2]

q_null = qdata_null[0]
I_null = qdata_null[1] 
sigq_null = qdata_null[2]

q_one = qdata_one[0]
I_one = qdata_one[1] 
sigq_one = qdata_one[2]

## Variable presets
side = 3
D = 100 #max density
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

#Q - space
qx = np.fft.fftfreq(x_.size)*n*df*2*np.pi #converting to frequency space
qz = np.fft.rfftfreq(x_.size)*n*df*2*np.pi
# qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
qx, qy, qz = np.meshgrid(qx,qx,qz,indexing='ij') 
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
#allow for any range of q data
qdata = qbinsc[np.where((qbinsc>=q.min()) & (qbinsc<=q.max()) )] #cuts qdata so that it's datapoints only fall within qmax and qmin. 
Idata = np.interp(qdata,q,I)
Idata_null = np.interp(qdata,q_null,I_null)
Idata_one = np.interp(qdata,q_one,I_one)

#create list of qbin indices just in region of data for later F scaling
qbin_args = np.in1d(qbinsc,qdata,assume_unique=True)
qba = qbin_args #just for brevity when using it later

sigqdata = np.interp(qbinsc,q,sigq)
sigqdata_null = np.interp(qbinsc,q_null,sigq_null)
sigqdata_one = np.interp(qbinsc, q_one, sigq_one)

Iq = np.column_stack((q, I, sigq)).T
# Iq = np.vstack((self.q,self.I,self.Ierr)).T
ne = Iq.shape[0] #number of electrons

scale_factor =(ne**2 / Idata[0])
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

erode = True
erosion_width = int(20/dx) #this is in pixels
if erosion_width ==0:
    #make minimum of one pixel
    erosion_width = 1

Imean = np.zeros((len(qbins)))
chi = np.zeros((steps+1))
chiV = np.inf
rg = np.zeros((steps+1))
supportV = np.zeros((steps+1))
support = np.ones(x.shape,dtype=bool) #protein specific
prng_0 = np.random.RandomState()
prng_1 = np.random.RandomState()
seed_0 = prng_0.randint(2**31-1)
seed_1 = prng_1.randint(2**31-1)

prng_0 = np.random.RandomState(seed_0)
prng_1 = np.random.RandomState(seed_1)

#Create a new array. Random
rho_null = prng_0.random_sample(size=x.shape) #- 0.5
rho_one = prng_1.random_sample(size=x.shape)
#newrho = np.zeros_like(rho) #for shrinkwraping
sigma_null = shrinkwrap_sigma_start
sigma_one = shrinkwrap_sigma_start
sigma = shrinkwrap_sigma_start

#Create support for null and 0
support_null = np.ones(rho_null.shape,dtype=bool)
support_one = np.ones(rho_one.shape,dtype=bool)

swbyvol = True
swV = V/2.0
Vsphere_Dover2 = 4./3 * np.pi * (D/2.)**3
swVend = Vsphere_Dover2
swV_decay = 0.9
first_time_swdensity = True
threshold_null = shrinkwrap_threshold_fraction
threshold_one = shrinkwrap_threshold_fraction
erosion_width = int(20/dx) #this is in pixels

counter = 0

#setting up centering
gridcenter_null = (np.array(rho_null.shape)-1.)/2.
gridcenter_one = (np.array(rho_one.shape)-1.)/2.

from scipy import ndimage
from scipy.ndimage import center_of_mass
com_null = np.array(ndimage.measurements.center_of_mass(np.abs(rho_null)))
com_one = np.array(ndimage.measurements.center_of_mass(np.abs(rho_one)))

rho_fullcom = (gridcenter_null-com_null)*dx


## Start For Loop
loop = 1000
for i in range(loop):
    ## Run through F transformation
    F_null = saxs.myrfftn(rho_null) #structure factors
    F_null[np.abs(F_null)==0] = 1e-16 #setting anything that's 0 to almost 0
    F_one = saxs.myrfftn(rho_one) #Why is this negative?
    F_one[np.abs(F_one)==0] = 1e-16 #setting anything that's 0 to almost 0

    #add together to create F_full
    F_full = F_null + F_one

    ## Create F profile to compare
    I3D = saxs.abs2(F_full) #calculating intensity (magnitude squared) #Create two of these
    I3D_null = saxs.abs2(F_null)
    I3D_one = saxs.abs2(F_one)

    Imean = saxs.mybinmean(I3D.ravel(), qblravel) #creates profile THIS IS CORRECT
    Imean_null = saxs.mybinmean(I3D_null.ravel(), qblravel)
    Imean_one = saxs.mybinmean(I3D_one.ravel(), qblravel)

    #scale Fs to match data
    #Calculated Factors
    factors_null = saxs.mysqrt(Idata_null/Imean_null)
    factors_one = saxs.mysqrt(Idata_one/Imean_one)
    factors = saxs.mysqrt(Idata/Imean) 

    #do not scale bins outside of desired range
    #so set those factors to 1.0
    factors[~qba] = 1.0
    factors_null[~qba] = 1.0
    factors_one[~qba] = 1.0
    
    F_null *= factors_null[qbin_labels]
    F_one *= factors_one[qbin_labels]

    # F_null *= factors[qbin_labels]
    # F_one *= factors[qbin_labels]

    #compare profiles
    chi[i]= saxs.mysum(((Imean[qba]-Idata[qba])/sigqdata[qba])**2)/Idata[qba].size

    counter += 1
    print(f"Chi: {chi[i]}, Counter: {counter}")

    # Apply real space constraints
    rho_null_new = saxs.myirfftn(F_null).real
    rho_one_new = saxs.myirfftn(F_one).real

    #Error Reduction/solvent flattening
    rho_null_z = np.zeros_like(rho_null)
    rho_null_z *= 0 
    rho_null_z[support_null] = rho_null_new[support_null] #zeroing everything outside of support
    rho_one_z = np.zeros_like(rho_one)
    rho_one_z *= 0 
    rho_one_z[support_one] = rho_one_new[support_one]
    rho_null = rho_null_z
    rho_one = rho_one_z

    #positivity
    rho_null[rho_null<0] = 0.0
    rho_one[rho_one<0] = 0.0

#shrinkwrap
    # if counter > 600:
    #     # swN = int(swV/dV)
    #     # if swbyvol and swV > swVend:
    #     # These aren't doing anythign? 
    #     new_rho_null, support_null, threshold_null = saxs.shrinkwrap_by_volume(rho_null, absv = True, sigma = sigma, N = swN)
    #     new_rho_one, support_one, threshold_one = saxs.shrinkwrap_by_volume(rho_one, absv = True, sigma = sigma, N = swN)
    #     # threshold_null = shrinkwrap_threshold_fraction #This isn't doing anything lol
    #     # threshold_one = shrinkwrap_threshold_fraction
    #     # rho_null = new_rho_null
    #     # rho_one = new_rho_one

    #     struct_null = ndimage.generate_binary_structure(3, 3)
    #     struct_one = ndimage.generate_binary_structure(3,3)
    #     labeled_support_null, num_features_null = ndimage.label(support_null, structure=struct_null)
    #     labeled_support_one, num_features_one = ndimage.label(support_one, structure=struct_one)        
    #     sums_null = np.zeros((num_features_null))
    #     sums_one = np.zeros((num_features_one))
    #     num_features_to_keep_null = np.min([num_features_null,enforce_connectivity_max_features])
    #     num_features_to_keep_one = np.min([num_features_null, enforce_connectivity_max_features])

    #     for f0 in range(num_features_null +1):
    #         sums_null[f0 - 1] = np.sum(rho_null[labeled_support_null == f0])

    #     for f1 in range(num_features_one + 1):
    #         sums_one[f1 - 1] = np.sum(rho_one[labeled_support_one == f1])

    #     big_feature_null = np.argmax(sums_null) + 1
    #     big_feature_one = np.argmax(sums_one) + 1

    #     sums_order_null = np.argsort(sums_null)[::-1]
    #     sums_order_one = np.argsort(sums_one)[::-1]

    #     sums_sorted_null = sums_null[sums_order_null]
    #     sums_sorted_one = sums_one[sums_order_one]

    #     features_sorted_null = sums_order_null+ 1
    #     features_sorted_one = sums_order_one + 1

    #     support_null *= False
    #     support_one *= False

    #     for f0 in range(num_features_to_keep_null):
    #         support_null[labeled_support_null == features_sorted_null[f0]] = True
    #     for f1 in range(num_features_to_keep_one):
    #         support_one[labeled_support_one == features_sorted_one[f1]] = True

    #     rho_null[~support_null] = 0
    #     rho_one[~support_one] = 0

    #Shrinkwrap OLD
    if counter > 600 and counter % 20 == 0:
        absv = True
        # sigma_null = shrinkwrap_sigma_decay*sigma_null
        # sigma_one = shrinkwrap_sigma_decay*sigma_one #These will be the same
        rho_null_new, support_null = saxs.shrinkwrap_by_density_value(rho_null,absv=absv,sigma=sigma,threshold=threshold_null)
        rho_one_new, support_one = saxs.shrinkwrap_by_density_value(rho_one,absv=absv,sigma=sigma,threshold=threshold_one)
        rho_null = rho_null_new
        rho_one = rho_one_new

        
    #Recentering
    if counter > 500 and counter % 50 == 0: 
        rho_full = rho_null + rho_one
        rho_fullcom = np.unravel_index(rho_full.argmax(), rho_full.shape)   
        full_gridcenter = (np.array(rho_full.shape)-1.)/2.
        full_shift = full_gridcenter-rho_fullcom
        full_shift = np.rint(full_shift).astype(int)
        rho_null = np.roll(np.roll(np.roll(rho_null, full_shift[0], axis=0), full_shift[1], axis=1), full_shift[2], axis=2)
        rho_one = np.roll(np.roll(np.roll(rho_one, full_shift[0], axis=0), full_shift[1], axis=1), full_shift[2], axis=2)
        support_null = np.roll(np.roll(np.roll(support_null, full_shift[0], axis=0), full_shift[1], axis=1), full_shift[2], axis=2)
        support_one = np.roll(np.roll(np.roll(support_one, full_shift[0], axis=0), full_shift[1], axis=1), full_shift[2], axis=2)

 #Write output files
qraw = q
Iraw = I
sigqraw = sigq
Iq_exp = np.vstack((qraw,Iraw,sigqraw)).T
Iq_calc = np.vstack((qbinsc, Imean, Imean*0.01)).T
idx = np.where(Iraw>0)
Iq_exp = Iq_exp[idx]
qmax = np.min([Iq_exp[:,0].max(),Iq_calc[:,0].max()])
Iq_exp = Iq_exp[Iq_exp[:,0]<=qmax]
Iq_calc = Iq_calc[Iq_calc[:,0]<=qmax]
final_chi2, exp_scale_factor, offset, fit = saxs.calc_chi2(Iq_exp, Iq_calc, scale=True, offset=False, interpolation=True,return_sf=True,return_fit=True)


fprefix = "Maija_loop_null"
mprefix = "Maija_loop_one"
sprefix = "Loop"
saxs.write_mrc(rho_null, side, fprefix+".mrc")
saxs.write_mrc(rho_one, side, mprefix+".mrc")
#np.savetxt(sprefix+'_map.fit', fit, delimiter=' ', fmt='%.5e'.encode('ascii'), header='q(data),I(data),error(data),I(density); chi2=%.3f'%final_chi2)