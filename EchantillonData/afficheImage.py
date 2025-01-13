

import numpy as np
import matplotlib.pyplot as plt
import cv2

import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap


import SimpleITK 
import skimage.io as io




##### Une mode de CMP############
top = cm.get_cmap('viridis', 64)
bottom = cm.get_cmap('plasma', 960)
newcolors = np.vstack((top(np.linspace(0, 1, 64)),
                       bottom(np.linspace(1, 0, 960))))
newcmp = ListedColormap(newcolors, name='MonteCarlo')
##### ##### ##### ##### ##### ##### ##### 


LS = io.imread('./EchantillonData/thorax/sample_0001/low_edep.mhd', plugin='simpleitk') #bad


HS = io.imread('./EchantillonData/thorax/sample_0001/high_edep.mhd', plugin='simpleitk') #good


CT = io.imread('./EchantillonData/thorax/sample_0001/ct.mhd', plugin='simpleitk') #ct


MaskCT = io.imread('./EchantillonData/thorax/sample_0001/mask_ct.mhd', plugin='simpleitk') #ct

fig1 = plt.figure(constrained_layout=True, figsize=(12, 3))
spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig1)

######
ax1 = fig1.add_subplot(spec[0])
plt.imshow(CT, interpolation=None, cmap='gray')
plt.ylabel('une coupe de thorax',fontweight='bold')
plt.xlabel('CT',fontweight='bold')
plt.xticks([], [])
plt.yticks([], [])

ax1b = fig1.add_subplot(spec[1])
plt.imshow(LS, interpolation=None, cmap=newcmp)
plt.xlabel('Low Sampling',fontweight='bold')
plt.xticks([], [])
plt.yticks([], [])

ax2 = fig1.add_subplot(spec[2])
plt.imshow(HS, interpolation=None, cmap=newcmp)
plt.xlabel('High Sampling',fontweight='bold')
# plt.colorbar(aspect=50)
plt.xticks([], [])
plt.yticks([], [])

ax2 = fig1.add_subplot(spec[3])
plt.imshow(MaskCT, interpolation=None, cmap='gray')
plt.xlabel('MaskCT',fontweight='bold')
# plt.colorbar(aspect=50)
plt.xticks([], [])
plt.yticks([], [])


###########
plt.show()



