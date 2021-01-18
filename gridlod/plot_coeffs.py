import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from gridlod import build_coefficient, util
from gridlod.world import PatchPeriodic, World

np.random.seed(123)

#random checkerboard
Nepsilon = np.array([128,128])
NFine = np.array([256,256])
alpha=0.1
beta=1.
p=0.1
aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

apertGrid = aPert.reshape(NFine, order='C')
im1 = ax1.imshow(apertGrid, origin='lower_left', extent=(0, 1, 0, 1), cmap='Greys')
fig.colorbar(im1, ax=ax1)
plt.show()

#random defects -- intro picture
Nepsilon = np.array([64,64])
NFine = np.array([256,256])
alpha=0.1
beta=1.
p=0.1
incl_bl = np.array([0.25, 0.25])
incl_tr = np.array([0.75, 0.75])
aPert = build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

apertGrid = aPert.reshape(NFine, order='C')
im1 = ax1.imshow(apertGrid, origin='lower_left', extent=(0, 1, 0, 1), cmap='Greys')
fig.colorbar(im1, ax=ax1)
plt.show()


#random defects -- different possibilities in num experiments
Nepsilon = np.array([16,16])
NFine = np.array([256,256])
alpha=1.
beta=10.
p=0.1
incl_bl = np.array([0.25, 0.25])
incl_tr = np.array([0.75, 0.75])
Lshape_bl = np.array([0.5, 0.5])
Lshape_tr = np.array([0.75, 0.75])
shift_bl = np.array([0.75, 0.75])
shift_tr=np.array([1., 1.])
model1={'name': 'inclfill'}
model2={'name':'inclshift', 'def_bl': shift_bl, 'def_tr': shift_tr}
model3={'name':'inclLshape', 'def_bl': Lshape_bl, 'def_tr': Lshape_tr}

aPertList = []
aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p))
aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, 0.5))
aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, 5.))
aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model1))
aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model2))
aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model3))
fig = plt.figure()

for ii in range(6):
    ax = fig.add_subplot(2, 3, ii+1)
    bounds = np.array([0.2, 0.8, 1.5, 5.5, 10.5])
    mycmap = plt.cm.get_cmap('Greys')
    norm = matplotlib.colors.BoundaryNorm(bounds, mycmap.N)
    apertGrid = aPertList[ii].reshape(NFine, order='C')
    im = ax.imshow(apertGrid, origin='lower_left', extent=(0, 1, 0, 1), cmap=mycmap, norm=norm)
    #fig.colorbar(im, ax=ax)

plt.show()

#two examples of the offline coeffs
Nepsilon = np.array([64,64])
NFine = np.array([256,256])
NCoarse=np.array([32, 32])
k=2
alpha = 0.1
beta = 1.
incl_bl = np.array([0.25, 0.25])
incl_tr = np.array([0.75, 0.75])
NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, None)
middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
patch = PatchPeriodic(world, k, middle)
xpFine = util.pCoordinates(world.NWorldCoarse, patch.iPatchWorldCoarse, patch.NPatchCoarse)

aRefList_rand = build_coefficient.build_checkerboardbasis(patch.NPatchCoarse, Nepsilon // NCoarse,
                                                             world.NCoarseElement, alpha, beta)

fig = plt.figure()
for ii in range(4):
    ax = fig.add_subplot(2, 2, ii+1)
    apertGrid = aRefList_rand[-1+ii].reshape(patch.NPatchFine, order='C')
    im = ax.imshow(apertGrid, origin='lower_left',
                    extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap='Greys')
    #fig.colorbar(im, ax=ax)
plt.show()

aRefList_incl = build_coefficient.build_inclusionbasis_2d(patch.NPatchCoarse, Nepsilon // NCoarse, world.NCoarseElement,
                                                         alpha, beta, incl_bl, incl_tr)

fig = plt.figure()
for ii in range(4):
    ax = fig.add_subplot(2, 2, ii+1)
    apertGrid = aRefList_incl[-1+ii].reshape(patch.NPatchFine, order='C')
    im = ax.imshow(apertGrid, origin='lower_left',
                   extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap='Greys')
    #fig.colorbar(im, ax=ax)
plt.show()