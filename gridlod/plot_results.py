import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

root= '../' #adapt path file possibly

#1d
NList1d = [8,16, 32,64]
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

i = -3
for N in NList1d:
    err = sio.loadmat(root+'_meanErr_Nc'+str(N)+'.mat')
    harmerr = err['meanHarmErr'][0]
    pList = err['pList'][0]
    relL2err = err['meanErr_comb'][0]
    labelplain = 'H=2^{'+str(i)+'}'
    ax1.plot(pList,harmerr, '-*', label=r'${}$'.format(labelplain))
    ax2.plot(pList, relL2err, '-*', label=r'${}$'.format(labelplain))
    i -= 1
ax1.legend()
ax2.legend()
ax1.set_xlabel('p')
ax1.set_ylabel('maximal error in elementwise harmonic means')
ax2.set_xlabel('p')
ax2.set_ylabel('relative $L^2$-error')
plt.show()

#2d
# mean error vs probability random checkerboard

err2d = sio.loadmat(root+'_meanErr2d.mat')
errNew = err2d['relerrL2new'][0]
pList2d = err2d['pList'][0]
relL2errpert = err2d['relerrL2pert'][0]
plt.plot(pList2d, errNew, '-*', label='new')
plt.plot(pList2d, relL2errpert, '-*', label='perturbed')
plt.legend()
plt.xlabel('p')
plt.ylabel('relative $L^2$-errors')
plt.show()

# errors vs sample numbers ranomd checkerboard

err2dsamp = sio.loadmat(root+'_relErr.mat')
errNew = err2dsamp['relErrNew'][0]
errpert = err2dsamp['relErrHKM'][0]
samp = err2dsamp['iiSamp'][0]
meanErrNew = [np.mean(errNew[:(j+1)]) for j in range(len(errNew))]
meanErrHKM = [np.mean(errpert[:(j+1)]) for j in range(len(errNew))]
ErrNewUp = [meanErrNew[j]+1.96*np.sqrt(np.var(errNew[:(j+1)])/(j+1)) for j in range(len(errNew))]
ErrHKMUp = [meanErrHKM[j]+1.96*np.sqrt(np.var(errpert[:(j+1)])/(j+1)) for j in range(len(errpert))]
ErrNewDown = [meanErrNew[j]-1.96*np.sqrt(np.var(errNew[:(j+1)])/(j+1)) for j in range(len(errNew))]
ErrHKMDown = [meanErrHKM[j]-1.96*np.sqrt(np.var(errpert[:(j+1)])/(j+1)) for j in range(len(errpert))]
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.plot(samp, meanErrNew, 'b')
ax1.plot(samp, ErrNewDown, '--b')
ax1.plot(samp, ErrNewUp, '--b')
ax2.plot(samp, meanErrHKM, 'b')
ax2.plot(samp, ErrHKMDown, '--b')
ax2.plot(samp, ErrHKMUp, '--b')
ax1.set_xlabel('Number of samples')
ax1.set_ylabel('relative $L^2$-errors')
ax2.set_xlabel('Number of samples')
ax2.set_ylabel('relative $L^2$-errors')
plt.show()

# random defect change in value
err2d_value = sio.loadmat(root+'_meanErr2d_defvalues.mat')
errNew_incl = err2d_value['relerrL2new_incl']
errNew_def = err2d_value['relerrL2new_def']
pList2d = err2d_value['pList'][0]
relL2errpert = err2d_value['relerrL2pert']
values = err2d_value['values'][0]
colors=['b', 'r', 'g']
for ii in range(len(values)):
    if ii !=0:
        plt.plot(pList2d, errNew_def[:,ii], colors[ii]+'-*', label='new (defect inclusions), value='+str(values[ii]))
    plt.plot(pList2d, errNew_incl[:,ii], colors[ii]+'-o', label='new (erasing inclusions), value=' + str(values[ii]))
    plt.plot(pList2d, relL2errpert[:,ii], colors[ii]+'-s', label='perturbed, value=' + str(values[ii]))
plt.legend()
plt.xlabel('p')
plt.ylabel('relative $L^2$-errors')
plt.show()

for ii in range(len(values)):
    plt.plot(pList2d, errNew_def[:,ii], colors[ii]+'-*', label='new (defect inclusions), value='+str(values[ii]))
plt.legend()
plt.xlabel('p')
plt.ylabel('relative $L^2$-errors')
plt.show()

for ii in range(len(values)):
    plt.plot(pList2d, errNew_incl[:,ii], colors[ii]+'-*', label='new (erasing inclusions), value='+str(values[ii]))
plt.legend()
plt.xlabel('p')
plt.ylabel('relative $L^2$-errors')
plt.show()

# random defect change in geometry
err2d_change = sio.loadmat(root+'_meanErr2d_defchanges_old.mat')
errNew_change = err2d_change['relerrL2new']
pList2d = err2d_change['pList'][0]
relL2errpert_change = err2d_change['relerrL2pert']
names = [err2d_change['names'][j][4:] for j in range(len(err2d_change['names']))]
colors=['b', 'r', 'g']
for ii in range(len(names)):
    plt.plot(pList2d, errNew_change[:,ii], colors[ii]+'-*', label='new, model ' + names[ii])
    plt.plot(pList2d, relL2errpert_change[:,ii], colors[ii]+'-s', label='perturbed, model ' + names[ii])
plt.legend()
plt.xlabel('p')
plt.ylabel('relative $L^2$-errors')
plt.show()

for ii in range(len(names)):
    plt.plot(pList2d, errNew_change[:,ii], colors[ii]+'-*', label='new, model '+ names[ii])
plt.legend()
plt.xlabel('p')
plt.ylabel('relative $L^2$-errors')
plt.show()

#indicator
pList = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11]

indicMiddleMean = []
errabsMean = []
errrelMean = []

for p in pList:
    errIndic = sio.loadmat(root + '_ErrIndic2drandcheck_p'+str(p)+'.mat')
    IndicListmiddle = errIndic['ETListmiddle'][0]
    errabsList = errIndic['absError'][0]
    errrelList = errIndic['relError'][0]

    indicMiddleMean.append(np.nanmean(IndicListmiddle))
    errabsMean.append(np.mean(errabsList))
    errrelMean.append(np.mean(errrelList))

plt.figure()
plt.plot(pList, indicMiddleMean, 'b-*', label='$E_T$')
plt.plot(pList, errabsMean, 'r-*', label='absolute $L^2$-error')
plt.plot(pList, errrelMean, 'g-*', label='relative $L^2$-error')
plt.legend()
plt.xlabel('p')
plt.ylabel('mean over 500 samples')

plt.show()