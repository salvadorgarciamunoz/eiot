    
"""
Example script for the use of pyEIOT to build a PLS based EIOT Supervised model
using data from:
    
    Dyrby, M., Engelsen, S.B., Nørgaard, L., Bruhn, M. and Lundsberg-Nielsen, L., 2002. 
    Chemometric quantitation of the active substance (containing C≡ N) in a pharmaceutical 
    tablet using near-infrared (NIR) transmittance and NIR FT-Raman spectra. 
    Applied Spectroscopy, 56(5), pp.579-585.
    
    Raw data available from: http://www.models.life.ku.dk/Tablets
    
    Needs eiot and eiot_packages both available at:
    https://github.com/salvadorgarciamunoz/eiot
    
    Needs PyPhi to make PLS models, package is available at:
        https://github.com/salvadorgarciamunoz/pyphi
    
"""

import scipy.io as spio
import numpy as np
import eiot
import eiot_extras as ee
import matplotlib.pyplot as plt
import pyphi as phi
import pyphi_plots as pp



#LOAD THE DATA FROM A MATLAB FILE
NIRData      = spio.loadmat('NIR_EIOT_Dyrby_et_al.mat')
nir_spectra  = np.array(NIRData['spectra'])
Ck           = np.array(NIRData['Ck'])
dose_source  = np.array(NIRData['dose_source'])
wavenumbers  = np.array(NIRData['wavenumber'])

# PRE-PROCESS SPECTRA 
#nir_spectra_2_use = ee.snv(nir_spectra)
nir_spectra_2_use,M = ee.savgol(5,1,2,nir_spectra)


# Divide the set into Calibration and Validation taking one in every two samples
nir_spectra_2_use_cal = nir_spectra_2_use[::2,:]
nir_spectra_2_use_val = nir_spectra_2_use[1:nir_spectra_2_use.shape[0]:2,:]
Ck_cal                = Ck[::2,:]
Ck_val                = Ck[1:Ck.shape[0]:2,:]
dose_source_cal       = dose_source[::2,:]
dose_source_val       = dose_source[1:dose_source.shape[0]:2,:]


# Build a PLS from [Ck_cal|dose_source_cal] --> Spectra to determine 
# number of components 
pls_obj=phi.pls(np.hstack((Ck_cal,dose_source_cal)),nir_spectra_2_use_cal,6)
pp.r2pv(pls_obj)

#Chose 3LV's moving forward. Now to determine the # of NCI
eiotpls_obj= eiot.buildwPLS(nir_spectra_2_use_cal,Ck_cal,3,num_si_u=0,rk=dose_source_cal)

#Plot the Lambdas using MATPLOTLIB
fig,ax=plt.subplots()
ax.plot(list(range(1,11)),eiotpls_obj['lambdas'][0:10],'ob')
ax.set_title('Lambda plot for Unsupervised EIOT w PLS')
ax.set_ylabel('Eigenvalues of $\epsilon_{ch}$')
plt.show()
print('Lambdas :' + str(eiotpls_obj['lambdas'][0:7]))



#PLOT RMSE vs # of NCI
rmse_vs_nci_ps = []
rmse_vs_nci_as = []
for nci in list(range(0,4)):
    eiot_obj_pls  = eiot.buildwPLS(nir_spectra_2_use_cal,Ck_cal,3,num_si_u=nci,rk=dose_source_cal)
    pred_sup_ps   = eiot.calc_pls(nir_spectra_2_use_val,eiot_obj_pls)
    pred_sup_as   = eiot.calc_pls(nir_spectra_2_use_val,eiot_obj_pls,rk=dose_source_val)
    rmse_ps       = np.sqrt(np.mean((Ck_val[:,0] - pred_sup_ps['r_hat'][:,0])**2))
    rmse_as       = np.sqrt(np.mean((Ck_val[:,0] - pred_sup_as['r_hat'][:,0])**2))
    rmse_vs_nci_ps.append(rmse_ps)
    rmse_vs_nci_as.append(rmse_as)

fig,ax=plt.subplots()
ax.plot(list(range(0,4)),rmse_vs_nci_ps,'ob',label='Passive Supervision')
ax.plot(list(range(0,4)),rmse_vs_nci_as,'or',label='Active Supervision')
ax.set_title('RMSE vs # of NCI Supervised EIOT w PLS')
ax.set_ylabel('RMSE')
ax.set_xlabel('# NCI')
ax.legend()
plt.show()

fig,ax=plt.subplots()
ax.plot(list(range(0,4)),rmse_vs_nci_ps,'ob',label='Passive Supervision')
ax.set_title('RMSE vs # of NCI Supervised EIOT w PLS')
ax.set_ylabel('RMSE')
ax.set_xlabel('# NCI')
ax.legend()
plt.show()

fig,ax=plt.subplots()
ax.plot(list(range(0,4)),rmse_vs_nci_as,'or',label='Active Supervision')
ax.set_title('RMSE vs # of NCI Supervised EIOT w PLS')
ax.set_ylabel('RMSE')
ax.set_xlabel('# NCI')
ax.legend()
plt.show()


# Build  Supervised EIOT Model with 1 NCI 
eiot_obj_pls  = eiot.buildwPLS(nir_spectra_2_use_cal,Ck_cal,3,num_si_u=1,rk=dose_source_cal)
print("Lambda threshold for Sup. EIOT w PLS = "+ str(eiot_obj_pls['lambdas']))


#Predict validation data w/ Supervised EIOT and ACTIVE Supervision
print('Making predictions of Validation set using Supervised object w AS')
pred_sup_as   = eiot.calc_pls(nir_spectra_2_use_val,eiot_obj_pls,rk=dose_source_val)

#Plot obs vs Pred.
fig,ax=plt.subplots()
ax.set_title('Pred. vs Obs - EIOT w PLS Supervised w AS')
ax.plot(Ck_val[:,0],pred_sup_as['r_hat'][:,0],'ob')
ax.set_xlabel('Observed r[API]')
ax.set_ylabel('Predicted r[API]')
plt.show()

#Calculate RMSE for API prediction
rmse_sup_as = np.sqrt(np.mean((Ck_val[:,0] - pred_sup_as['r_hat'][:,0])**2))
print('RMSE EIOT w PLS Supervised w AS:' + str (rmse_sup_as) )
TSS       = np.sum(Ck_val[:,0]**2)
RSS_supAS = np.sum((Ck_val[:,0]  - pred_sup_as['r_hat'][:,0])**2)
R2_supAS= 1 - RSS_supAS/TSS
print('R2Y EIOT w PLS Supervised w AS on Validation Set:' + str (R2_supAS) )


