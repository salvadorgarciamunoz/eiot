    
"""
Example script for the use of pyEIOT using data from:
    
    Dyrby, M., Engelsen, S.B., Nørgaard, L., Bruhn, M. and Lundsberg-Nielsen, L., 2002. 
    Chemometric quantitation of the active substance (containing C≡ N) in a pharmaceutical 
    tablet using near-infrared (NIR) transmittance and NIR FT-Raman spectra. 
    Applied Spectroscopy, 56(5), pp.579-585.
    
    Raw data available from: http://www.models.life.ku.dk/Tablets
    
Needs eiot and eiot_packages both available at:
    https://github.com/salvadorgarciamunoz/eiot
    
"""

import scipy.io as spio
import numpy as np
import eiot
import eiot_extras as ee
import matplotlib.pyplot as plt


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


# Build  Unsupervised EIOT Model and plot lambdas
eiot_obj = eiot.build(nir_spectra_2_use_cal,Ck_cal)

#Plot the Lambdas using MATPLOTLIB
fig,ax=plt.subplots()
ax.plot(list(range(1,11)),eiot_obj['lambdas'][0:10],'ob')
ax.set_title('Lambda plot for Unsupervised EIOT')
ax.set_ylabel('Eigenvalues of $\epsilon_{ch}$')
plt.show()
print('Lambdas :' + str(eiot_obj['lambdas'][0:7]))


#Buid EIOT objects with increreasing number of NCI and calculate RMSE
rmse_vs_nci = []
for nci in [0,1,2,3,4,5,6,7]:
    eiot_obj    = eiot.build(nir_spectra_2_use_cal,Ck_cal,num_si_u=nci)
    pred_unsup  = eiot.calc(nir_spectra_2_use_val,eiot_obj)
    rmse_unsup  = np.sqrt(np.mean((Ck_val[:,0] - pred_unsup['r_hat'][:,0])**2))
    rmse_vs_nci.append(rmse_unsup)

#PLOT RMSE vs # of NCI
fig,ax=plt.subplots()
ax.plot([0,1,2,3,4,5,6,7],rmse_vs_nci,'ob')
ax.set_title('RMSE vs # of NCI Unsupervised EIOT')
ax.set_ylabel('RMSE')
ax.set_xlabel('# NCI')
plt.show()

# Build  Supervised EIOT Model and plot lambdas
eiot_obj_S = eiot.build(nir_spectra_2_use_cal,Ck_cal,rk=dose_source_cal)
fig,ax=plt.subplots()
ax.set_title('Lambda plot for Supervised EIOT')
ax.plot(list(range(1,11)),eiot_obj_S['lambdas'][0:10],'ob')
ax.set_ylabel('Eigenvalues of $\epsilon_{ch}$')
plt.show()
print('Lambdas :' + str(eiot_obj_S['lambdas'][0:7]))

#PLOT RMSE vs # of NCI
rmse_vs_nci_ps = []
rmse_vs_nci_as = []
for nci in list(range(0,7)):
    eiot_obj_S  = eiot.build(nir_spectra_2_use_cal,Ck_cal,rk=dose_source_cal,num_si_u=nci)
    pred_sup_ps = eiot.calc(nir_spectra_2_use_val,eiot_obj_S)
    pred_sup_as = eiot.calc(nir_spectra_2_use_val,eiot_obj_S,rk=dose_source_val)
    rmse_ps     = np.sqrt(np.mean((Ck_val[:,0] - pred_sup_ps['r_hat'][:,0])**2))
    rmse_as     = np.sqrt(np.mean((Ck_val[:,0] - pred_sup_as['r_hat'][:,0])**2))
    rmse_vs_nci_ps.append(rmse_ps)
    rmse_vs_nci_as.append(rmse_as)

fig,ax=plt.subplots()
ax.plot(list(range(0,7)),rmse_vs_nci_ps,'ob',label='Passive Supervision')
ax.plot(list(range(0,7)),rmse_vs_nci_as,'or',label='Active Supervision')
ax.set_title('RMSE vs # of NCI Supervised EIOT')
ax.set_ylabel('RMSE')
ax.set_xlabel('# NCI')
ax.legend()
plt.show()




# Build  Unsupervised EIOT Model with 1 NCI
eiot_obj = eiot.build(nir_spectra_2_use_cal,Ck_cal,num_si_u=1)
print("Lambda threshold for EIOT Unsup = "+ str(eiot_obj['lambdas']))

# Build  Supervised EIOT Model with 1 NCI 
eiot_obj_S = eiot.build(nir_spectra_2_use_cal,Ck_cal,rk=dose_source_cal,num_si_u=1)
print("Lambda threshold for EIOT Sup = "+ str(eiot_obj_S['lambdas']))

#Predict validation data w/ Unsup EIOT
print('Making predictions of Validation set using Unsupervised object')
pred_unsup=eiot.calc(nir_spectra_2_use_val,eiot_obj)

#Plot obs vs Pred.
fig,ax=plt.subplots()
ax.set_title('Pred. vs Obs - EIOT Unsupervised')
ax.plot(Ck_val[:,0],pred_unsup['r_hat'][:,0],'ob')
ax.set_xlabel('Observed r[API]')
ax.set_ylabel('Predicted r[API]')
plt.show()

#Predict validation data w/ Supervised EIOT using Passive Supervision
print('Making predictions of Validation set using Supervised object w PS')
pred_sup_ps=eiot.calc(nir_spectra_2_use_val,eiot_obj_S)

#Plot obs vs Pred.
fig,ax=plt.subplots()
ax.set_title('Pred. vs Obs - EIOT Supervised w PS')
ax.plot(Ck_val[:,0],pred_sup_ps['r_hat'][:,0],'ob')
ax.set_xlabel('Observed r[API]')
ax.set_ylabel('Predicted r[API]')
plt.show()

#Predict validation data w/ Supervised EIOT and ACTIVE Supervision
print('Making predictions of Validation set using Supervised object w AS')
pred_sup_as=eiot.calc(nir_spectra_2_use_val,eiot_obj_S,rk=dose_source_val)

#Plot obs vs Pred.
fig,ax=plt.subplots()
ax.set_title('Pred. vs Obs - EIOT Supervised w AS')
ax.plot(Ck_val[:,0],pred_sup_as['r_hat'][:,0],'ob')
ax.set_xlabel('Observed r[API]')
ax.set_ylabel('Predicted r[API]')
plt.show()

#Calculate RMSE for API prediction
rmse_unsup  = np.sqrt(np.mean((Ck_val[:,0] - pred_unsup['r_hat'][:,0])**2))
rmse_sup_ps = np.sqrt(np.mean((Ck_val[:,0] - pred_sup_ps['r_hat'][:,0])**2))
rmse_sup_as = np.sqrt(np.mean((Ck_val[:,0] - pred_sup_as['r_hat'][:,0])**2))

print('RMSE EIOT Unsupervised   :' + str (rmse_unsup) )
print('RMSE EIOT Supervised w PS:' + str (rmse_sup_ps) )
print('RMSE EIOT Supervised w AS:' + str (rmse_sup_as) )



TSS       = np.sum(Ck_val[:,0]**2)
RSS_unsup = np.sum((Ck_val[:,0]  - pred_unsup['r_hat'][:,0] )**2)
RSS_supPS = np.sum((Ck_val[:,0]  - pred_sup_ps['r_hat'][:,0])**2)
RSS_supAS = np.sum((Ck_val[:,0]  - pred_sup_as['r_hat'][:,0])**2)


R2_unsup= 1 - RSS_unsup/TSS
R2_supPS= 1 - RSS_supPS/TSS
R2_supAS= 1 - RSS_supAS/TSS

print('R2Y EIOT Unsupervised   :' + str (R2_unsup) )
print('R2Y EIOT Supervised w PS:' + str (R2_supPS) )
print('R2Y EIOT Supervised w AS:' + str (R2_supAS) )


