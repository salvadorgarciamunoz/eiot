'''
Sample script using EIOT on data produced by:
    
    Wülfert, F., Kok, W.T. and Smilde, A.K., 1998. Influence of temperature on 
    vibrational spectra and consequences for the predictive ability of 
    multivariate models. Analytical chemistry, 70(9), pp.1761-1767.

Raw data is available from:
    
    http://www.bdagroup.nl/content/Downloads/datasets/datasets.php


 Needs eiot and eiot_packages both available at:
    https://github.com/salvadorgarciamunoz/eiot
    
    Needs PyPhi to make PLS models, package is available at:
        https://github.com/salvadorgarciamunoz/pyphi    

This example uses a much leaner data set for training and illustrates the 
enhanced predictability obtained with an EIOT model    
    
'''

import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import eiot
import eiot_extras as ee
import pyphi as phi

Wulfert_Data      = spio.loadmat('flodat.mat')


### BUILD DATA SETS FOR TRAINING AND TESTING

#set_1 = [1,5,8,11,18,20,22]
#set_2 = [2,3,4,6,7,9,10,12,13,14,15,16,17,19,21]

# Set indexes wrt 0
set_1 = [0,4,7,10,17,19,21]
set_2 = [1,2,3,5,6,8,9,11,12,13,14,15,16,18,20]


# Data available at  30,40,50,60,70  degrees C

#Training set:
#    Temperature        set
#----------------------------
#       30             set_1
#       50             set 1
#       70             set_1

# Test Set
#    Temperature        set
#----------------------------
#       30             set_2
#       40             set_1 & set_2
#       50             set 2
#       60             set_1 & set_2
#       70             set_2

S30_1=Wulfert_Data['spec30'][:,set_1]
S30_1=S30_1.T
S30_2=Wulfert_Data['spec30'][:,set_2]
S30_2=S30_2.T

S40_1=Wulfert_Data['spec40'][:,set_1]
S40_1=S40_1.T
S40_2=Wulfert_Data['spec40'][:,set_2]
S40_2=S40_2.T

S50_1=Wulfert_Data['spec50'][:,set_1]
S50_1=S50_1.T
S50_2=Wulfert_Data['spec50'][:,set_2]
S50_2=S50_2.T

S60_1=Wulfert_Data['spec60'][:,set_1]
S60_1=S60_1.T
S60_2=Wulfert_Data['spec60'][:,set_2]
S60_2=S60_2.T

S70_1=Wulfert_Data['spec70'][:,set_1]
S70_1=S70_1.T
S70_2=Wulfert_Data['spec70'][:,set_2]
S70_2=S70_2.T

conc_1=Wulfert_Data['conc'][set_1,:]
conc_2=Wulfert_Data['conc'][set_2,:]

spectra_cal=np.vstack((S30_1,S50_1,S70_1))
spectra_val=np.vstack((S30_2,S40_1,S40_2,S50_2,S60_1,S60_2,S70_2))

temp_cal = np.vstack((Wulfert_Data['temper30'][set_1],
                      Wulfert_Data['temper50'][set_1],
                      Wulfert_Data['temper70'][set_1]))


temp_val = np.vstack((Wulfert_Data['temper30'][set_2],
                      Wulfert_Data['temper40'][set_1],
                      Wulfert_Data['temper40'][set_2],
                      Wulfert_Data['temper50'][set_2],
                      Wulfert_Data['temper60'][set_1],
                      Wulfert_Data['temper60'][set_2],
                      Wulfert_Data['temper70'][set_2]))

conc_cal = np.vstack((conc_1,conc_1,conc_1))
conc_val = np.vstack((conc_2,conc_1,conc_2,conc_2,conc_1,conc_2,conc_2))


# PRE PROCESS SPECTRA
spectra_cal,M = ee.savgol(5,1,2,spectra_cal)
spectra_val,M = ee.savgol(5,1,2,spectra_val)

wl=Wulfert_Data['wl']
wl=wl[5:-5]



del S30_1,S30_2,S40_1,S40_2,S50_1,S50_2,S60_1,S60_2,S70_1,S70_2,conc_1,conc_2

# PLOT SPECTRA
fig,ax=plt.subplots()
ax.plot(wl,spectra_cal.T)
ax.set(title='Calibration Spectra',xlabel='Wavelength',ylabel='ua')
plt.show()

fig,ax=plt.subplots()
ax.plot(wl,spectra_val.T)
ax.set(title='Validation Spectra',xlabel='Wavelength',ylabel='ua')
plt.show()

# BUILD A MODEL WITH ZERO NCI AND LOOK AT LAMBDA PLOTS
eiot_obj=eiot.build(spectra_cal,conc_cal,R_ik=temp_cal)
fig,ax=plt.subplots()
ax.plot(eiot_obj['lambdas'][0:10],'ob')
ax.set(title='Lambdas for EIOT Sup w Temp NCI=0',xlabel='#NCI',ylabel='lambda')
plt.show()

# CALCULATE RMSE AS A FUNCTION OF INCREASING NUMBER OF NCI
rmse_vs_nci_ps = []
rmse_vs_nci_as = []
for nci in [0,1,2,3,4,5,6,8,9,10]:
    eiot_obj    = eiot.build(spectra_cal,conc_cal,R_ik=temp_cal,num_si_u=nci)
    pred_sup_ps = eiot.calc(spectra_val,eiot_obj)
    pred_sup_as = eiot.calc(spectra_val,eiot_obj,r_ik=temp_val)
    rmse_ps     = np.sqrt(np.mean((conc_val[:,0] - pred_sup_ps['r_hat'][:,0])**2))
    rmse_as     = np.sqrt(np.mean((conc_val[:,0] - pred_sup_as['r_hat'][:,0])**2))
    rmse_vs_nci_ps.append(rmse_ps)
    rmse_vs_nci_as.append(rmse_as)

fig,ax=plt.subplots()
ax.plot(list(range(0,10)),rmse_vs_nci_ps,'ob',label='Passive Supervision')
ax.plot(list(range(0,10)),rmse_vs_nci_as,'or',label='Active Supervision')
ax.set_title('RMSE vs # of NCI Supervised EIOT')
ax.set_ylabel('RMSE')
ax.set_xlabel('# NCI')
ax.legend()
plt.show()    

# BASED ON PREVIOUS PLOT CHOOSE 5 NCI
eiot_obj    = eiot.build(spectra_cal,conc_cal,R_ik=temp_cal,num_si_u=5)
pred_sup_as = eiot.calc(spectra_val,eiot_obj,r_ik=temp_val)
rmse_as     = np.sqrt(np.mean((conc_val - pred_sup_as['r_hat'])**2,axis=0,keepdims=True))

#Plot obs vs Pred.
fig,ax=plt.subplots()
ax.set_title('Pred. vs Obs - EIOT Supervised w AS')
ax.plot(conc_val[:,0],pred_sup_as['r_hat'][:,0],'ob')
ax.set_xlabel('Observed r[A]')
ax.set_ylabel('Predicted r[A]')
plt.show()

#Plot obs vs Pred.
fig,ax=plt.subplots()
ax.set_title('Pred. vs Obs - EIOT Supervised w AS')
ax.plot(conc_val[:,1],pred_sup_as['r_hat'][:,1],'ob')
ax.set_xlabel('Observed r[B]')
ax.set_ylabel('Predicted r[B]')
plt.show()

#Plot obs vs Pred.
fig,ax=plt.subplots()
ax.set_title('Pred. vs Obs - EIOT Supervised w AS')
ax.plot(conc_val[:,2],pred_sup_as['r_hat'][:,2],'ob')
ax.set_xlabel('Observed r[C]')
ax.set_ylabel('Predicted r[C]')
plt.show()

#CALCULATE RMSE
TSS       = np.sum(conc_val**2,axis=0,keepdims=True)
RSS_supAS = np.sum((conc_val  - pred_sup_as['r_hat'])**2,axis=0,keepdims=True)
R2_supAS= 1 - RSS_supAS/TSS
print('RMSE EIOT Supervised w AS:' + str (rmse_as) )
print('R2Y EIOT Supervised w AS:' + str (R2_supAS) )


#Same data but using PLS
rmse_vs_nci = []
for a in list(range(1,15)):
    pls_obj= phi.pls(spectra_cal,conc_cal,a,shush=True)
    pls_pred=phi.pls_pred(spectra_val,pls_obj)
    rmse_pls   = np.sqrt(np.mean((conc_val[:,0] - pls_pred['Yhat'][:,0])**2))
    rmse_vs_nci.append(rmse_pls)
    
fig,ax=plt.subplots()
ax.plot(list(range(1,15)),rmse_vs_nci,'ob')
ax.set_title('RMSE vs # of LVs PLS')
ax.set_ylabel('RMSE')
ax.set_xlabel('# LV''s')
ax.legend()
plt.show()   

# CHOOSE 3LV'S PER RMSE VS #LV'S PLOT
pls_obj= phi.pls(spectra_cal,conc_cal,3,shush=True)
pls_pred=phi.pls_pred(spectra_val,pls_obj)
rmse_pls = np.sqrt(np.mean((conc_val - pls_pred['Yhat'])**2,axis=0,keepdims=True))
RSS_pls = np.sum((conc_val  - pls_pred['Yhat'])**2,axis=0,keepdims=True)
R2_pls= 1 - RSS_pls/TSS
print('RMSE PLS w 11 LV''s:' + str (rmse_pls) )
print('R2Y PLS w 11 LV''s:' + str (R2_pls) )

