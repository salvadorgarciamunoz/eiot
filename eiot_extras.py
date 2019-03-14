import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy.special import factorial

def humid_2_dry(Ck,RH,gab_coeffs):
    W      = GAB(RH,gab_coeffs)
    W      = W/100
    Ck_dry = Ck/(1+W)
    Ck_h20 = 1-np.sum(Ck_dry,1,keepdims=1)
    Ck_dry = np.hstack((Ck_dry,Ck_h20))
    Ck_out = {'Ck_humid':Ck, 'Ck_dry':Ck_dry}
    return Ck_out

        
def write_eiot_matlab(filename,eiot_obj):
    spio.savemat(filename,eiot_obj,appendmat=True)
    
def read_eiot_matlab(filename):
    eiot_obj=spio.loadmat(filename)
    del eiot_obj['__header__']
    del eiot_obj['__version__']
    del eiot_obj['__globals__']    
    eiot_obj['num_e_sI']=int(eiot_obj['num_e_sI'])
    eiot_obj['num_sI']=int(eiot_obj['num_sI'])
    eiot_obj['abs_max_exc_ri']=eiot_obj['abs_max_exc_ri'][0]
    
    #Convert Numpy objects to lists and dict for PYOMO
    pyo_L = np.arange(1,eiot_obj['S_hat'].shape[1]+1)  #index for wavenumbers
    pyo_N = np.arange(1,eiot_obj['S_hat'].shape[0]+1)  #index for chemical species
    pyo_L = pyo_L.tolist()
    pyo_N = pyo_N.tolist()
    eiot_obj['index_rk_eq'] = eiot_obj['index_rk_eq'].tolist()
    eiot_obj['index_rk_eq'] = eiot_obj['index_rk_eq'][0]
    
    if eiot_obj['num_e_sI']!=0:
        pyo_Me = np.arange(eiot_obj['num_sI']-eiot_obj['num_e_sI']+1,eiot_obj['num_sI']+1)
        pyo_Me = pyo_Me.tolist()
    pyo_S_hat  = np2D2pyomo(eiot_obj['S_hat']) #convert numpy to dictionary
    if not(isinstance(eiot_obj['S_I'],float)):
        pyo_S_I = np2D2pyomo(eiot_obj['S_I'])   #convert numpy to dictionary
        pyo_M   = np.arange(1,eiot_obj['S_I'].shape[0]+1)    #index for non-chemical interferences
        pyo_M   = pyo_M.tolist()
    else:
        pyo_S_I = np.nan
        pyo_M   = [0]
    eiot_obj['pyo_L']     = pyo_L
    eiot_obj['pyo_N']     = pyo_N
    eiot_obj['pyo_M']     = pyo_M
    eiot_obj['pyo_S_hat'] = pyo_S_hat
    eiot_obj['pyo_S_I']   = pyo_S_I   
    if eiot_obj['num_e_sI']!=0:
        aux = eiot_obj['num_sI']-eiot_obj['num_e_sI'] 
        aux_dict=dict(((j+aux+1), eiot_obj['abs_max_exc_ri'][j]) for j in range(len(eiot_obj['abs_max_exc_ri'])))
        eiot_obj['pyo_Me']             = pyo_Me
        eiot_obj['pyo_abs_max_exc_ri'] = aux_dict
    else:
        eiot_obj['abs_max_exc_ri']=np.nan
    return eiot_obj
    
def np2D2pyomo(arr):
    output=dict(((i+1,j+1), arr[i][j]) for i in range(arr.shape[0]) for j in range(arr.shape[1]))
    return output

def np1D2pyomo(arr,*,indexes=False):
    if arr.ndim==2:
        arr=arr[0]
    if isinstance(indexes,bool):
        output=dict(((j+1), arr[j]) for j in range(len(arr)))
    elif isinstance(indexes,list):
        output=dict((indexes[j], arr[j]) for j in range(len(arr)))
    return output


def eiot_summary_plot(eiot_obj,filename,saveplot_flag):
    if not(isinstance(eiot_obj['S_I'],float)):
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(eiot_obj['S_hat'].T)
        ax1.set(title='Apparent spectra of pure species', ylabel=r'$\hat S$')
        ax2.plot(eiot_obj['S_I'].T)
        ax2.set(title='Non-chemical interferences', ylabel=r'$S_I$')
        ax3.plot(eiot_obj['r_I'])
        ax3.set(title='Strength of non-chem int.',ylabel=r'$r_I$')
    else:
        fig, (ax1) = plt.subplots(1)
        ax1.plot(eiot_obj['S_hat'].T)
        ax1.set(title='Apparent spectra of pure species', ylabel=r'$\hat S$')

    plt.subplots_adjust(hspace=.75)
    if saveplot_flag==1: 
        plt.savefig(filename+'.png',dpi=1000)
    plt.show()
    
    
    fig
    first_time=1
    for i in np.arange(1,eiot_obj['S_hat'].shape[0]+1):
        ax = plt.subplot(np.ceil(eiot_obj['S_hat'].shape[0]/2),2,i)
        s_hat    = eiot_obj['S_hat'][i-1,:]
        s_hat_ci = eiot_obj['S_E_CONF_INT'][i-1,:]
        ax.plot(s_hat,linewidth=2.0)
        ax.plot(s_hat+s_hat_ci,':r',linewidth=.5)
        ax.plot(s_hat-s_hat_ci,':r',linewidth=.5)
        ax.set(ylabel='$\hat S_'+str(i)+'$')
        if first_time==1:
            ax.set(title='Apparent Pure Spectra with C.I.')
            first_time=0 
    plt.subplots_adjust(hspace=.5,wspace=.5)
    if saveplot_flag==1: 
        plt.savefig(filename+'_wCI.png',dpi=1000)
    plt.show()
    return   
        
def snv (x):
    if x.ndim ==2:
        mean_x = np.mean(x,axis=1,keepdims=1)     
        mean_x = np.tile(mean_x,(1,x.shape[1]))
        x      = x - mean_x
        std_x  = np.sum(x**2,axis=1)/(x.shape[1]-1)
        std_x  = np.sqrt(std_x)
        std_x  = np.reshape(std_x,(len(std_x),1))
        std_x =  np.tile(std_x,(1,x.shape[1]))
        x      = x/std_x
        return x
    else:
        x = x - np.mean(x)
        stdx = np.sqrt(np.sum(x**2)/(len(x)-1))
        x = x/stdx
        return x
    
def savgol(ws,od,op,Dm):
    if Dm.ndim==1: 
        l = Dm.shape[0]
    else:
        l = Dm.shape[1]
        
    x_vec=np.arange(-ws,ws+1)
    x_vec=np.reshape(x_vec,(len(x_vec),1))
    X = np.ones((2*ws+1,1))
    for oo in np.arange(1,op+1):
        X=np.hstack((X,x_vec**oo))
    XtXiXt=np.linalg.inv(X.T @ X) @ X.T
    coeffs=XtXiXt[od,:] * factorial(od)
    coeffs=np.reshape(coeffs,(1,len(coeffs)))
    for i in np.arange(1,l-2*ws+1):
        if i==1:
            M=np.hstack((coeffs,np.zeros((1,l-2*ws-1))))
        elif i < l-2*ws:
            m_= np.hstack((np.zeros((1,i-1)), coeffs))
            m_= np.hstack((m_,np.zeros((1,l-2*ws-1-i+1))))
            M = np.vstack((M,m_))
        else:
            m_=np.hstack((np.zeros((1,l-2*ws-1)),coeffs))
            M = np.vstack((M,m_))
    if Dm.ndim==1: 
        Dm_sg= M @ Dm
    else:
        for i in np.arange(1,Dm.shape[0]+1):
            dm_ = M @ Dm[i-1,:]
            if i==1:
                Dm_sg=dm_
            else:
                Dm_sg=np.vstack((Dm_sg,dm_))
    return Dm_sg,M

def is_list_integer (mylist):
    try:
        for j in list(range(0,len(mylist))):
            if not(isinstance(mylist[j],int)):
                return False
            return True
    except:
            return False

def dry_basis(Ck,RH,gab_coeffs):
    W      = GAB(RH,gab_coeffs)
    W      = W/100
    Ck_dry = Ck/(1+W)
    Ck_h20 = 1-np.sum(Ck_dry,1,keepdims=1)
    Ck_dry = np.hstack((Ck_dry,Ck_h20))
    Ck_out = {'humid':Ck, 'dry':Ck_dry}
    return Ck_out

        
        
def GAB(RH,gab_coeffs):
#
# RH = [0 100]
# W  = [0 100] 
#
#  Will return the prediction of the percent increase of weight from bone-dry 
#
#  RH can be a vector 
#  gab_coeffs is given by function "getGAB" as output.values
# 
    RH=np.array(RH)
    RH=RH/100
    for r in list(range(0,RH.size)):  
        if gab_coeffs.ndim>1:
            RHaux = np.tile(RH[r],(gab_coeffs.shape[0],1))
            Wm = gab_coeffs[:,0:1]
            C  = gab_coeffs[:,1:2]
            K  = gab_coeffs[:,2:3]
        else:
            if RH.ndim==0:
                RHaux = RH
            else: 
                RHaux = RH[r]
            Wm = gab_coeffs[0]
            C  = gab_coeffs[1]
            K  = gab_coeffs[2]
        Waux = Wm*C*K*RHaux/((1-K*RHaux)*(1-K*RHaux+C*K*RHaux))
        if r==0:
            W=Waux.T
        else:
            W=np.vstack((W,Waux.T))
    if W.ndim==0:
        if W<0:
            W=0
    else:
        W[W<0]=0
    return W


def getGAB(input):
# Simple function to get coefficients for GAB equation for excipients
# To get list of materials with index:
#  getGAB('LIST')
#
# To get coefficients, call function in indexes: for example
#
#  getGAB([1, 3, 5])  will return the GAB coefficients for materials #1, #3 and #5
#
#   output.values are the coefficients
#   output.labels are the materials chosen
# 
# GAB Equation is:
# W = Wm*C*K*RH /( (1-K*RH)*(1-K*RH+ C*K*RH) )
# 
#  Where W is in % [0 1] and RH is in percent [0 100] {Not my preference but taken from the paper}
#
# Data taken from JOURNAL OF PHARMACEUTICAL SCIENCES, VOL. 99, NO. 11, NOVEMBER 2010 
# Table 4. GAB Parameters for Common Pharmaceutical ExcipientsExcipient WmCK
#
   GABS=[ 
     ['0 Non-hygroscopic API'                   ,0          ,1          ,1    ],      
     ['1 Calcium carbonate'                     ,0.149      ,33.933     ,0.803],
     ['2 Croscarmellose sodium'                 ,10.206     ,3.960      ,0.822],
     ['3 Crospovidone'                          ,15.688     ,2.630      ,0.844],
     ['4 Crosslinked polyacrylic acid'          ,9.864      ,1.011      ,0.892],
     ['5 Dibasic calcium phosphate anhydrous'   ,0.173      ,37.490     ,0.671],
     ['6 Hydroxypropyl cellulose'               ,4.507      ,1.526      ,0.907],
     ['7 Hydroxypropylmethyl cellulose'         ,6.372      ,2.295      ,0.821],
     ['8 Lactose monohydrate'                   ,0.039      ,7.582      ,0.967],
     ['9 Lactose regular'                       ,0.647      ,0.023      ,0.770],
     ['10 Lactose, spray dried'                  ,8.658      ,4.148      ,0.026],
     ['11 Magnesium stearate'                    ,1.336      ,1500.488   ,0.004],
     ['12 Magnesium carbonate'                   ,0.665      ,11.177     ,0.703],
     ['13 Mannitol'                              ,0.848      ,0.101      ,0.701],
     ['14 Microcrystalline cellulose'            ,3.979      ,12.524     ,0.770],
     ['15 OpadryTM85F'                           ,1.013      ,2.592      ,0.991],
     ['16 OpadryTM85G184090'                     ,3.114      ,1.247      ,0.837],
     ['17 OpadryTMclear'                         ,2.295      ,2.918      ,0.956],
     ['18 OpadryIITMwhite OY-LS-28914'           ,1.013      ,2.592      ,0.991],
     ['19 Polyethylene oxide'                    ,0.389      ,1.028      ,1.089],
     ['20 Polyvinylpyrrolidone'                  ,0.721      ,1.962      ,17.471],
     ['21 Silicon dioxide'                       ,1.039      ,7.748      ,0.606],
     ['22 Sodium starch glycolate'               ,6.100      ,5.330      ,0.991],
     ['23 Sorbitol'                              ,357.379    ,0.087      ,0.372],
     ['24 Starch1500'                            ,7.400      ,15.930     ,0.736],
     ['25 Talc'                                  ,0.846      ,8.733      ,0.144],
     ['26 Talc [lo micron]'                      ,0.142      ,8.964      ,0.854],
     ['27 Titanium dioxide'                      ,0.202      ,10.544     ,0.771],
     ['28 Lilly MCCPH102'                        ,13.72028895, 0.352618729 , 0.613290498],
     ['29 Lilly SSF'                             ,76.03501406, 0.000452186 , 0.714186336]]
     
   if  input=='LIST':
       return [row[0] for row in GABS]
   else:
       if is_list_integer(input) or isinstance(input,(int)) :
           if is_list_integer(input):
               aux_=[GABS[i] for i in input]
               output=np.array([[row[1] for row in aux_ ],[row[2] for row in aux_ ],[row[3] for row in aux_ ]])
               return output.T
           else:
               aux_=GABS[input]
               output=np.array([aux_[1], aux_[2], aux_[3] ])
               return output
       else:
           return 0
   
    