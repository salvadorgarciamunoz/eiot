import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy.special import factorial
import pandas as pd
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource

        
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


def eiot_summary_plot(eiot_obj,*,filename='myplot',saveplot_flag=False):
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
    if saveplot_flag: 
        plt.savefig(filename+'_wCI.png',dpi=1000)
    plt.show()
    
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
    if saveplot_flag: 
        plt.savefig(filename+'.png',dpi=1000)
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
        Dm_sg= Dm @ M.T
    return Dm_sg,M

def is_list_integer (mylist):
    try:
        for j in list(range(0,len(mylist))):
            if not(isinstance(mylist[j],int)):
                return False
            return True
    except:
            return False


   
def predvsobsplot(Y,Yhat,*,variable_names=False,obs_names=False):
    """
    Plot observed vs predicted values
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    Y:    Numpy array or pandas dataframe with observed values
    
    Yhat: Numpy array with predicted data
    
    variable_names: List with names of the variables  (columns of Y) if Y
    is a numpy array [otherwise names are taken from Pandasdataframe]
    
    """


    if isinstance(Y,np.ndarray):
        if isinstance(variable_names,np.bool):
            YVar = []
            for n in list(np.arange(Y.shape[1])+1):
                YVar.append('YVar #'+str(n))
        else:
            YVar=variable_names
            
        if isinstance(obs_names,np.bool):
            ObsID_ = []
            for n in list(np.arange(Y.shape[0])+1):
                ObsID_.append('Obs #'+str(n))
        else:
            ObsID_=obs_names

    elif isinstance(Y,pd.DataFrame):
        Y_=np.array(Y.values[:,1:]).astype(float)
        YVar = Y_.columns.values
        YVar = YVar[1:]
        YVar = YVar.tolist()
        ObsID_ = Y.values[:,0].astype(str)
        ObsID_ = ObsID_.tolist()
        
    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID")
                ]
    
    rnd_num=str(int(np.round(1000*np.random.random_sample())))
    output_file("ObsvsPred_"+rnd_num+".html",title='ObsvsPred')
    plot_counter=0
    
    for i in list(range(Y_.shape[1])):
        x_ = Y[:,i]
        y_ = Yhat[:,i]           
        source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_))
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title=YVar[i])
        p.circle('x', 'y', source=source,size=7,color='darkblue')
        p.line([np.nanmin(x_),np.nanmax(x_)],[np.nanmin(y_),np.nanmax(y_)],line_color='cyan',line_dash='dashed')
        p.xaxis.axis_label ='Observed'
        p.yaxis.axis_label ='Predicted'
        if plot_counter==0:
            p_list=[p]
        else:
            p_list.append(p)
        plot_counter = plot_counter+1            
    show(column(p_list))        
    return    

    