"""
EIOT code for Python v 1.0
by Sal Garcia (sal.garcia@lilly.com)

"""

import eiot_extras as ee
import numpy as np
from pyomo.environ import *
import pyphi as phi

from shutil import which
from pyomo.solvers.plugins.solvers.GAMS import GAMSDirect, GAMSShell
import warnings
# Autoconfig pyomo NLP solver. Use 1) Ipopt Binary (with hsl if available),
# 2) gams:ipopt, or 3) neos server

def ma57_dummy_check():
    """
    Instantiates a trivial NLP to solve with IPOPT and MA57.
    Returns:
          ma57_ok: boolean, True if IPOPT solved with SolverStaus.ok
    """
    m = ConcreteModel()
    m.x = Var()
    m.Obj = Objective(expr = m.x**2 -1)

    s = SolverFactory('ipopt')
    s.options['linear_solver'] = 'ma57'

    import logging
    pyomo_logger = logging.getLogger('pyomo.core')
    LOG_DEFAULT = pyomo_logger.level
    pyomo_logger.setLevel(logging.ERROR)
    r = s.solve(m)
    pyomo_logger.setLevel(LOG_DEFAULT)
    
    ma57_ok = r.solver.status == SolverStatus.ok
    return ma57_ok

hsl_ok = False
if bool(which('ipopt')):
    ipopt_solver = 'ipopt'
    hsl_ok = ma57_dummy_check()
    
    if hsl_ok:
        print("Solving NLPs with IPOPT/MA57 by default.")
    else:
        print("Solving NLPs with IPOPT/MUMMPS by default.")

elif (GAMSDirect().available(exception_flag=False) or GAMSShell().available(exception_flag=False)):
    ipopt_solver = 'gams:ipopt'
    print("Solving NLPs with GAMS/IPOPT by default")
    warnings.warn("GAMS/IPOPT has had issues with constraint qualifications and long solve times. Binary IPOPT strongly recommened.")

else:
    warnings.warn("Too many NLPs to solve with NEOS/IPOPT. Must assign have environment variable \"NEOS_EMAIL\" declared. Binary IPOPT strongly recommended. Optionally, set eiot.ipopt_solver = \"custom\" and define eiot.custom_nlpsolver or redefine eiot.auto_nlpsolver with alternative.")
    ipopt_solver = 'neos' # Way too slow


def auto_nlpsolver(model, tee):
    """
    model: NLP Pyomo Model to solve via our autoconfigured IPOPT Solver

    define function custom_nlpsolve(model, tee) and set the package variable ipopt_solver
    to "custom" if you want to implement your own customized SolverFactory, settings, then
    return the results of solver.solve(model).

    return: results of solve(model)
    """

    if ipopt_solver == "ipopt":
        solver = SolverFactory("ipopt")
        if hsl_ok:
            solver.options['linear_solver']='ma57'
        results = solver.solve(model,tee=tee)
    elif ipopt_solver == "gams:ipopt":
        solver = SolverFactory('gams:ipopt')

        # It doesn't seem to notice the opt file when I write it
        results = solver.solve(model, tee=tee)
    elif ipopt_solver == "custom":
        # Will throw error if custom_nlpsolver is not defined
        results = custom_nlpsolver(model, tee)
    else:
        # Default to neos as backup
        solver_manager = SolverManagerFactory('neos')
        results = solver_manager.solve(model, opt='ipopt', tee=tee)
    return results

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build(Dm,R,*,R_ik=False,num_si_u=0,num_si_e=0):
    """

     EIOT BUILD Routines by Salvador Garcia-Munoz (sgarciam@ic.ac.uk, sal.garcia@lilly.com)

    Parameters
    ----------
    Dm : TYPE: Numpy Array
        DESCRIPTION: Mixture Spectra [o x lambda].
    R : TYPE: Numpy Array
        DESCRIPTION: Known composition per sample (mass/mole/volume fractions) [o x n].
    R_ik : TYPE: Numpy Array OPTIONAL.
        DESCRIPTION: Supervised non-chemical interferences [o x m], Default = None.
    num_si_u : TYPE: Integer OPTIONAL.
        DESCRIPTION: Number of unsupervised non-chemical interferences, Default = 0.
    num_si_e : TYPE: Integer OPTIONAL.
        DESCRIPTION: Number of exclusive non-chemical interferences, Default = 0.

    Returns
    -------
    eiot_object (Dictionary)

    """

    if not(isinstance(R_ik,bool)):
        print('Building supervised EIOT')
        eiot_obj=build_supervised_(Dm,R,Rk=R_ik,num_e_sI=num_si_e,num_sI_U=num_si_u)
    else:
        print('Building Unsupervised EIOT')
        eiot_obj=build_(Dm,R,num_sI=num_si_u)  
    return eiot_obj
        
def build_supervised_(Dm,Ck,Rk,num_e_sI,num_sI_U):
        
    Rk_mean  = np.mean(Rk,axis=0,keepdims=1)
    Rk_      = Rk - np.tile(Rk_mean,(Rk.shape[0],1))
    Rk_std   = np.std(Rk_,axis=0,keepdims=1,ddof=1)
    Rk_      = Rk_ / np.tile(Rk_std,(Rk.shape[0],1))
    Rk_n     = Rk_
# Do not scale binary numbers for exclusive signatures
    if num_e_sI!=0:
        for i in range(Rk.shape[1]-num_e_sI,Rk.shape[1]):
            Rk_n[:,i]  = Rk[:,i]
            Rk_mean[0][i] = 0
            Rk_std[0][i]  = 1    
    Rk      = Rk_n        
    Ck_aug  = np.hstack((Ck,Rk))
    num_sI  = Rk.shape[1]+num_sI_U
    S_hat   = np.linalg.pinv(Ck_aug.T@Ck_aug)@Ck_aug.T@Dm
    S_E_tmp = S_hat
        
    S_I_S   = S_hat[Ck.shape[1]:]    # Supervised non-chemical interferences
    S_hat   = S_hat[0:Ck.shape[1]]   # Apparent pure spectrum for chemical species
        
    flag_force_no_exclusives = 0
        
    if num_e_sI==0:
        S_I_E = np.nan
        r_I_E = np.nan
        r_I_S = Rk
    elif S_I_S.shape[0]==num_e_sI:
        S_I_E = S_I_S
        S_I_S = np.nan
        r_I_S = np.nan
        r_I_E = Rk
    elif S_I_S.shape[0]>num_e_sI:
        S_I_E = S_I_S[-num_e_sI:]
        S_I_S = S_I_S[0:S_I_S.shape[0]-num_e_sI]
        r_I_E = Rk[:,Rk.shape[1]-num_e_sI:]
        r_I_S = Rk[:,0:Rk.shape[1]-num_e_sI]
    else:
        S_I_E = np.nan
        r_I_E = np.nan
        flag_force_no_exclusives = 1  #if the given number of exclusive non-chem int is > size(Rk,2)
    
    E_ch_tmp = Dm- Ck_aug @ S_E_tmp

    if num_sI_U>0:
        [U,S,Vh]   = np.linalg.svd(E_ch_tmp)
        V          = Vh.T
        S_I_U      = V[:,0:num_sI_U]
        S_I_U      = S_I_U.T
        S_short    = S[0:num_sI_U]
        r_I_U      = U[:,0:num_sI_U] * np.tile(S_short,(U.shape[0],1))
        if isinstance(S_I_E,float):
            S_E        = np.vstack((S_hat,S_I_S,S_I_U))
            S_I        = np.vstack((S_I_S,S_I_U))
            r_I        = np.hstack((r_I_S,r_I_U))
            index_rk_eq    = list(range(1,1+S_I_S.shape[0]))
            index_rk_ex_eq = False
        else: 
            if isinstance(S_I_S,float):
                S_E        = np.vstack((S_hat,S_I_U,S_I_E))
                S_I        = np.vstack((S_I_U,S_I_E))
                r_I        = np.hstack((r_I_U,r_I_E))   
                index_rk_eq    = list(range(S_I_U.shape[0]+1,S_I_U.shape[0]+S_I_E.shape[0]+1 ))
                index_rk_ex_eq = index_rk_eq
            else:
                S_E        = np.vstack((S_hat,S_I_S,S_I_U,S_I_E))
                S_I        = np.vstack((S_I_S,S_I_U,S_I_E))
                r_I        = np.hstack((r_I_S,r_I_U,r_I_E))
                index_rk_eq    = list(range(1,1+S_I_S.shape[0])) + list(range(S_I_S.shape[0]+S_I_U.shape[0]+1,S_I_S.shape[0]+S_I_U.shape[0]+S_I_E.shape[0]+1 ))
                index_rk_ex_eq = list(range(S_I_S.shape[0]+S_I_U.shape[0]+1,S_I_S.shape[0]+S_I_U.shape[0]+S_I_E.shape[0]+1 )) 
        lambdas    = S[num_sI]
    else:
        S_E            = S_E_tmp
        if isinstance(S_I_E,float):
            S_I        = S_I_S
            r_I        = r_I_S
            index_rk_ex_eq = False
        elif isinstance(S_I_S,float):
            S_I        = S_I_E
            r_I        = r_I_E
            index_rk_ex_eq = list(range(1,1+S_I_E.shape[0]))
        else:    
            S_I        = np.vstack((S_I_S,S_I_E))
            r_I        = np.hstack((r_I_S,r_I_E))
            index_rk_ex_eq = list(range(1+S_I_S.shape[0],1+S_I.shape[0]))
        index_rk_eq    = list(range(1,1+S_I.shape[0]))
        
        [U,S,Vh] = np.linalg.svd(E_ch_tmp)
        V        = Vh.T
        lambdas    = S
        
    SR     = Dm - np.hstack((Ck,r_I)) @ S_E
    SSR    = np.sum(SR**2,axis=1,keepdims=1)
    if isinstance(S_I_E,float):
        abs_max_exc_ri = np.nan
    else:
        abs_max_exc_ri = (r_I_E.max(axis=0)- r_I_E.min(axis=0))/2
    
    E_ch   = Dm - Ck @ S_hat    
    if isinstance(r_I,float):
        Ck_r_I= Ck
    else:
        Ck_r_I= np.hstack((Ck,r_I))
    myaux = np.diagflat(np.diag(Ck_r_I.T @ Ck_r_I))    
    A     = np.linalg.pinv(myaux) 
    A_    = np.reshape(np.diag(A),(A.shape[0],1))
    e_T_e = np.diag(SR.T @ SR)
    e_T_e = e_T_e.T
    # 1.96 = 95 %  CI in a t-distribution
    S_E_CONF_INT   = 1.96 * np.sqrt(np.tile(e_T_e,(A_.shape[0],1)) * np.tile(A_,(1,e_T_e.shape[0])))
    
    eiot_obj                   = {'S_hat' : S_hat}
    eiot_obj['S_I']            = S_I
    eiot_obj['S_E']            = S_E
    eiot_obj['E_ch']           = E_ch
    eiot_obj['r_I']            = r_I
    eiot_obj['SR']             = SR
    eiot_obj['SSR']            = SSR
    eiot_obj['num_sI']         = num_sI
    eiot_obj['Rk_mean']        = Rk_mean
    eiot_obj['Rk_std']         = Rk_std
    
    if flag_force_no_exclusives==1:
        eiot_obj['num_e_sI'] = 0
    else:
        eiot_obj['num_e_sI'] = num_e_sI 
        
    eiot_obj['abs_max_exc_ri'] = abs_max_exc_ri
    eiot_obj['S_E_CONF_INT']   = S_E_CONF_INT
    
    #Convert Numpy objects to lists and dict for PYOMO

    pyo_L = np.arange(1,eiot_obj['S_hat'].shape[1]+1)  #index for wavenumbers
    pyo_N = np.arange(1,eiot_obj['S_hat'].shape[0]+1)  #index for chemical species
    pyo_Me= np.arange(eiot_obj['num_sI']-eiot_obj['num_e_sI']+1,eiot_obj['num_sI']+1)
    pyo_L = pyo_L.tolist()
    pyo_N = pyo_N.tolist()
    pyo_Me= pyo_Me.tolist()
    
    pyo_S_hat     = ee.np2D2pyomo(eiot_obj['S_hat']) #convert numpy to dictionary
    pyo_S_I = ee.np2D2pyomo(eiot_obj['S_I'])   #convert numpy to dictionary
    pyo_M   = np.arange(1,eiot_obj['S_I'].shape[0]+1)    #index for non-chemical interferences
    pyo_M   = pyo_M.tolist()
    
    if num_e_sI!=0:
        aux = num_sI-num_e_sI 
        aux_dict=dict(((j+aux+1), abs_max_exc_ri[j]) for j in range(len(abs_max_exc_ri)))
    
    
    eiot_obj['pyo_L']          = pyo_L
    eiot_obj['pyo_N']          = pyo_N
    eiot_obj['pyo_M']          = pyo_M
    eiot_obj['pyo_S_hat']      = pyo_S_hat
    eiot_obj['pyo_S_I']        = pyo_S_I
    eiot_obj['pyo_Me']         = pyo_Me
    eiot_obj['index_rk_eq']    = index_rk_eq
    eiot_obj['index_rk_ex_eq'] = index_rk_ex_eq
    eiot_obj['lambdas']        = lambdas
    
    if num_e_sI!=0:
        eiot_obj['pyo_abs_max_exc_ri']=aux_dict
    
    return eiot_obj
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
def build_(Dm,Ck,num_sI):
        
    S_hat  = np.linalg.pinv(Ck.T@Ck)@Ck.T@Dm
    Dm_hat = Ck @ S_hat
    
    if num_sI>0:
        E_ch       = Dm-Dm_hat
        [U,S,Vh]   = np.linalg.svd(E_ch)
        V          = Vh.T
        S_I        = V[:,0:num_sI]
        S_short    = S[0:num_sI]
        r_I        = U[:,0:num_sI] * np.tile(S_short,(U.shape[0],1))
        S_E        = np.vstack((S_hat,S_I.T))
        SR         = Dm-np.hstack((Ck,r_I)) @ S_E
        SSR        = np.sum(SR**2,1,keepdims=1)
        lambdas    = S[num_sI]
    else:
        S_I        = np.nan
        S_E        = S_hat
        SR         = Dm-Dm_hat 
        E_ch       = SR
        [U,S,Vh]   = np.linalg.svd(E_ch)
        V          = Vh.T
        lambdas    = np.diag(S)
        lambdas    = np.diag(lambdas)
        SSR        = np.sum(SR**2,axis=1)  
        r_I        = np.nan

   #Conf. Intervals for S vectors.
    if isinstance(r_I,float):   #equivalent to isnan(r_I)
        Ck_r_I= Ck
    else:
        Ck_r_I= np.hstack((Ck,r_I))
    A     = np.linalg.pinv(Ck_r_I.T @ Ck_r_I ) 
    A_    = np.reshape(np.diag(A),(A.shape[0],1))
    e_T_e = np.diag(SR.T @ SR)
    e_T_e = e_T_e.T
    # 1.96 = 95 %  CI in a t-distribution
    S_E_CONF_INT   = 1.96 * np.sqrt(np.tile(e_T_e,(A_.shape[0],1)) * np.tile(A_,(1,e_T_e.shape[0])))

    if not(isinstance(S_I,float)):
        S_I=S_I.T
    
    eiot_obj                   = {'S_hat' : S_hat}
    eiot_obj['S_I']            = S_I
    eiot_obj['S_E']            = S_E
    eiot_obj['E_ch']           = E_ch
    eiot_obj['r_I']            = r_I
    eiot_obj['SR']             = SR
    eiot_obj['SSR']            = SSR
    eiot_obj['num_sI']         = num_sI
    eiot_obj['num_e_sI']       = 0
    eiot_obj['abs_max_exc_ri'] = np.nan
    eiot_obj['S_E_CONF_INT']   = S_E_CONF_INT
    
    #Convert Numpy objects to lists and dict for PYOMO

    pyo_L = np.arange(1,eiot_obj['S_hat'].shape[1]+1)  #index for wavenumbers
    pyo_N = np.arange(1,eiot_obj['S_hat'].shape[0]+1)  #index for chemical species
    pyo_L = pyo_L.tolist()
    pyo_N = pyo_N.tolist()
    
    pyo_S_hat     = ee.np2D2pyomo(eiot_obj['S_hat']) #convert numpy to dictionary
    
    if not(isinstance(S_I,float)):
        pyo_S_I = ee.np2D2pyomo(eiot_obj['S_I'])   #convert numpy to dictionary
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
    eiot_obj['index_rk_eq'] = False
    eiot_obj['index_rk_ex_eq']=False
    eiot_obj['lambdas'] = lambdas
    
    return eiot_obj
        
        
def calc(dm,eiot_obj,*,sum_r_nrs=0,see_solver_diagnostics=False,r_ik=False):  
    """
    
    EIOT CALC Routines by Salvador Garcia-Munoz (sgarciam@ic.ac.uk, sal.garcia@lilly.com)

    Parameters
    ----------
    dm : TYPE: Numpy Array or Vector.
        DESCRIPTION. Matrix [o x lambda] or Vector of mixture spectra. 
    eiot_obj : TYPE: Dictionary object created by 'eiot.build'.
        DESCRIPTION: Parameters for EIOT calculations.
    sum_r_nrs : TYPE: float or Numpy vector OPTIONAL.
        DESCRIPTION: Sumation of fraction for non-resolved species. The default is 0.
    see_solver_diagnostics : TYPE, Boolean OPTIONAL.
        DESCRIPTION: If True will display output from IPOPT.
    r_ik : TYPE: Numpy Array.
        DESCRIPTION: Supervised non-chemical interferences per sample, for active supervision mode.

    Returns
    -------
    eiot_prediction_object: TYPE: Dictionary.
        DESCRIPTION: Predictions and diagnostics.

    """
    print('Evaluating EIOT on data')
    rk=r_ik
    if not(isinstance(sum_r_nrs,np.ndarray)):
        if sum_r_nrs==0:
         sum_r_nrs=np.array([0])
    if eiot_obj['num_e_sI']==0 or not(isinstance(rk,bool)):
        model        = ConcreteModel()
        model.L      = Set(initialize = eiot_obj['pyo_L'] )
        model.N      = Set(initialize = eiot_obj['pyo_N'] )
        model.r      = Var(model.N, within=NonNegativeReals)
        model.dm_hat = Var(model.L, within=Reals)
        model.S_hat  = Param(model.N,model.L,initialize = eiot_obj['pyo_S_hat'])
         
        if not isinstance(eiot_obj['pyo_S_I'],float):
            model.M      = Set(initialize = eiot_obj['pyo_M'] )
            model.ri     = Var(model.M, within=Reals) 
            model.S_I    = Param(model.M,model.L,initialize = eiot_obj['pyo_S_I'])     

        if not isinstance(eiot_obj['pyo_S_I'],float):
            def dm_hat_calc(model,i):
                return model.dm_hat[i] == (sum(model.r[n] * model.S_hat[n,i] for n in model.N  ) 
                                         + sum(model.ri[m] * model.S_I[m,i] for m in model.M  ))
            model.con1 = Constraint(model.L,rule=dm_hat_calc)   
        else:
            def dm_hat_calc(model,i):
                return model.dm_hat[i] == (sum(model.r[n] * model.S_hat[n,i] for n in model.N  ) )
            model.con1 = Constraint(model.L,rule=dm_hat_calc)
            
        if not(isinstance(rk,bool)):    
            model.index_rk_eq = Set(initialize = eiot_obj['index_rk_eq'])
            
        if dm.ndim==1:        
            dm_o      = dm
            dm        = ee.np1D2pyomo(dm)                #convert numpy to dictionary
            
            if isinstance(sum_r_nrs,np.ndarray):  # convert numpy to float
                sum_r_nrs=np.float64(sum_r_nrs)

                
            model.sum_r_nrs = Param(initialize = sum_r_nrs)
            model.dm        = Param(model.L, initialize = dm)
            def sum_r_eq_1(model):
                return sum(model.r[n] for n in model.N) == 1 - model.sum_r_nrs
            model.con2 = Constraint(rule=sum_r_eq_1)
            def obj_rule(model):
                return sum((model.dm[l]-model.dm_hat[l])**2 for l in model.L)
            model.obj = Objective(rule=obj_rule) 
            
            if not isinstance(eiot_obj['pyo_S_I'],float):
                for i in model.M:
                    model.ri[i].value = 0   
                
            if not(isinstance(rk,bool)):
                if isinstance(rk,list):
                    rk=np.array(rk)  
                rk       = rk - eiot_obj['Rk_mean']
                rk       = rk / eiot_obj['Rk_std']
                rk       = ee.np1D2pyomo(rk,indexes=eiot_obj['index_rk_eq'])
                model.rk = Param(model.index_rk_eq,initialize=rk)
               
                def known_supervised_eq_const(model,i):
                    return model.ri[i] == model.rk[i]
                model.rk_eq = Constraint(model.index_rk_eq,rule=known_supervised_eq_const)
            for i in model.N:
                model.r[i].value = (1 - model.sum_r_nrs)/max(model.N)
             
            for i in model.L:
                model.dm_hat[i] =model.S_hat[1,i]

            results = auto_nlpsolver(model,tee=see_solver_diagnostics)
            r_hat = []
            for i in model.r:
                r_hat.append(value(model.r[i]))       
            r_I_hat = []
            if not isinstance(eiot_obj['pyo_S_I'],float):
                for i in model.ri:
                    r_I_hat.append(value(model.ri[i]))
                r_I_hat = np.array(r_I_hat)
            r_hat   = np.array(r_hat)
            dm_hat = np.hstack((r_hat,r_I_hat)) @ eiot_obj['S_E']
            sr     = (dm_o - dm_hat)**2
            ssr    = np.sum(sr,axis=0,keepdims=1)
            
        else:
            O=dm.shape[0]
            for o in np.arange(O):
                if np.mod(o,500)==0 and o>0:
                    print('Evaluated 500 more rows of Dm')
                dm_   = dm[o,:]
                dm_o  = dm[o,:]
                dm_   = ee.np1D2pyomo(dm_)                #convert numpy to dictionary

                if isinstance(sum_r_nrs,float):
                    s_r_nrs=sum_r_nrs
                else:
                    if len(sum_r_nrs)>1:   
                        s_r_nrs=sum_r_nrs[o]
                    else:
                        s_r_nrs=sum_r_nrs
                    s_r_nrs=np.float64(s_r_nrs)
                 
                model.sum_r_nrs = Param(initialize = s_r_nrs)
                model.dm        = Param(model.L, initialize = dm_)
                def sum_r_eq_1(model):
                    return sum(model.r[n] for n in model.N) == 1 - model.sum_r_nrs
                model.con2 = Constraint(rule=sum_r_eq_1)
                def obj_rule(model):
                    return sum((model.dm[l]-model.dm_hat[l])**2 for l in model.L)
                model.obj = Objective(rule=obj_rule) 
                if not isinstance(eiot_obj['pyo_S_I'],float):
                    for i in model.M:
                        model.ri[i].value = 0
                    
                if not(isinstance(rk,bool)):
                    rk_       = rk[o,:]
                    rk_       = rk_ - eiot_obj['Rk_mean']
                    rk_       = rk_ / eiot_obj['Rk_std']
                    rk_       = ee.np1D2pyomo(rk_,indexes=eiot_obj['index_rk_eq'])
                    model.rk = Param(model.index_rk_eq,initialize=rk_)
               
                    def known_supervised_eq_const(model,i):
                        return model.ri[i] == model.rk[i]
                    model.rk_eq = Constraint(model.index_rk_eq,rule=known_supervised_eq_const)
                
                for i in model.N:
                    model.r[i].value = (1 - model.sum_r_nrs)/max(model.N)
                    
                for i in model.L:
                    model.dm_hat[i] =model.S_hat[1,i]
                results = auto_nlpsolver(model,tee=see_solver_diagnostics)
                r_hat_ = []
                for i in model.r:
                    r_hat_.append(value(model.r[i]))  
                r_hat_   = np.array(r_hat_)
                r_I_hat_ = []
                if not isinstance(eiot_obj['pyo_S_I'],float):
                    for i in model.ri:
                        r_I_hat_.append(value(model.ri[i]))
                    r_I_hat_ = np.array(r_I_hat_)
                dm_hat_ = np.hstack((r_hat_,r_I_hat_)) @ eiot_obj['S_E']
                sr     = (dm_o - dm_hat_)**2
                ssr_   = np.sum(sr,axis=0,keepdims=1)  
                model.del_component(model.sum_r_nrs)
                model.del_component(model.dm)
                model.del_component(model.con2)
                model.del_component(model.obj)  
                if not(isinstance(rk,bool)):
                    model.del_component(model.rk)
                    model.del_component(model.rk_eq)
                
                if o==0:
                    r_hat   = r_hat_
                    r_I_hat = r_I_hat_
                    ssr     = ssr_
                    dm_hat  = dm_hat_
                else:
                    r_hat   = np.vstack((r_hat,r_hat_))
                    r_I_hat = np.vstack((r_I_hat,r_I_hat_))
                    ssr     = np.vstack((ssr,ssr_))
                    dm_hat  = np.vstack((dm_hat,dm_hat_))
    else:
        model        = ConcreteModel()
        model.L      = Set(initialize = eiot_obj['pyo_L'] )
        model.N      = Set(initialize = eiot_obj['pyo_N'] )
        model.M      = Set(initialize = eiot_obj['pyo_M'] )
        model.Me     = Set(within = model.M,initialize = eiot_obj['pyo_Me'])
        model.S_hat  = Param(model.N,model.L,initialize = eiot_obj['pyo_S_hat'])
        model.S_I    = Param(model.M,model.L,initialize = eiot_obj['pyo_S_I']) 
        model.ri         = Var(model.M, within=Reals) 
        model.r          = Var(model.N, within=NonNegativeReals)
        model.dm_hat     = Var(model.L, within=Reals)
        model.r_I_ex_bin = Var(model.Me, within=Binary)
                             
        def dm_hat_calc(model,i):
            return model.dm_hat[i] == (sum(model.r[n] * model.S_hat[n,i] for n in model.N  ) 
                               + sum(model.ri[m] * model.S_I[m,i] for m in model.M  ))
        model.con1 = Constraint(model.L,rule=dm_hat_calc)
    
        def ri_exc_l(model,i):
            return model.ri[i] == model.r_I_ex_bin[i] 
        model.con4 = Constraint(model.Me,rule=ri_exc_l)
        
        def ri_exc_sum_eq_1(model):
            return sum(model.r_I_ex_bin[n] for n in model.Me) == 1
        model.con5 = Constraint(rule=ri_exc_sum_eq_1)
        
        
        if dm.ndim==1:     
            dm_o         = dm
            dm           = ee.np1D2pyomo(dm)                #convert numpy to dictionary   
            if isinstance(sum_r_nrs,np.ndarray):
                sum_r_nrs=np.float64(sum_r_nrs)
            model.sum_r_nrs = Param(initialize = sum_r_nrs)
            model.dm     = Param(model.L, initialize = dm)
            def sum_r_eq_1(model):
                return sum(model.r[n] for n in model.N) == 1 - model.sum_r_nrs
            model.con2 = Constraint(rule=sum_r_eq_1)           
            def obj_rule(model):
                return sum((model.dm[l]-model.dm_hat[l])**2 for l in model.L)
            model.obj = Objective(rule=obj_rule)     
           
            #Doing explicit enumeration fixing one binary at a time and solving the NLP problem
            counter=1
            for ii in model.Me:
                def fix_binary_ii(model):
                    return model.r_I_ex_bin[ii]==1
                model.con6 =Constraint(rule=fix_binary_ii)
                for i in model.M:
                    model.ri[i].value = 0
                model.ri[ii] = 1
                for i in model.N:
                    model.r[i].value = (1 - model.sum_r_nrs)/max(model.N)
                for i in model.L:
                    model.dm_hat[i] =model.S_hat[1,i]

                results = auto_nlpsolver(model,tee=see_solver_diagnostics)
                model.del_component(model.con6);
                if counter==1:
                    OBJS=value(model.obj)
                else:
                    OBJS=np.vstack((OBJS,value(model.obj)))  
                counter=counter+1;    
            
            indx=eiot_obj['pyo_Me'][np.argmin(OBJS)]  
            
            for i in model.M:
                    model.ri[i].value = 0
            def fix_binary_ii(model):
                return model.r_I_ex_bin[indx]==1
            model.con6 =Constraint(rule=fix_binary_ii)
            model.ri[indx] = 1
            for i in model.N:
                model.r[i].value = (1 - model.sum_r_nrs)/max(model.N)
            for i in model.L:
                model.dm_hat[i] =model.S_hat[1,i]

            results=auto_nlpsolver(model,tee=see_solver_diagnostics)
            r_hat = []
            for i in model.r:
                r_hat.append(value(model.r[i]))   
            r_hat   = np.array(r_hat)
        
            r_I_hat = []
            for i in model.ri:
                r_I_hat.append(value(model.ri[i]))    
            r_I_hat = np.array(r_I_hat)     
        
            dm_hat = np.hstack((r_hat,r_I_hat)) @ eiot_obj['S_E']
            sr     = (dm_o - dm_hat)**2
            ssr    = np.sum(sr,axis=0,keepdims=1)
        else:
            O=dm.shape[0]
            for o in np.arange(O):
                if np.mod(o,500)==0 and o>0:
                    print('Evaluated 500 more rows of Dm')
                dm_   = dm[o,:]
                dm_o  = dm[o,:]
                dm_   = ee.np1D2pyomo(dm_)                #convert numpy to dictionary

                if isinstance(sum_r_nrs,float):
                   s_r_nrs=sum_r_nrs
                else:
                    if len(sum_r_nrs)>1:   
                        s_r_nrs=sum_r_nrs[o]
                    else:
                        s_r_nrs=sum_r_nrs
                    s_r_nrs=np.float64(s_r_nrs)
 
                model.sum_r_nrs = Param(initialize = s_r_nrs)
                model.dm        = Param(model.L, initialize = dm_)
                def sum_r_eq_1(model):
                    return sum(model.r[n] for n in model.N) == 1 - model.sum_r_nrs
                model.con2 = Constraint(rule=sum_r_eq_1)           
                def obj_rule(model):
                    return sum((model.dm[l]-model.dm_hat[l])**2 for l in model.L)
                model.obj = Objective(rule=obj_rule)     
           
                #Doing explicit enumeration fixing one binary at a time and solving the NLP problem
                counter=1
                for ii in model.Me:
                    def fix_binary_ii(model):
                        return model.r_I_ex_bin[ii]==1
                    model.con6 =Constraint(rule=fix_binary_ii)
                    for i in model.M:
                        model.ri[i].value = 0
                    model.ri[ii] = 1
                    for i in model.N:
                        model.r[i].value = (1-model.sum_r_nrs)/max(model.N) 
                    for i in model.L:
                        model.dm_hat[i] =model.S_hat[1,i]

                    results = auto_nlpsolver(model,tee=see_solver_diagnostics)
                    model.del_component(model.con6)
                    if counter==1:
                        OBJS=value(model.obj)
                    else:
                        OBJS=np.vstack((OBJS,value(model.obj)))  
                    counter=counter+1;    
         
                indx=eiot_obj['pyo_Me'][np.argmin(OBJS)]  
    
                def fix_binary_ii(model):
                    return model.r_I_ex_bin[indx]==1
                model.con6 =Constraint(rule=fix_binary_ii)
                for i in model.M:
                    model.ri[i].value = 0    
                model.ri[indx] = 1
                for i in model.N:
                    model.r[i].value = (1-model.sum_r_nrs)/max(model.N)
                for i in model.L:
                    model.dm_hat[i] =model.S_hat[1,i]

                results = auto_nlpsolver(model,tee=see_solver_diagnostics)
                model.del_component(model.con6);
                r_hat_ = []
                for i in model.r:
                    r_hat_.append(value(model.r[i]))   
                r_hat_   = np.array(r_hat_)
        
                r_I_hat_ = []
                for i in model.ri:
                    r_I_hat_.append(value(model.ri[i]))    
                r_I_hat_ = np.array(r_I_hat_)     
        
                dm_hat_ = np.hstack((r_hat_,r_I_hat_)) @ eiot_obj['S_E']
                sr     = (dm_o - dm_hat_)**2
                ssr_   = np.sum(sr,axis=0,keepdims=1)

                model.del_component(model.sum_r_nrs)
                model.del_component(model.dm)
                model.del_component(model.con2)
                model.del_component(model.obj)            
                if o==0:
                    r_hat   = r_hat_
                    r_I_hat = r_I_hat_
                    ssr     = ssr_
                    dm_hat  = dm_hat_
                else:
                    r_hat   = np.vstack((r_hat,r_hat_))
                    r_I_hat = np.vstack((r_I_hat,r_I_hat_))
                    ssr     = np.vstack((ssr,ssr_))
                    dm_hat  = np.vstack((dm_hat,dm_hat_))
    
    M = r_I_hat**2/np.tile(np.var(r_I_hat,axis=0,ddof=1,keepdims=True),(r_I_hat.shape[0],1))
    M = np.sum(M,axis=1,keepdims=True)
    
    output = {'r_hat':r_hat,'r_I_hat':r_I_hat,'dm_hat':dm_hat,'ssr':ssr,'M': M}    
    return output

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def buildwPLS(Dm,R,A,*,num_si_u=0,R_ik=False):
    """
    PLS-EIOT BUILD Routines by Salvador Garcia-Munoz (sgarciam@ic.ac.uk, sal.garcia@lilly.com)

    Parameters
    ----------
    Dm : TYPE, Numpy Array.
        DESCRIPTION. Mixture spectra [o x lambda].
    R : TYPE, Numpy Array.
        DESCRIPTION. Compositional matrix [o x n] with mass/mole/vol fractions.
    A : TYPE, Integer.
        DESCRIPTION. Number of latent variables to use in PLS model.
    num_si_u : TYPE, optional.
        DESCRIPTION. Number of un-supervised non-chemical interferences, the default is 0.
    R_ik : TYPE, Numpy Array.
        DESCRIPTION. Matrix with supervised non-chemical interferences The default is False.

    Returns
    -------
    eiot_pred_obj a Dictionary with predicions and diagnostics

    """
    rk=R_ik
    Ck=R
    num_sI=num_si_u
    if isinstance(rk,bool):     
        print('Building a PLS based EIOT Unsupervised object')
        pls_obj=phi.pls(Ck,Dm,A)
        Dm_hat=phi.pls_pred(Ck,pls_obj)
    else:
        print('Building a PLS based EIOT Supervised object')
        aux=np.hstack((Ck,rk))
        pls_obj=phi.pls(aux,Dm,A)
        Dm_hat=phi.pls_pred(aux,pls_obj)
    pls_obj=phi.conv_pls_2_eiot(pls_obj,r_length=Ck.shape[1])
    
    Dm_hat=Dm_hat['Yhat']
    if num_sI>0:
        E_ch       = Dm-Dm_hat
        [U,S,Vh]   = np.linalg.svd(E_ch)
        V          = Vh.T
        S_I        = V[:,0:num_sI]
        S_short    = S[0:num_sI]
        r_I        = U[:,0:num_sI] * np.tile(S_short,(U.shape[0],1))
        SR         = E_ch- r_I @ S_I.T
        SSR        = np.sum(SR**2,1,keepdims=1)
        lambdas    = S[num_sI]
    else:
        S_I        = np.nan
        SR         = Dm-Dm_hat 
        E_ch       = SR
        [U,S,Vh]   = np.linalg.svd(E_ch)
        V          = Vh.T
        lambdas    = np.diag(S)
        lambdas    = np.diag(lambdas)
        SSR        = np.sum(SR**2,axis=1)  
        r_I        = np.nan
        
    if not(isinstance(S_I,float)):
        S_I=S_I.T
    
    eiot_obj                   = pls_obj
    eiot_obj['wPLS']           = True
    eiot_obj['S_I']            = S_I
    eiot_obj['E_ch']           = E_ch
    eiot_obj['r_I']            = r_I
    eiot_obj['SR']             = SR
    eiot_obj['SSR']            = SSR
    eiot_obj['num_sI']         = num_sI
    eiot_obj['num_e_sI']       = 0
    eiot_obj['abs_max_exc_ri'] = np.nan
    eiot_obj['S_E_CONF_INT']   = np.nan

#Convert Numpy objects to lists and dict for PYOMO    
    if not(isinstance(S_I,float)):
        pyo_S_I = ee.np2D2pyomo(eiot_obj['S_I'])   #convert numpy to dictionary
        pyo_K   = np.arange(1,eiot_obj['S_I'].shape[0]+1)    #index for non-chemical interferences
        pyo_K   = pyo_K.tolist()
    else:
        pyo_S_I = np.nan
        pyo_K   = [0]

    eiot_obj['pyo_K']     = pyo_K
    eiot_obj['pyo_S_I']   = pyo_S_I
    eiot_obj['lambdas'] = lambdas
    return eiot_obj

def calc_pls(dm,eiot_pls_obj,*,sum_r_nrs=0,see_solver_diagnostics=False,r_ik=False): 
    """
    PLS-EIOT CALC Routines by Salvador Garcia-Munoz (sgarciam@ic.ac.uk, sal.garcia@lilly.com)

    Parameters
    ----------
    dm : TYPE Numpy Array
        DESCRIPTION. Matrix [o x lambda] or Vector of mixture spectra.
    eiot_pls_obj : TYPE Dictionary object produced by eiot.buildwPLS
        DESCRIPTION. EIOT object created with PLS.
    sum_r_nrs : TYPE, Scalar or Vector 
        DESCRIPTION. sumation of fractions for non-resolved species optional The default is 0.
    see_solver_diagnostics : TYPE, Boolean optional.
        DESCRIPTION. Display diagnostcs from IPOPT The default is False.
    R_ik : TYPE, Numpy Matrix, optional.
        DESCRIPTION. Supervised non-chemical interferences.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    print('Evaluating a PLS based EIOT on data')
    rk=r_ik
    pls_obj=eiot_pls_obj
    if not(isinstance(sum_r_nrs,np.ndarray)):
        if sum_r_nrs==0:
         sum_r_nrs=np.array([0])
    model         = ConcreteModel()
    model.A       = Set(initialize = pls_obj['pyo_A'] )
    model.N       = Set(initialize = pls_obj['pyo_N'] )
    model.M       = Set(initialize = pls_obj['pyo_M'] )
    model.indx_r  = Set(initialize = pls_obj['indx_r'] ) #elements of x_hat to be constrained to be == 1-sum_r_nrs
    model.y_hat   = Var(model.M, within=Reals)
    model.x_hat   = Var(model.N, within=Reals)
    model.tau     = Var(model.A,within = Reals)
    #model.spe_x   = Var(within = Reals)
    model.ht2     = Var(within = Reals)
    model.Ws      = Param(model.N,model.A,initialize = pls_obj['pyo_Ws'])
    #model.P       = Param(model.N,model.A,initialize = pls_obj['pyo_P'])
    model.Q       = Param(model.M,model.A,initialize = pls_obj['pyo_Q'])
    model.mx      = Param(model.N,initialize = pls_obj['pyo_mx'])
    model.sx      = Param(model.N,initialize = pls_obj['pyo_sx'])
    model.my      = Param(model.M,initialize = pls_obj['pyo_my'])
    model.sy      = Param(model.M,initialize = pls_obj['pyo_sy'])
    model.var_t   = Param(model.M,initialize = pls_obj['pyo_var_t'])
    if not(isinstance(rk,bool)) and (pls_obj['indx_rk_eq']!=0):    
        model.indx_rk_eq = Set(initialize = pls_obj['indx_rk_eq'])
        
    def calc_scores(model,i):
        return model.tau[i] == sum(model.Ws[n,i] * ((model.x_hat[n]-model.mx[n])/model.sx[n]) for n in model.N )
    model.eq1 = Constraint(model.A,rule=calc_scores)
  
    def calc_ht2(model):
        return model.ht2 == sum( model.tau[a]**2/model.var_t[a] for a in model.A)
    model.eq3 = Constraint(rule=calc_ht2)
    
    def non_neg_const(model,i):
            return model.x_hat[i]  >= 0
    model.eq4 = Constraint(model.indx_r, rule=non_neg_const)
          
    if not isinstance(pls_obj['pyo_S_I'],float):
        model.K      = Set(initialize = pls_obj['pyo_K'] )
        model.ri     = Var(model.K, within=Reals) 
        model.S_I    = Param(model.K,model.M,initialize = pls_obj['pyo_S_I'])     
        def y_hat_calc(model,i):
            return model.y_hat[i]== (((sum(model.Q[i,a]*model.tau[a] for a in model.A))*model.sy[i]) + model.my[i]) + sum(model.ri[k] * model.S_I[k,i] for k in model.K)
        model.eq2 = Constraint(model.M,rule=y_hat_calc)
    else:
        def y_hat_calc(model,i):
            return (model.y_hat[i]-model.my[i])/model.sy[i]==sum(model.Q[i,a]*model.tau[a] for a in model.A)
        model.eq2 = Constraint(model.M,rule=y_hat_calc)
        

    if dm.ndim==1:        
        dm_o      = dm
        dm        = ee.np1D2pyomo(dm)         #  convert numpy to dictionary
        
        if isinstance(sum_r_nrs,np.ndarray):  # convert numpy to float
            sum_r_nrs=np.float64(sum_r_nrs)
    
        model.sum_r_nrs = Param(initialize = sum_r_nrs)
        model.dm        = Param(model.M, initialize = dm)
        def sum_r_eq_1(model):
            return sum(model.x_hat[i] for i in model.indx_r) == 1 - model.sum_r_nrs
        model.con2 = Constraint(rule=sum_r_eq_1)
        
        def obj_rule(model):
            return sum((model.dm[m]-model.y_hat[m])**2 for m in model.M) #+ 0.0*model.ht2
        model.obj = Objective(rule=obj_rule) 
    
        for i in model.N:
            model.x_hat[i].value = (1 - model.sum_r_nrs)/max(model.N)
            
        for i in model.M:
            model.y_hat[i] = model.my[i]
                    
        if not isinstance(pls_obj['pyo_S_I'],float):
            for i in model.K:
                model.ri[i].value = 0
    
        if not(isinstance(rk,bool)) and (pls_obj['indx_rk_eq']!=0):
            if isinstance(rk,list):
                rk=np.array(rk)  
            rk       = ee.np1D2pyomo(rk,indexes=pls_obj['indx_rk_eq'])
            model.rk = Param(model.indx_rk_eq,initialize=rk)
           
            def known_supervised_eq_const(model,i):
                return model.x_hat[i] == model.rk[i]
            model.rk_eq = Constraint(model.indx_rk_eq,rule=known_supervised_eq_const)
    

        results = auto_nlpsolver(model,tee=see_solver_diagnostics)
        x_hat = []
        for i in model.x_hat:
            x_hat.append(value(model.x_hat[i]))  
        y_hat = []
        for i in model.y_hat:
            y_hat.append(value(model.y_hat[i]))  
        y_hat   = np.array(y_hat)
        x_hat   = np.array(x_hat)
        tau = []
        for i in model.tau:
            tau.append(value(model.tau[i]))       
        tau   = np.array(tau)
        tau   = tau.reshape(-1,1)
        dm_hat = pls_obj['my'].T+(( pls_obj['Q'] @ tau)*pls_obj['sy'].T)
        sr     = (dm_o - dm_hat.T)**2
        ssr    = np.sum(sr)
        
        r_hat = x_hat[np.array(pls_obj['indx_r'])-1]
        # Need to return the supervised signatures
        if (pls_obj['indx_rk_eq']!=0):
            r_Ik_hat=x_hat[np.array(pls_obj['indx_rk_eq'])-1]
        else:
            r_Ik_hat = np.nan
            
        r_I_hat = []
        if not isinstance(eiot_obj['pyo_S_I'],float):
            for i in model.ri:
                r_I_hat.append(value(model.ri[i]))
            r_I_hat = np.array(r_I_hat)
                
                
    else:
        O=dm.shape[0]
        for o in np.arange(O):
            if np.mod(o,500)==0 and o>0:
                print('Evaluated 500 more rows of Dm')
            dm_   = dm[o,:]
            dm_o  = dm[o,:]
            dm_   = ee.np1D2pyomo(dm_)                #convert numpy to dictionary

            if isinstance(sum_r_nrs,float):
                s_r_nrs=sum_r_nrs
            else:
                if len(sum_r_nrs)>1:   
                    s_r_nrs=sum_r_nrs[o]
                else:
                    s_r_nrs=sum_r_nrs
                s_r_nrs=np.float64(s_r_nrs)
             
            model.sum_r_nrs = Param(initialize = s_r_nrs)
            model.dm        = Param(model.M, initialize = dm_)
            def sum_r_eq_1(model):
                return sum(model.x_hat[i] for i in model.indx_r) == 1 - model.sum_r_nrs
            model.con2 = Constraint(rule=sum_r_eq_1)
        
            def obj_rule(model):
                return sum((model.dm[m]-model.y_hat[m])**2 for m in model.M) #+ 0.0*model.ht2
            model.obj = Objective(rule=obj_rule) 

            for i in model.N:
                model.x_hat[i].value = (1 - model.sum_r_nrs)/max(model.N)
                    
            for i in model.M:
                model.y_hat[i] =model.my[i]
                
            for i in model.A:
                model.tau[i]=0
                
            if not isinstance(pls_obj['pyo_S_I'],float):
                for i in model.K:
                    model.ri[i].value = 0
            
            if not(isinstance(rk,bool)) and (pls_obj['indx_rk_eq']!=0):
                rk_       = rk[o,:]
                rk_       = ee.np1D2pyomo(rk_,indexes=pls_obj['indx_rk_eq'])
                model.rk = Param(model.indx_rk_eq,initialize=rk_)
           
                def known_supervised_eq_const(model,i):
                    return model.x_hat[i] == model.rk[i]
                model.rk_eq = Constraint(model.indx_rk_eq,rule=known_supervised_eq_const)
            
            


            results = auto_nlpsolver(model,tee=see_solver_diagnostics)
            x_hat_ = []
            for i in model.x_hat:
                x_hat_.append(value(model.x_hat[i]))  
            y_hat_ = []
            for i in model.y_hat:
                y_hat_.append(value(model.y_hat[i]))  
            y_hat_   = np.array(y_hat_)
            x_hat_   = np.array(x_hat_)
            tau_ = []
            for i in model.tau:
                tau_.append(value(model.tau[i]))       
            tau_   = np.array(tau_)
            tau_   = tau_.reshape(-1,1)
            dm_hat_ = pls_obj['my'].T+(( pls_obj['Q'] @ tau_)*pls_obj['sy'].T)
            sr_     = (dm_o - dm_hat_.T)**2
            ssr_    = np.sum(sr_)
            
            r_I_hat_ = []
            if not isinstance(pls_obj['pyo_S_I'],float):
                for i in model.ri:
                    r_I_hat_.append(value(model.ri[i]))    
                r_I_hat_ = np.array(r_I_hat_)   
                
            model.del_component(model.sum_r_nrs)
            model.del_component(model.dm)
            model.del_component(model.con2)
            model.del_component(model.obj)  
            if not(isinstance(rk,bool)) and (pls_obj['indx_rk_eq']!=0):
                model.del_component(model.rk)
                model.del_component(model.rk_eq)
            
            if o==0:
                x_hat   = x_hat_
                y_hat   = y_hat_
                tau     = tau_.T
                dm_hat  = dm_hat_
                ssr     = ssr_
                r_I_hat = r_I_hat_
            else:
                x_hat   = np.vstack((x_hat,x_hat_))
                y_hat   = np.vstack((y_hat,y_hat_))
                tau     = np.vstack((tau,tau_.T))
                dm_hat  = np.vstack((dm_hat,dm_hat_))
                ssr     = np.vstack((ssr,ssr_))
                r_I_hat = np.vstack((r_I_hat,r_I_hat_))
                
        r_hat = x_hat[:,np.array(pls_obj['indx_r'])-1]
        # Need to return the supervised signatures
        if (pls_obj['indx_rk_eq']!=0):
            r_Ik_hat=x_hat[:,np.array(pls_obj['indx_rk_eq'])-1]
        else:
            r_Ik_hat = np.nan
    
    M = r_I_hat**2/np.tile(np.var(r_I_hat,axis=0,ddof=1,keepdims=True),(r_I_hat.shape[0],1))
    M = np.sum(M,axis=1,keepdims=True)
    
    output = {'r_hat':r_hat,'x_hat':x_hat,'dm_hat':dm_hat,'tau':tau,'ssr':ssr,'r_I_hat':r_I_hat,'r_Ik_hat':r_Ik_hat, 'M':M}
    return output
