import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ratecurve_to_discountcurve(ratecurve, n_compound=None):

    if isinstance(ratecurve,pd.DataFrame):
        ratecurve = ratecurve.iloc[:,0]
        
    if n_compound is None:
        discountcurve = np.exp(-ratecurve * ratecurve.index)
    else:
        discountcurve = 1 / (1+(ratecurve / n_compound))**(n_compound * ratecurve.index)

    return discountcurve  





def ratecurve_to_forwardcurve(ratecurve, n_compound=None, dt=None):
    if isinstance(ratecurve,pd.DataFrame):
        ratecurve = ratecurve.iloc[:,0]
        
    if dt is None:
        dt = ratecurve.index[1] - ratecurve.index[0]
        
    discountcurve = ratecurve_to_discountcurve(ratecurve, n_compound=n_compound)
    
    F = discountcurve / discountcurve.shift()
    
    forwardcurve = n_compound * (1/(F**(n_compound * dt)) - 1)
    
    return forwardcurve




def discount_to_intrate(discount, maturity, n_compound=None):
        
    if n_compound is None:
        intrate = - np.log(discount) / maturity
    
    else:
        intrate = n_compound * (1/discount**(1/(n_compound * maturity)) - 1)    
        
    return intrate



def interp_curves(data,dt=None, date=None, interp_method='linear',order=None, extrapolate=True):
    if dt is None:
        dt = data.columns[1] - data.columns[0]
    
    if date is None:
        temp = data
    else:
        temp = data.loc[date,:]
        
    curves = pd.DataFrame(dtype=float, index=np.arange(dt,temp.index[-1]+dt,dt),columns=['quotes','interp'])
    curves.loc[temp.index,['quotes','interp']] = temp
    
    if extrapolate:
        curves['interp'].interpolate(method=interp_method, order=order, limit_direction='both', fill_value = 'extrapolate',inplace=True)
    else:
        curves['interp'].interpolate(method=interp_method, order=order,inplace=True)
    
    return curves

def plot_interp_curves(curves,plot_contin=True):
    fig, ax = plt.subplots()
    curves['quotes'].plot.line(ax=ax, linestyle='None',marker='*')
    curves.iloc[:,1:].plot.line(ax=ax, linestyle='--',marker='')
            
    plt.legend()
    plt.show()

    
    
    
def price_bond(ytm, T, cpn, cpnfreq=2, face=100, accr_frac=0):
    ytm_n = ytm/cpnfreq
    cpn_n = cpn/cpnfreq
    accr_frac = (T-round(T))*cpnfreq
    N = T * cpnfreq
    price = face * ((cpn_n / ytm_n) * (1-(1+ytm_n)**(-N)) + (1+ytm_n)**(-N)) * (1+ytm_n)**(accr_frac)
    return price