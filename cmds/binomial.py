import pandas as pd
import numpy as np
from scipy.optimize import fsolve


def construct_rate_tree(dt,T):
    timegrid = pd.Series(np.arange(0,T,dt),name='time',index=pd.Index(range(int(T/dt)),name='state'))
    tree = pd.DataFrame(dtype=float,columns=timegrid,index=timegrid.index)

    formatted = tree.style.format('{:.2%}',na_rep='').format_index('{:.1f}',axis=1)
    
    return tree, formatted

def construct_quotes(maturities,prices):
    quotes = pd.DataFrame({'maturity':maturities,'price':prices})    
    quotes['continuous ytm'] = -np.log(quotes['price']/100) / quotes['maturity']
    quotes.set_index('maturity',inplace=True)
    style = quotes.style.format({'price':'{:.4f}','continuous ytm':'{:.2%}'}).format_index('{:.1f}')
    
    return quotes, style


def bintree_pricing(payoff,ratetree,undertree,pstars,style='european'):

    if undertree.columns.to_series().diff().std()>1e-8:
        display('time grid is unevenly spaced')
        
    dt = undertree.columns[1]-undertree.columns[0]
    
    valuetree = pd.DataFrame(dtype=float, index=undertree.index, columns=undertree.columns)

    for steps_back, t in enumerate(valuetree.columns[-1::-1]):
        if steps_back==0:
            valuetree[t] = payoff(undertree[t])
        else:
            for state in valuetree[t].index[:-1]:
                valuetree.loc[state,t] = np.exp(-ratetree.loc[state,t]*dt) * (pstars[t] * valuetree.iloc[state,-steps_back] + (1-pstars[t]) * valuetree.iloc[state+1,-steps_back])

            if style=='american':
                valuetree.loc[:,t] = np.maximum(valuetree.loc[:,t],payoff(undertree.loc[:,t]))
    
    return valuetree

def estimate_pstar_one_period(ratetree, undertree, quote):

    dt = ratetree.columns[1] - ratetree.columns[0]    
    A = np.exp(ratetree.iloc[0,0] * dt)
    
    pstar = (A * quote - undertree.iloc[1,1])/(undertree.iloc[0,1] - undertree.iloc[1,1])
    return pstar


def payoff_bond(r,dt,facevalue=100):
    price = np.exp(-r * dt) * facevalue
    return price

def bond_price_error(quote, pstars, ratetree, style='european'):
    FACEVALUE = 100
    dt = ratetree.columns[1] - ratetree.columns[0]    
    payoff = lambda r: payoff_bond(r,dt)
    modelprice = bintree_pricing(payoff, ratetree, ratetree, pstars, style=style).loc[0,0]
    error = modelprice - quote

    return error            


def estimate_pstar(quotes,ratetree,style='european'):

    pstars = pd.Series(dtype=float, index= ratetree.columns[:-1])        
    p0 = .5
    
    for steps_forward, t in enumerate(ratetree.columns[1:]):        
        ratetreeT = ratetree.copy().loc[:,:t].dropna(axis=0,how='all')
        t_prev = ratetreeT.columns[steps_forward]
        
        pstars_solved = pstars.loc[:t_prev].iloc[:-1]
        wrapper_fun = lambda p: bond_price_error(quotes['price'].iloc[steps_forward+1], pd.concat([pstars_solved, pd.Series(p,index=[t_prev])]), ratetreeT, style=style)

        pstars[t_prev] = fsolve(wrapper_fun,p0)[0]
        
    return pstars
