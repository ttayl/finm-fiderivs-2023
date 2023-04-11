import pandas as pd
import numpy as np
from scipy.optimize import fsolve


def format_bintree(df,style='{:.2f}'):
    return df.style.format(style,na_rep='').format_index('{:.2f}',axis=1)


def construct_rate_tree(dt,T):
    timegrid = pd.Series(np.arange(0,T+dt,dt),name='time',index=pd.Index(range(int(T/dt)+1),name='state'))
    tree = pd.DataFrame(dtype=float,columns=timegrid,index=timegrid.index)
    
    return tree

def construct_quotes(maturities,prices):
    quotes = pd.DataFrame({'maturity':maturities,'price':prices})    
    quotes['continuous ytm'] = -np.log(quotes['price']/100) / quotes['maturity']
    quotes.set_index('maturity',inplace=True)
    
    return quotes





def payoff_bond(r,dt,facevalue=100):
    price = np.exp(-r * dt) * facevalue
    return price





def replicating_port(quotes,undertree,derivtree,dt=None,Ncash=100):
    if dt is None:
        dt = undertree.columns[1] - undertree.columns[0]
    
    delta = (derivtree.loc[0,dt] - derivtree.loc[1,dt]) / (undertree.loc[0,dt] - undertree.loc[1,dt]) 
    cash = (derivtree.loc[0,dt] - delta * undertree.loc[0,dt]) / Ncash
    
    out = pd.DataFrame({'positions':[cash,delta], 'value':quotes},index=['cash','under'])
    out.loc['derivative','value'] = out['positions'] @ out['value']
    return out





def bintree_pricing(payoff=None, ratetree=None, undertree=None,cftree=None, pstars=None,timing=None,style='european'):
    
    if payoff is None:
        payoff = lambda r: payoff_bond(r,dt)
    
    if undertree is None:
        undertree = ratetree
        
    if cftree is None:
        cftree = pd.DataFrame(0, index=undertree.index, columns=undertree.columns)
        
    if pstars is None:
        pstars = pd.Series(.5, index=undertree.columns)

    if undertree.columns.to_series().diff().std()>1e-8:
        display('time grid is unevenly spaced')
    dt = undertree.columns[1]-undertree.columns[0]

    
    valuetree = pd.DataFrame(dtype=float, index=undertree.index, columns=undertree.columns)

    for steps_back, t in enumerate(valuetree.columns[-1::-1]):
        if steps_back==0:                           
            valuetree[t] = payoff(undertree[t])
            if timing== 'deferred':
                valuetree[t] *= np.exp(-ratetree[t]*dt)
        else:
            for state in valuetree[t].index[:-1]:
                valuetree.loc[state,t] = np.exp(-ratetree.loc[state,t]*dt) * (pstars[t] * valuetree.iloc[state,-steps_back] + (1-pstars[t]) * valuetree.iloc[state+1,-steps_back] + cftree.loc[state,t])

            if style=='american':
                valuetree.loc[:,t] = np.maximum(valuetree.loc[:,t],payoff(undertree.loc[:,t]) + np.exp(-ratetree.loc[:,t]*dt) * cftree.loc[:,t])

    return valuetree




def bond_price_error(quote, pstars, ratetree, style='european'):
    FACEVALUE = 100
    dt = ratetree.columns[1] - ratetree.columns[0]    
    payoff = lambda r: payoff_bond(r,dt)
    modelprice = bintree_pricing(payoff, ratetree, pstars=pstars, style=style).loc[0,0]
    error = modelprice - quote

    return error            








def estimate_pstar(quotes,ratetree,style='european'):

    pstars = pd.Series(dtype=float, index= ratetree.columns[:-1], name='pstar')
    p0 = .5
    
    for steps_forward, t in enumerate(ratetree.columns[1:]):        
        ratetreeT = ratetree.copy().loc[:,:t].dropna(axis=0,how='all')
        t_prev = ratetreeT.columns[steps_forward]
        
        pstars_solved = pstars.loc[:t_prev].iloc[:-1]
        wrapper_fun = lambda p: bond_price_error(quotes['price'].iloc[steps_forward+1], pd.concat([pstars_solved, pd.Series(p,index=[t_prev])]), ratetreeT, style=style)

        pstars[t_prev] = fsolve(wrapper_fun,p0)[0]

    return pstars



def exercise_decisions(payoff, undertree, derivtree):
    exer = (derivtree == payoff(undertree)) & (derivtree > 0)
    return exer






def rates_to_BDTstates(ratetree):
    ztree = np.log(100*ratetree)
    return ztree

def BDTstates_to_rates(ztree):
    ratetree = np.exp(ztree)/100
    return ratetree

def incrementBDTtree(ratetree, theta, sigma, dt=None):
    if dt is None:
        dt = ratetree.columns[1] - ratetree.columns[0]

    tstep = len(ratetree.columns)-1
    
    ztree = rates_to_BDTstates(ratetree)
    ztree.iloc[:,-1] = ztree.iloc[:,-2] + theta * dt + sigma * np.sqrt(dt)
    ztree.iloc[-1,-1] = ztree.iloc[-2,-2] + theta * dt - sigma * np.sqrt(dt)
    
    newtree = BDTstates_to_rates(ztree)
    return newtree

def incremental_BDT_pricing(tree, theta, sigma_new, dt=None):
    if dt==None:
        dt = tree.columns[1] - tree.columns[0]
        
    pstars = pd.Series(.5,index=tree.columns)
    
    payoff = lambda r: payoff_bond(r,dt)
    newtree = incrementBDTtree(tree, theta, sigma_new)
    model_price = bintree_pricing(payoff, newtree, pstars=pstars)
    return model_price
