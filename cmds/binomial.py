import pandas as pd
import numpy as np
from scipy.optimize import fsolve



def format_bintree(df,style='{:.2f}'):
    return df.style.format(style,na_rep='').format_index('{:.2f}',axis=1)


def construct_rate_tree(dt,T):
    timegrid = pd.Series((np.arange(0,round(T/dt)+1)*dt).round(6),name='time',index=pd.Index(range(round(T/dt)+1),name='state'))
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

    if undertree.columns.to_series().diff().std()>1e-5:
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
    
    payoff = lambda r: payoff_bond(r,dt)
    newtree = incrementBDTtree(tree, theta, sigma_new)
    model_price = bintree_pricing(payoff, newtree)
    return model_price


def estimate_theta(sigmas,quotes_zeros,dt=None,T=None):
    if dt is None:
        dt = quotes_zeros.index[1] - quotes_zeros.index[0]

    if T is None:
        T = quotes_zeros.index[-2]

    ratetree = construct_rate_tree(dt,T)
    theta = pd.Series(dtype=float, index=ratetree.columns, name='theta')
    dt = ratetree.columns[1] - ratetree.columns[0]
    
    if type(sigmas) is float:
        sigmas = pd.Series(sigmas,index=theta.index)

    for tsteps, t in enumerate(quotes_zeros.index):
        if tsteps==0:
            ratetree.loc[0,0] = -np.log(quotes_zeros.iloc[tsteps]/100)/dt
        else:
            subtree = ratetree.iloc[:tsteps+1,:tsteps+1]
            wrapper = lambda theta: incremental_BDT_pricing(subtree, theta, sigmas.iloc[tsteps]).loc[0,0] - quotes_zeros.iloc[tsteps]
            
            theta.iloc[tsteps] = fsolve(wrapper,.5)[0]
            ratetree.iloc[:,tsteps] = incrementBDTtree(subtree, theta.iloc[tsteps], sigmas.iloc[tsteps]).iloc[:,tsteps]
            
            print(f'Completed: {tsteps/len(quotes_zeros.index):.1%}')
            
    return theta, ratetree





def construct_bond_cftree(T, compound, cpn, cpn_freq=2, face=100):
    step = int(compound/cpn_freq)

    cftree = construct_rate_tree(1/compound, T)
    cftree.iloc[:,:] = 0
    cftree.iloc[:, -1:0:-step] = (cpn/cpn_freq) * face
    
    # final cashflow is accounted for in payoff function
    # drop final period cashflow from cashflow tree
    cftree = cftree.iloc[:,:-1]
    
    return cftree

def construct_accinttree(cftree, compound, cpn, cpn_freq=2, face=100, cleancall=True):
    accinttree = cftree.copy()
    step = int(compound/cpn_freq)
    if cleancall is True:
        accinttree.iloc[:,-1::-step] = face * (cpn/compound)
        
    return accinttree



def price_callable(quotes, fwdvols, cftree, accinttree, wrapper_bond, payoff_call):

    theta, ratetree = estimate_theta(fwdvols,quotes)
    bondtree = bintree_pricing(payoff=wrapper_bond, ratetree=ratetree, cftree= cftree)
    cleantree = np.maximum(bondtree - accinttree,0)
    calltree = bintree_pricing(payoff=payoff_call, ratetree=ratetree, undertree= cleantree, style='american')
    callablebondtree = bondtree - calltree
    model_price = callablebondtree.loc[0,0]

    return model_price



def BDTtree(thetas, sigmas, r0=None, px_bond0=None, dt=None, T=None):

    if dt is None:
        dt = thetas.index[1] - thetas.index[0]

    if T is None:
        T = thetas.index[-1]

    if r0 is None:
        r0 = -np.log(px_bond0)/dt

    ztree = construct_rate_tree(dt,T)
    ztree.iloc[0,0] = rates_to_BDTstates(r0)

    # sigmas is indexed starting at dt, so tsteps is lagged
    for tsteps, t in enumerate(sigmas.index):
        ztree.iloc[:,tsteps+1] = ztree.iloc[:,tsteps] + thetas.iloc[tsteps] * dt + sigmas.iloc[tsteps] * np.sqrt(dt)
        ztree.iloc[tsteps+1,tsteps+1] = ztree.iloc[tsteps,tsteps] + thetas.iloc[tsteps] * dt - sigmas.iloc[tsteps] * np.sqrt(dt)
            
    bdttree = BDTstates_to_rates(ztree)

    return bdttree