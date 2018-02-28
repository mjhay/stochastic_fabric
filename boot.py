from numba import jit
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

plt.style.use("seaborn-paper")
df = pd.read_csv("wais_fabric_v4.csv")
df = df[df['area']>0]

def get_var_weighted(x,weights):
    s = weights.sum()
    mu = np.dot(x,weights)/s
    return (((x-mu)**2*weights)).sum()


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def get_eigs(df,return_V=False):
    ps = df[["c1","c2","c3"]]
    area = df["area"]
    tot_area = np.sum(area)
    A2 = np.matmul(ps.T*area,ps)/tot_area
    (vals,V) = np.linalg.eigh(A2)
    ps_diag = np.matmul(ps,V)
    a2_od = np.array([A2[1,2],A2[0,2],A2[0,1]])
    a4 = np.zeros(3)
    #where a4_od_2323 = 0, 1313 = 1, 1212 = 2
    a4_od = np.zeros(3)
    for i in xrange(0,3):
        a4[i] = np.dot(ps_diag[:,i]**4,df["area"])/tot_area
    a4_od[0] = np.dot(ps_diag[:,1]**2*ps_diag[:,2]**2,df["area"])/tot_area
    a4_od[1] = np.dot(ps_diag[:,0]**2*ps_diag[:,2]**2,df["area"])/tot_area
    a4_od[2] = np.dot(ps_diag[:,0]**2*ps_diag[:,1]**2,df["area"])/tot_area
    ps_diag = pd.DataFrame(ps_diag,columns=["c1","c2","c3"])
    if return_V == False:
        return [ps_diag,vals,a4,a2_od,a4_od,df['area']]
    else:
        return [ps_diag,vals,a4,a2_od,a4_od,df['area'],V]

def get_var(df):
    vars = np.array([get_var_weighted(df[i]**2,df['area']) for i in ['c1','c2','c3']])
    sn2 = np.sum(df["area"]**2)/np.sum(df["area"])**2
    return vars*sn2

def get_var_od(ts):
#    (ps_diag,vals,a4,a2_od,a4_od,area) = get_eigs(ps)
    od_var = np.zeros(3)
    od_var[0] = get_var_weighted(ts['c2']*ts['c3'],ts['area'])
    od_var[1] = get_var_weighted(ts['c1']*ts['c3'],ts['area'])
    od_var[2] = get_var_weighted(ts['c1']*ts['c2'],ts['area'])
    vals = [((ts[i]**2)*ts['area']).sum() for i in ['c1','c2','c3']] 
    z = np.zeros(3)
    z[0] = vals[2]-vals[1]
    z[1] = vals[2]-vals[0]
    z[2] = vals[1]-vals[0]
    sn2 = np.sum(ts['area']**2)/np.sum(ts['area'])**2
    return od_var/z**2*sn2

def vec_resamp(resamp,vals):
    (ps_diag,vals,a4,a2_od,a4_od,area,V) = get_eigs(resamp,True)
#    a2_od = np.zeros(3)
#    a2_od[0] = (resamp['c2']*resamp['c3']*resamp['area']).sum()
#    a2_od[1] = (resamp['c1']*resamp['c3']*resamp['area']).sum()
#    a2_od[2] = (resamp['c1']*resamp['c2']*resamp['area']).sum()
#    vals = np.array([(resamp[i]**2*resamp['area']).sum() for i in 'c1','c2','c3'])
#    z0 = vals[2]-vals[1]
#    z1 = vals[2]-vals[0]
#    z2 = vals[1]-vals[0]
    return np.array([V[1,2], V[0,2], V[0,1]])
    return np.array([V[1,2]/z0, V[0,2]/z1, V[0,1]/z2])

def eig_resamp_uw(resamp):
    ps = resamp.values[:,1:4]
    A2 = np.matmul(ps.T,ps)/(ps.shape[0])
    (vals,V) = np.linalg.eigh(A2)
    return vals
    #ps_diag = np.matmul(ps,V)

def eig_resamp(resamp):
    lams = get_eigs(resamp)[1]
    return lams
#    return np.array([resamp.iloc[0]["Depth"], lams[0], lams[1], lams[2] ])

def sq_rss(p,ax=np.array([0,1/np.sqrt(2),1/np.sqrt(2)])):
#    Sn = [0, p[2], p[1]]
    return p[2]*p[2] + p[1]*p[1] - (p[2]*p[1]+p[1]*p[2])**2
#    return np.dot(Sn,Sn) - np.dot(Sn,p)**2

def ef_resamp(resamp):
    return np.array([resamp.iloc[0]["Depth"], np.dot(resamp["sf"],resamp["area"])**2/np.sum(resamp["area"])**2*25/4])

def bootstrap_df(df, col_names, stat_fun, n_resamps = 500, columns=None, size_resamp=1.):
    init_df = stat_fun(df.sample(frac=1.0,replace=True))
    boot_df = pd.DataFrame(columns=col_names, 
            data=np.zeros((n_resamps,len(col_names))))
    for i in xrange(0,n_resamps):
        resamp = df.sample(frac=size_resamp,replace=True)
#        resamp = resamp_until(df)
        boot_df.iloc[i] =  stat_fun(resamp)
    return boot_df
def resamp_rescale(df):
    resamp = df.sample(frac=1.0,replace=True)
    resamp['area'] = resamp['area']/resamp['area'].sum()
    return resamp

@jit
def resamp_until(df):
    tot_area = 0
    n = df.shape[0]
    indices = np.zeros(df.shape[0]*10)
    j = 0
    while tot_area < 1 and j<n:
        i = np.random.randint(0,n)
        tot_area += df.iloc[i]['area']
        indices[j] = i
        j += 1
    return df.iloc[indices[0:j]]

def bootstrap(ser, stat_fun, n_resamps = 500, columns=None, size_resamp=None):
    if size_resamp == None:
        size_resamp = ser.size
    boot_series = pd.Series(np.empty(n_resamps))
    for i in xrange(1,n_resamps):
        resamp = ser.sample(size_resamp,replace=True)
        boot_series[i] =  stat_fun(resamp)
    return boot_series

def group_bootstrap(df, grouping_col, stat_fun,n_resamps=500):
    return df.groupby(grouping_col).apply(
            lambda x: bootstrap(x,stat_fun,n_resamps))

def bootstrap_col(data, grouping_col, boot_col):
    return data.groupby(grouping_col).apply(
            lambda x: bootstrap(x[boot_col],np.mean,n_resamps=1000)).transpose()

def weighted_boot(df, weight_col, n_resamps=1000, until=1.):
    n,d = df.shape
    resamp = df.sample(frac=1.)
    resamp = pd.DataFrame(np.zeros((n*n_resamps, d)), columns = df.columns)
    resamp['resamp_num'] = 0
    current_row = 0
    for i in xrange(0,n_resamps):
        accm = 0.
        print i
        while accm < 1.:
            if current_row >= resamp.shape[0]:
                resamp = resamp.append(pd.DataFrame(np.zeros((n,d))))
            resamp.iloc[current_row,0:-1] = np.array(df.sample()).flatten()
            resamp.iloc[current_row,-1] = i
            accm += resamp[weight_col].iloc[current_row]
            current_row += 1
    return resamp.iloc[0:current_row]        
