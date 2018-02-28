from boot import *
import seaborn as sns
import scipy.stats as st
sns.set_style('white',rc={'text.usetex':True})

ts_depth = 1940
ts = df.query('area > 0 & Depth == @ts_depth')
ps = get_eigs(ts)
ts_diag = ts
ts_diag[['c1','c2','c3']]=np.array(ps[0])
od_var = get_var_od(ts_diag)
od_std = np.sqrt(od_var)

resamps = bootstrap_df(ts_diag,['c1','c2','c3'],lambda x: vec_resamp(x,ps[1]),n_resamps=5000) 
resamps_min = resamps.min()
resamps_max = resamps.max()

fig,ax=plt.subplots()
ax.grid(linestyle='dotted',color='k')
colors = ['#1b9e77','#d95f02','#7570b3']
sns.kdeplot(resamps['c1'],ax=ax,color=colors[0])
sns.kdeplot(resamps['c2'],ax=ax,color=colors[1])
sns.kdeplot(resamps['c3'],ax=ax,color=colors[2])

for i in xrange(0,3):
    x = np.linspace(-od_std[i]*4,od_std[i]*4,100)
    y = st.norm.pdf(x,0,od_std[i])
    ax.plot(x,y,label=i,color=colors[i],linestyle='--',lw=1.5)


 
labs = [r'bootstrap $V_{23}$',r'bootstrap $V_{13}$',r'bootstrap $V_{12}$',r'analytic $V_{23}$',r'analytic $V_{13}$',r'analytic $V_{12}$']
ax.set_xlabel('Approx. rotation angle (rad)')
ax.set_ylabel('Density')
ax.legend(labs)
fig.tight_layout()
ax.autoscale()
ax.set_xlim(-0.3,0.3)
fig.set_size_inches(2,1.25)
print resamps.std()
print np.sqrt(od_var)
