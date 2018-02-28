from boot import *
import seaborn as sns
import scipy.stats as st
sns.set_style('white',rc={'text.usetex':True})
colors = ['#1b9e77','#d95f02','#7570b3']

ts_depth = 1940
ts = df.query('area > 0 & Depth == @ts_depth')
ps = get_eigs(ts)
ts_diag = ts
ts_diag[['c1','c2','c3']]=np.array(ps[0])
od_var = get_var(ts)
od_std = np.sqrt(od_var)
resamps = bootstrap_df(ts_diag,['c1','c2','c3'],eig_resamp,n_resamps=3000) 
resamps_min = resamps.mean()
resamps_max = resamps.max()

fig,ax=plt.subplots()
ax.grid(linestyle='dotted',color='k')
sns.kdeplot(resamps['c1'],ax=ax,color=colors[0])
sns.kdeplot(resamps['c2'],ax=ax,color=colors[1])
sns.kdeplot(resamps['c3'],ax=ax,color=colors[2])

for i in xrange(0,3):
    x = np.linspace(ps[1][i]-3*od_std[i],ps[1][i]+3*od_std[i],1000)
    y = st.norm.pdf(x,ps[1][i],np.sqrt(od_var[i]))
    ax.plot(x,y,label=i,linestyle='--',color=colors[i],lw=1.5)


 

labs = [r'bootstrap $\lambda_1$',r'bootstrap $\lambda_2$',r'bootstrap $\lambda_3$',r'analytic $\lambda_1$',r'analytic $\lambda_2$',r'analytic $\lambda_3$']
ax.set_xlabel('Approx. rotation angle (rad)')
ax.set_ylabel('Density')
ax.legend(labs)
fig.tight_layout()
ax.autoscale()
fig.set_size_inches(2,1.25)
print resamps.std()
print np.sqrt(od_var)
plt.show()
