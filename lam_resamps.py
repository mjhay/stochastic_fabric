from boot import *


ef_resamps = df.groupby("Depth").apply(lambda x:
        bootstrap_df(x,["Depth","ef"],ef_resamp,n_resamps=1000))

eigs_resamps = df.groupby("Depth").apply(lambda x:
        bootstrap_df(x,["Depth","lam1","lam2","lam3"], eig_resamp, n_resamps=1000))

def get_efquants():
    efquants=ef_resamps.groupby("Depth").apply(lambda x:
                x["ef"].quantile([0.05,0.5,0.95])) 
    efquants["Depth"]=efquants.index
    return efquants
def plot_efquants(efquants):
    fig,ax=plt.subplots()
    ax.errorbar(x=efquants["Depth"],y=efquants[0.5],
            yerr=[efquants[0.95] - efquants[0.5],
            efquants[0.5]-efquants[0.05]],ls="None",
            fmt='o',elinewidth=1,markersize=3,color='black',
            markeredgewidth=0.3,markeredgecolor='k')
    ax.set_xlabel("Depth (m)",fontsize=10)
    ax.set_ylabel("Enhancement factor",fontsize=10)
    ax.grid(linewidth=1)
    fig.tight_layout()
    fig.set_size_inches([10,5.675],forward=True)
    plt.show()
    return (fig,ax)

def get_lamquants(df):
    lamquants = [0,0,0]
    for i in xrange(0,3):
        lamquants[i]=eigs_resamps.groupby("Depth").apply(lambda x:
                x["lam{0}".format(i+1)].quantile([0.05,0.5,0.95])) 
        lamquants[i]["Depth"]=lamquants[i].index
    xticks = np.arange(0,3501,500)
    return lamquants
def plot_lamquants(lamquants):
    fig,ax=plt.subplots()
    for i in xrange(0,3):
        ax.errorbar(x=lamquants[i]["Depth"],y=lamquants[i][0.5],yerr=[lamquants[i][0.95] - lamquants[i][0.5],lamquants[i][0.5]-lamquants[i][0.05]],ls="None",fmt='o')
        ax.scatter(x=lamquants[i]["Depth"],y=lamquants[i][0.5])
    fig.tight_layout()
    fig.set_size_inches([16,9])
    plt.show()

    return (fig,ax)

def plot_lamquants(lamquants):
    fig,ax=plt.subplots()
    marks = ['^','o','v']
    colors = ['#1b9e77','#d95f02','#7570b3']
    labs = [r'$\lambda_1$',r'$\lambda_2$',r'$\lambda_3$']
    for i in xrange(0,3):
        ax.errorbar(x=lamquants[i]["Depth"],y=lamquants[i][0.5],yerr=[lamquants[i][0.95] - lamquants[i][0.5],lamquants[i][0.5]-lamquants[i][0.05]],ls="None",fmt='o',elinewidth=1,markersize=3.5,marker=marks[i],label=labs[i],color=colors[i],markeredgewidth=0.3,markeredgecolor=colors[i])
        ax.set_xlabel("Depth (m)",fontsize=10)
        ax.set_ylabel("Eigenvalue",fontsize=10)
        ax.grid(linewidth=1)
    ax.legend(loc='upper left',fontsize=10)
    ax.set_xlim([0,3500])
    ax.set_ylim([0,1])
    fig.set_size_inches([10,5.675],forward=True)
    ax.annotate('a',xy=(2905,0.5))
    ax.annotate('a',xy=(3365,0.5))
    ax.annotate('a',xy=(3405,0.5))
    plt.show()
    return (fig,ax)

ax.scatter(x=lamquants[i]["Depth"],y=lamquants[i][0.5],s=1)
lamquants = get_lamquants(df)
(figlam,axlam)=plot_lamquants(lamquants)
figlam.savefig("eig_quants.pdf")
efquants = get_efquants()
(figef,axef)=plot_efquants(efquants)
figef.savefig("ef_quants.pdf")
