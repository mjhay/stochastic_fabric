using PyCall
include("mle.jl")
include("lebedev.jl")

function get_bingham_density(kappa)
    V = eye(3)
    Z=d2(kappa,V,leb_X)
    return x -> (bingham_unnorm_density(kappa,V,x)./Z)
end

p60 = Array(fabric[fabric[:Depth].==dr[60],[:c1,:c2,:c3]])'
Utils.proj2UpHem!(p60)
peig = eigfact(p60*p60')
V = peig[:vectors]
lams = peig[:values]/size(p60,2)
pdiag = V'*p60
Utils.proj2UpHem!(pdiag)
function euclid2lambert(ps)
    X = -sqrt(2./(1+ps[3,:])).*ps[1,:]
    Y = -sqrt(2./(1+ps[3,:])).*ps[2,:]
    r = sqrt(X.^2 + Y.^2)
    theta = atan2(Y,X)
    return (r/sqrt(2),theta)
end

(r_ts,theta_ts)=euclid2lambert(pdiag)

r_samp = linspace(0.001,1,30)
theta_samp = linspace(0,2*pi,50)
RR = [r for r in r_samp, theta in theta_samp]
TT = [theta for r in r_samp, theta in theta_samp]
#x_samp = r_samp.*(cos(theta_samp))
#y_samp = r_samp.*(sin(theta_samp))

XX_samp = RR.*cos(TT)
YY_samp = RR.*sin(TT)

XYZ_samp = zeros(size(XX_samp)[1],size(XX_samp)[2],3)
XYZ_samp[:,:,1]=XX_samp
XYZ_samp[:,:,2]=YY_samp
XYZ_samp[:,:,3]=sqrt(max(0,min(1,1 - XX_samp.^2 - YY_samp.^2)))

bs = dinh_mle_B(p60,false,true)[1]
# = (V'*B*V)[1:2]

kappa = fisher_mle(p60')[2]
bing_kappa = ps_bingham_mle(p60).minimizer
bing_density=get_bingham_density(bing_kappa)
bing_samp = [bing_density(XYZ_samp[i,j,:]) for i in 1:size(XYZ_samp,1), j in 1:size(XYZ_samp,2)]
fisher_samp = [fisher(XYZ_samp[i,j,:],kappa) for i in 1:size(XYZ_samp,1), j in 1:size(XYZ_samp,2)]
dinh_samp = [dinh(XYZ_samp[i,j,:],bs) for i in 1:size(XYZ_samp,1), j in 1:size(XYZ_samp,2)]


@pyimport matplotlib.pyplot as plt
function plot_pts_density(r_ts,theta_ts, r_samp, theta_samp,vals)
    f,axes=plt.subplots(1,3,subplot_kw=Dict("polar"=>true))
    z = [fisher_samp,dinh_samp,bing_samp]
    levels = linspace(0, maximum(dinh_samp), 500)
    levels_fisher = linspace(0,maximum(fisher_samp),30)
    levels = [levels_fisher,levels,levels]
    cs=Array(Any,3)
    for i=1:3
        axes[i][:set_ylim](0,1)
        cs[i]=axes[i][:contourf](TT,RR,z[i],20,cmap=plt.cm[:plasma],levels=levels[i])
        for c in cs[i][:collections]
                c[:set_edgecolor]("face")
        end
        f[:colorbar](cs[i],ax=axes[i],shrink=0.4)
        axes[i][:scatter](theta_ts,r_ts,s=1,color="white")

    end
    plt.show()
    return (f,axes)
end
println("DEPTH")
println(dr[60])
println("DINH")
print(dinh_mle_B(p60)[2])
println("BING")
println(ps_bingham_mle(p60))
println("FISHER MLE")
println(fisher_mle(p60'))
