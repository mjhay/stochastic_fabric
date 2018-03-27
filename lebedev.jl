using ForwardDiff, Distributions, NLsolve,DataFrames,Gadfly,Combinatorics
include("Utils.jl")
#include("gausspro.jl")
import  Distributions.Normal
const lebedev_n=5810
function ldlebedev_n()
    lebedev_full=5810
    x=Array(Float64,lebedev_full)
    y=Array(Float64,lebedev_full)
    z=Array(Float64,lebedev_full)
    w=Array(Float64,lebedev_full)

    ccall((:ld5810,"./leb"), Void, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), x, y, z, w)

    return (x,y,z,w)
end

function readlarr(file="larr.csv")
    larr=Array(Float64,3,3,3,3,100,100)
    larrdf=readtable(file)
    for w=1:nrow(larrdf)
        x=larrdf[w,[:I,:J,:K,:L,:R,:Q,:val]]
        larr[x[1],x[2],x[3],x[4],x[5],x[6]]=x[7]
    end
    return larr
end
larr=readlarr();
macro ss_str(st)
    quote
        symbol($st)
    end
end


#cd("..")
#require("ts.jl")
#cd("../stochastic")
const Gk=readdlm("Gk.mat")
(Xf,Yf,Zf,Wf)=ldlebedev_n()
nps=(1:length(Xf))[Zf .> 0]
#const (X,Y,Z,W)=(Xf[nps],Yf[nps],Zf[nps],Wf[nps])
const (X,Y,Z,W)=(Xf,Yf,Zf,Wf)
const leb_X=[X Y Z]'
function theta(x,y,z,B)
    return 1/(4*pi*p'*B*p)^(1.5)
end

function get_a11()
   (x,y,z,w)=ld0110()
   p=[x',y',z']
   B=Utils.fisher_rot_mat(0.1)
end

function ol_bud(kappa,V,x)
    return (x'*V*diagm([kappa,0])*V'*x)[1]
end

function bingham_unnorm_density(kappa,V,x)
    d1=0.
    d2=0.
    @inbounds @simd for j=1:3
       d1 += V[j,1]*x[j]
       d2 += V[j,2]*x[j]
    end
    return exp(d1*d1*kappa[1] + d2*d2*kappa[2])
end

function bingham_density(kappa,V,x)
    Z=d2(kappa,V,leb_X)
    return bingham_unnorm_density(kappa,V,x)/Z
end

function bingham_expect(f,kappa,V=eye(3))
    res=zeros(size(f(leb_X[:,1])))
    n=d2(kappa,V,leb_X)
    for i=1:size(leb_X,2)
        res+=f(leb_X[:,i]).*bingham_density(kappa,V,leb_X[:,i]).*W[i]
    end
    return res
end

function d2(kappa,V=eye(3),locs=leb_X)
    res=0
    for i=1:size(locs,2)
        res+=W[i]*bingham_unnorm_density(kappa,V,locs[:,i])
    end
    return res
end

function grid_kappas()
    Gk=zeros(201,100)
    Mk=zeros(201,100)
    for i=0.005:0.005:0.5
        j=0.005
        while (1-2*i -j  >= -1e-6 && i-j >= -1e-6)
            idr=int(round(i*200))
            idc=int(round(j*200))
            Gk[2*idr-1:2*idr,idc]=ev_bingham_mle([j,i,0])
            Mk[2*idr-1:2*idr,idc]=[i,j]
            j+=0.005
        end
    end
    return (Gk,Mk)
end

#smallest is first!
function eigs2kappa(eigvs,Gk=Gk)
    eigvs[1]>0.5?eigvs[1]=1-eigvs[2]-eigvs[1]:nothing
    eigvs[2]>0.5?eigvs[2]=1-eigvs[2]-eigvs[1]:nothing
    idc=convert(Int,round(minimum(eigvs)*200))
    idr=convert(Int,round(maximum(eigvs)*200))
    idc==0?idc=1:nothing
    idr==0?idr=1:nothing
    Gk[2*idr-1:2*idr,idc]
end

function grad_kappa(eigvs,Gk=Gk)
    eigvs[1]>0.5?eigvs[1]=1-eigvs[2]-eigvs[1]:nothing
    eigvs[2]>0.5?eigvs[2]=1-eigvs[2]-eigvs[1]:nothing
    idcl=int(floor(minimum(eigvs)*200))
    idrl=int(floor(maximum(eigvs)*200))
    idcl==0?idcl=1:nothing
    idrl==0?idrl=1:nothing
    kappal=Gk[2*idrl-1:2*idrl,idcl]
    idch=int(ceil(minimum(eigvs)*200))
    idrh=int(ceil(maximum(eigvs)*200))
    idch==0?idch=1:nothing
    idrh==0?idrh=1:nothing
    kappah=Gk[2*idrh-1:2*idrh,idch]
    return kappah/0.005
end

function get_lookup_a4(Gk=Gk)
    larr=zeros(3,3,3,3,100,100)
    for i=1:100
        for j=1:i
            eigbig=i/200
            eigsm=j/200
            kappa=eigs2kappa([eigbig,eigsm,0])
            larr[:,:,:,:,i,j]=get_pc_a4(kappa)
            println(i," ",j)
        end
    end
    return larr
end

function write_larrdf(larr)
    larrdf=DataFrame(I=Int64[],J=Int64[],K=Int64[],L=Int64[],R=Int64[],Q=Int64[],val=Float64[])
    for r=1:size(larr,6), q=1:size(larr,6)
        for i=1:3,j=1:3,k=1:3,l=1:3
            a4val=larr[i,j,k,l,r,q]
            push!(larrdf,[i,j,k,l,r,q,a4val])
        end
    end
    writetable("larr.csv",larrdf)
   return larrdf
end

function ev_bingham_mle(evs,kappa0=[1.0,1])
    log2d(kappa)=log(d2(kappa))
    g=gradient(log2d)#,Float64,fadtype=:typed)
    function logdd2dk!(kappa,residual)
        residual[:]=g(kappa)[1:2] - evs[1:2]
    end
    residual=zeros(2)
    kappa=kappa0
    return nlsolve(logdd2dk!, kappa[1:2]).zero
end 

function bingham_mle(a2,kappa0)
    Evv=eigfact(a2)
    
    log2d(kappa)=log(d2(diagm(kappa), Evv[:vectors],leb_X))
    g=gradient(log2d)#,Float64,fadtype=:typed)
    function logdd2dk!(kappa,residual)
        residual[:]=g([kappa,0])[1:2] - Evv[:values][1:2]
    end

    residual=zeros(2)
    kappa=kappa0
    return nlsolve(logdd2dk!, kappa[1:2]).zero
end

#a2=p*p'/size(p,2)
#evecs=eigfact(a2)[:vectors]
#kappa0=[2.,2.]
#kappa=bingham_mle(a2,kappa0)
#Kappa=diagm([kappa,0])
#n=d2(Kappa,evecs,leb_X)
#kn(x1)=(bingham_unnorm_density(kappa,evecs,x1)/n)

function getvals(kn,locs)
   vals=Array(Float64,lebedev_n)
   for i=1:lebedev_n
       vals[i]=kn(locs[:,i])
   end
   return vals
end
#vals = getvals(kn,leb_X)
#a2xx=dot(X.*X.*vals,W)

macro unless(expr,exe)
    quote
        if $expr==false
            $exe
        end
    end
end

function pushall!(set,permu)
    for s in permu
        push!(set,s)
    end
end

function perm()
    unperms=Set()
    ins=false
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        tind=[i,j,k,l]
        setperms=Set()
        pushall!(setperms,permutations(tind))
        push!(unperms,setperms)
    end
    return unperms
end

const a4inds=perm()

function i2c(i)
    if i==1
        c=X
    elseif i==:2
        c=Y
    elseif i==:3
        c=Z
    end
    c
end

#function get_a4(vals)
#   A4=zeros(3,3,3,3)
#   for S in a4inds
#      pind=first(S)
#      tm = sum(leb_X[pind[1],:].*leb_X[pind[2],:].*leb_X[pind[3],:].*leb_X[pind[4],:].*W.*vals)/lebedev_n
#      for s2 in S
#            A4[s2[1],s2[2],s2[3],s2[4]] = tm
#            A4[s2[1],s2[2],s2[3],s2[4]] = mean(@i2c(s2[1]).*@i2c(s2[2]).*@i2c(s2[3]).*@i2c(s2[4]).*W.*vals)
#        end
#    end
#    return A4
#end
function get_a4(vals,locs)
    A4=zeros(3,3,3,3)
    for i=1:3,j=1:3,k=1:3,l=1:3
        A4[i,j,k,l]=sum(locs[:,i].*locs[:,j].*locs[:,k].*locs[:,l].*W.*vals)
    end
    return A4
end

function get_a2(vals,locs)
    A2=zeros(3,3)
    for i=1:3,j=1:3
        A2[i,j]=sum(locs[:,i].*locs[:,j].*W.*vals)
    end
    return A2
end


function get_pc_a2(kappa)
    leb_X2=[X Y Z]
    n=d2(kappa,eye(3),leb_X)
    knab(x1)=(bingham_unnorm_density(kappa,eye(3),x1)/n)
    vals=getvals(knab,leb_X)
    a4=get_a2(vals,leb_X2)
end


function get_pc_a4(kappa)
    leb_X2=[X Y Z]
    n=d2(kappa,eye(3),leb_X)
    knab(x1)=(bingham_unnorm_density(kappa,eye(3),x1)/n)
    vals=getvals(knab,leb_X)
    a4=get_a4(vals,leb_X2)
end




#const larr=get_lookup_a4()
function mul42(A,b)
    n=size(b,1)
    c=zeros(n,n)
    for i=1:n
        for j=1:n
            for k=1:n
                for l=1:n
                    c[i,j]+=A[i,j,k,l]*b[k,l]
                end
            end
        end
    end
    return c
end

function rk4(f::Function,n::Int64,x,vort_b,epsdot_b,dt)
   #vort=deepcopy(epsdot); vort[3,1:2]=0
   #vort[2,1]=0; vort=vort-vort'
   xstar=deepcopy(x)
   for i=1:n
      k1=f(x,vort_b,epsdot_b,dt)
      k2=f(x+k1*dt/2,vort_b,epsdot_b,dt)
      k3=f(x+k2*dt/2,vort_b,epsdot_b,dt)
      k4=f(x+k3*dt/2,vort_b,epsdot_b,dt)
      xstar+=(1/6)*dt*(k1+2*k2+2*k3+k4)
      end
   return xstar
   end

function rot_a4(v,a4)
    a4r=zeros(3,3,3,3)
    for i=1:3
        for j=1:3
            for k=1:3
                for l=1:3
                    for a=1:3
                        for b=1:3
                            for c=1:3
                                @simd for d=1:3
                                    @inbounds a4r[i,j,k,l] += v[a,i]*v[b,j]*v[c,k]*v[d,l] * a4[a,b,c,d]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return a4r
end
#don't supply vert comp on dvincomp
function a2rk4(a2v,vv,dvincomp,dt=1.0)
    a2=zeros(3,3)
    spin=zeros(3,3)
    spin[2,3]=vv[1]
    spin[1,3]=vv[2]
    spin[1,2]=vv[3]
    spin[3,2]=-vv[1]
    spin[3,1]=-vv[2]

    spin[2,1]=-vv[3]
    
    epsdot=zeros(3,3)
    epsdot[1,1]=dvincomp[1]
    epsdot[2,2]=dvincomp[2]
    epsdot[3,3]=-dvincomp[1]-dvincomp[2]
    epsdot[3,2]=dvincomp[3]
    epsdot[2,3]=dvincomp[3]
    epsdot[1,3]=dvincomp[4]
    epsdot[3,1]=dvincomp[4]
    epsdot[1,2]=dvincomp[5]
    epsdot[2,1]=dvincomp[5]
    a2[1,1]=a2v[1]
    a2[2,2]=a2v[2]
    a2[3,3]=1-a2v[2]-a2v[1]
    a2[2,3]=a2v[3]
    a2[1,3]=a2v[4]
    a2[1,2]=a2v[5]
    a2[3,2]=a2v[3]
    a2[3,1]=a2v[4]
    a2[2,1]=a2v[5]
#    eigvv=eigfact(a2)
#    lam=eigvv[:values][1:2]
#    inds=round(Int,lam*200+1)
#    global inds411=inds
#    if inds[1]<=0;inds[1]=1;end
#    if inds[2]<=0;inds[2]=1;end
#    if inds[1]>=100;inds[1]=100;end
#    if inds[2]>=100;inds[2]=100;end
 
#    v=eigvv[:vectors]
    #kappa=eigs2kappa(lam)
    #norm_const=d2(kappa)
    #kn(x1)=(bingham_unnorm_density(kappa,v,x1)/norm_const)
    #vals=getvals(kn,leb_X)
    #a4=get_a4(vals)
#    a4=deepcopy(larr[:,:,:,:,maximum(inds),minimum(inds)])
#    a4=rot_a4(v',a4)
    #dA=(spin*a2 -a2*spin + epsdot*a2 + a2*epsdot - 2*mul42(a4,epsdot)
    #  - 5e-1*(1-3/2*(eigvv[:values][3]-1/3))^2*sqrt(abs(Utils.secondInv(epsdot)))*(3*a2-eye(3)))
    #global dA411=dA
#   dA=(spin*a2 - a2*spin) + epsdot*a2 + a2*epsdot - 
#       2*trace(a2*epsdot)*a2 - 0.0*sqrt(abs(Utils.secondInv(epsdot)))*(3*a2-eye(3))
   dA=(spin*a2 - a2*spin) + epsdot*a2 + a2*epsdot - 
       2*a2*a2 - 0.02*sqrt(abs(Utils.secondInv(epsdot)))*(3*a2-eye(3))

    return [dA[1,1],dA[2,2],dA[2,3],dA[1,3],dA[1,2]]
end
neu=2
jeff_deterministic(a2vs) = a2rk4(a2vs,dvincomp,vv)
function jeff_stochastic(a2vs,gwn)
    return a2rk4(a2vs,gwn[1:6],vv)
end
function srk4(x, t, h, q, fi, gi,m=5,sd=0.05)
  a21 =   2.71644396264860;
  a31 = - 6.95653259006152;
  a32 =   0.78313689457981;
  a41 =   0.0;
  a42 =   0.48257353309214;
  a43 =   0.26171080165848
  a51 =   0.47012396888046;
  a52 =   0.36597075368373;
  a53 =   0.08906615686702;
  a54 =   0.07483912056879;

  q1 =   2.12709852335625;
  q2 =   2.73245878238737;
  q3 =  11.22760917474960;
  q4 =  13.36199560336697;
  n=length(x)
  t1 = t;
  x1 = x;
  w1 = rand(Distributions.Normal(),m) * sqrt(q1 * q / h );
#  k1 = h * fi(x1) + h * gi( x1 ) .* w1;
  k1 = h * fi(x1) + h * gi(x1, w1 );
  t2 = t1 + a21 * h;
  x2 = x1 + a21 * k1;
  w2 = rand(Distributions.Normal(),m) * sqrt(q2 * q / h );
#  k2 = h * fi(x2) + h * gi ( x2 ) .* w2;
  k2 = h * fi(x2) + h * gi(x2, w2)
  t3 = t1 + a31 * h  + a32 * h;
  x3 = x1 + a31 * k1 + a32 * k2;
  w3 = rand(Distributions.Normal(),m) * sqrt(q3 * q / h );
  k3 = h * fi(x3) + h * gi(x2, w3)
#  k3 = h * fi ( x3 ) + h * gi ( x3 ) .* w3;
  t4 = t1 + a41 * h  + a42 * h + a43 * h;
  x4 = x1 + a41 * k1 + a42 * k2;
  w4 = rand(Distributions.Normal(),m) * sqrt(q4 * q / h );
#  k4 = h * fi ( x4 ) + h * gi ( x4 ) .* w4;
  k4 = h * fi(x4) + h * gi(x2, w4)
  xstar = x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4;
  return xstar;
end

function get_stoch_d(niter)
    z=zeros(5)
    for i=2:10000
       z=srk4(z,0.,1e-4,1,x->-10*x,(x,y)->1e-3*y)
    end
    return get_stoch_d(niter,z)
end 

function get_stoch_d(niter,z0=zeros(5),regress=10,innovation=1e-3)
   z=zeros(5,niter)
   z[:,1]=z0
   for i=2:niter
#       z[:,i]=srk4(z[:,i-1],0.,1e-4,1,x->-10*x,(x,y)->1e-3*y)
       z[:,i]=srk4(z[:,i-1],0.,1e-4,1,x->-regress*x,(x,y)->innovation*y)
#       z[:,i]=srk4(z[:,i-1],0.,1e-3,1,x->-x,(x,y)->0.5*y)

   end
   return z
end

function get_stoch_jeff(niter=1000,x0=[1/3,1/3,0,0,0], 
                        vv=[1.0,.0,0],dvincomp=[-0.1,-0.1,-1.0,.0,0.],timestep=1e-5,frac=2e-1)
   xi=zeros(5,niter)
   z=get_rand_fun(niter*2)'[:,niter:end]
   xi[:,1]=x0
   for i=2:niter
       xi[:,i]=rk4(a2rk4,1,xi[:,i-1], timestep*frac*z[6:end,i],timestep*frac*z[1:5,i],1)
#      xi[:,i]=rk4(a2rk4,1,xi[:,i-1],timestep*vv + timestep*frac*z[3:5,i],timestep*dvincomp + timestep*frac*z[1:5,i],1)
#      xi[:,i]=rk4(a2rk4,1,xi[:i-1],vv,
#                  (dvincomp+ 1e-2*rand(Distributions.Normal(0,1),5)),1)
   global last411=xi[:,i]
   end
   return xi
end

function compare_diffs(niter=50000)
    xi=zeros(5,niter,nt)
    dxi=DataFrame()
    dxi[:trial]=0.
    dxi[ss"Total shear strain"]=0.
    dxi[ss"A<sub>11</sub>"]=0.
    dxi[ss"A<sub>22</sub>"]=0.

    dxi[ss"A<sub>33</sub>"]=0.

    for i=1:nt
        #xi[:,:,i]=get_stoch_jeff(niter)
        xi[:,:,i]=get_stoch_jeff(niter)
        for j=1:niter
            push!(dxi,[i, j*1e-3, xi[1,j,i], xi[2,j,i],1-xi[1,j,i]-xi[2,j,i]])
        end
    end
end 

function get_dj_jeff(niter=5000,dt=1.0,x0=[1/3,1/3,0,0,0],current_depth=0.0,
        tot_depth=1.0,surface_vel=1e-2,kink=0.7,regress=1,innovation=1e3,a=1e-9)
#niter=5000;x0=[1/3,1/30,0,0];current_depth=0.0;
#        tot_depth=1.0;surface_vel=1e-2;kink=0.7;regress=10.0;innovation=1e-3;a=1e-4

    dvincomp=zeros(5)
    vv=zeros(3)
    z=get_stoch_d(niter*2,zeros(5),regress,innovation)[:,niter:end]
    z/=sqrt(mean(z.^2))
    xi = zeros(5,niter)
    xi[:,1]=x0
    depths=zeros(niter)
    for i=2:niter
        (vert_strain,horiz_strain) = get_dj_strain(current_depth,a,tot_depth,
        surface_vel, kink)
        vert_strain=a
        vv[1]=horiz_strain
        dvincomp=[0, vert_strain, -horiz_strain,0,0]
        dnorm=norm(dvincomp[3:5,:])
        current_depth-=dj_vel(current_depth,a,tot_depth,surface_vel,kink)*1e5
        depths[i]=current_depth
        xi[:,i]=rk4(a2rk4,1,xi[:,i-1],vv + dnorm*1e-9*z[3:5,i]*dnorm,dvincomp + dnorm*1e-9*z[1:5,i],dt)
    end
    return xi,depths
end

function dj_vel(depth,a,tot_depth,surface_vel,kink)
    kink_depth=tot_depth*kink
    tau = -a/(0.5+0.5*kink)
    if depth>= kink_depth
        return 0.5*(1.0-depth)*tau
    else
        return 0.5*(1-kink)*tau + (kink-depth)*tau
    end
end

function get_dj_strain(depth,a=1e-4,tot_depth=1.0,surface_vel=1e-2, kink=0.7)
    uz = a/tot_depth
    area = kink*tot_depth + 0.5*(1.0-kink)*tot_depth
    kink_depth = kink*tot_depth
    if depth < kink_depth
        vert_strain = uz/area
    else
        vert_strain = uz*(tot_depth - depth)/(tot_depth - kink_depth)/area
    end
    horiz_strain = surface_vel*((depth/tot_depth).^3)/4
    return [vert_strain,horiz_strain]
end

#main function to solve multiple Monte Carlo realizations
#of jefferys equation.
function mc_jeff(nt=10,niter=5000,solver=get_stoch_jeff)
    xi=zeros(5,niter,nt)
    dxi=DataFrame()
    dxi[:trial]=0.
    dxi[ss"Total shear strain"]=0.
    dxi[ss"A<sub>11</sub>"]=0.
    dxi[ss"A<sub>22</sub>"]=0.
    dxi[ss"A<sub>33</sub>"]=0.
    for i=1:nt
        #xi[:,:,i]=get_stoch_jeff(niter)
       xi[:,:,i]=solver(niter)
        for j=1:niter
            push!(dxi,[i, j*1e-2, xi[1,j,i], xi[2,j,i],1-xi[1,j,i]-xi[2,j,i]])
        end
    end
#    plot(dxi, x="Total shear strain", y="A<sub>11</sub>",color="trial",Geom.line)
    return dxi
end




function plot_jeff(mc)
    quants = by(mc[2:end,:], ss"Total shear strain") do df
        quants = DataFrame()
        quants[ss"2.5% quantile"] = quantile(df[ss"A<sub>11</sub>"],0.05)
        quants[ss"median"] = quantile(df[ss"A<sub>11</sub>"],0.5)
        quants[ss"97.5% quantile"] = quantile(df[ss"A<sub>11</sub>"],0.95)
        return quants
    end
    quants[ss"Total shear strain"] = log(1+quants[ss"Total shear strain"])
    pl=plot(quants, x="Total shear strain", y="median",
        ymin="2.5% quantile",ymax="97.5% quantile",
        Geom.line,Geom.ribbon, Guide.xlabel("Total shear strain"),
        Guide.ylabel("A<sub>33</sub>"))
    return (quants,pl)
end
