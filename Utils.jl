module Utils
using Base.rand,Optim,Distributions
#@pyimport pyutils
import Base.size
#include("lsap/Assignment.jl")
export voigt2Tensor,tensor2Voigt,rk4!,rot2vert!
export halton,vdc,unifmesh,randir,minind
export diffrandi,secondInv,binBoolInd, rep_els,deviator,voigt2Tensor,tensor2Voigt,make_asm,flatten_to_2d
export cddo
#unshift!(PyVector(pyimport("sys")["path"]), "")

macro s_str(s)
    Expr(:quote, symbol(s))
end
#function get_perms(v,depth=1)
#    n=length(v)
#    u=deepcopy(v)
#    assert(n<7, "too long")
#    perms=Array(Any,factorial(v))
#
#    if n==1
#        perms=[v]; return;
#    for i=depth+1:n #swap first and i
#        u[depth]=v[i]
#        u[i]=v[depth]
#        get_perms(u,depth+1)
function rep_els(x, n)
    m=length(x)
    y=Array(Float64,n*m)
    for i=1:m
        y[(i-1)*n+1:i*n]=x[i]
    end
    return y
end

function minind(a)
    for i=1:length(a)
        if a[i]
            return i
        end
    end
    return 0
end

function cddo(fn,dir)
    cdir=pwd()
    cd(dir)
    try
       fn()
    catch
       cd(cdir)
    end
    cd(cdir)
end

#rotate v
function norm_by_first(x)
    m=size(x,1)
    n=length(x)/m
    y=deepcopy(x)
    for i=1:m
        y[:,i]/=norm(y[:,i])
    end
    return y
end

function fabric_eigs_pd(pd)
    x=Array(Float64,3,length(dr))
    for d=1:length(dr)
        i=int(dr[d])
        n=size(pd[i],2)
        a2=zeros(3,3)
        for j=1:n
            a2+=pd[i][:,j]*pd[i][:,j]'
        end
        x[:,d]=eigvals(a2)
        x[:,d]/=norm(x[:,d])
    end
    return x
end

#m: size of sample; n: num of resamples
function bootstrap_eigs(ps,f::Function)
    n=size(ps,2)
    for i=1:m
        lam+=fabric_eigs(p[rand(1:n,n)])
    end
    lam /= n;
end



    

function fabric_eigs(p::Array{Float64,4},site=1)
    m=size(p,3)
    feig=Array(Float64,3,m)
    for i=1:m
        feig[:,i]=fabric_eigs(p[:,:,i,site])
    end
    return feig
end
function fabric_eigs(p::Array{Float64,2})
    n=size(p,2)
    a2=zeros(3,3)
    for i=1:n
        a2+=p[:,i]*p[:,i]'
    end
    return eigvals(a2/2n)
end

function res_ss(sigma,p,n=3)
    tr=sigma*p
    return dot(tr,tr) - dot(tr,p)^2
end

function rss_vert_2d(p::Array{Float64,2})
    n=size(p,2)
    res=zeros(n)
    sigma=zeros(3,3)
    sigma[1,3]=1;sigma[3,1]=1
    for i=1:n
        res[i]=res_ss(sigma,p[:,i])
    end
    return res
end

function rss__mean_vert(p)
    #size(p)=(3,n,54,m)
    n=size(p)[2]
    m=size(p)[1]
    for i=1:m
        for j=1:53
            rssm[j,i]=mean(rss_vert_2d(p[:,:,j,i]))
        end
    end
    return rssm
end

function rss_core(p::Array{Float64,3})
    n,m=size(p)[2:3]
    rss=Array(Float64,n,m)
    for i=1:m
        rss[i]=rss_vert_2d(p[:,:,i])
    end
    return rss
end

function res_stress_tensor(p,sigma)
    rot_mat=rotate_mat_rodriguez(p)
    sr=rot_mat*sigma*rot_mat
    sr13=sr[1,3];sr23=sr[2,3]
    sr=zeros(3,3);sr[1,3]=sr13;
    sr[2,3]=sr23;sr+=sr'
    return rot_mat'*sr*rot_mat
end

function poly_rss_comp(ps,i,j)
    sigma=zeros(3,3)
    sigma[1,3]=1;sigma[2,3]=1;
    sigma+=sigma'
    n=size(ps)[2]
    res=0
    for j=1:n
        res+=res_stress_tensor(ps[:,k],sigma)[1,3]
    end
    return res
end

#function rss_girdle(p)
#    n=size

function rotate_mat_rodriguez(v)
    if v[3]>.99999
        return eye(3)
    end
    axis=cross(v,[0,0,1])
    axis/=norm(axis)
    theta=acos(v[3])
    K=zeros(3,3)
    K[1,2]=-axis[3];K[1,3]=axis[2];K[2,3]=-axis[1]
    K[2,1]=axis[3];K[3,1]=-axis[2];K[3,2]=axis[1]
    R=eye(3)
    R+=sin(theta)*K
    R+=(1-cos(theta))*K*K
    return R
end

function fisher_rot_mat(k)
    axis=rand(Distributions.Gaussian(0,1),3)
    axis/=norm(axis)
    theta=rand(Distributions.VonMises(0,k),1)
    c=cos(theta)
    s=sin(theta)
    x=axis[1];y=axis[2];z=axis[3]
    C=1-c
    R = [ x*x*C+c   x*y*C-z*s  x*z*C+y*s;
          y*x*C+z*s  y*y*C+c    y*z*C-x*s;
          z*x*C-y*s  z*y*C+x*s  z*z*C+c]
    return R
end

function rodriguez_rotate!(v,axis,theta)
   v=cos(theta)*v+sin(theta)*cross(v,axis)+(1-cos(theta))*dot(v,axis)*axis
   return v;
end
#rotate v towards w by angle
function rotate_towards!(v,w,theta)
    axis=cross(v,w)
    axis/=norm(axis)
    v=rodriguez_rotate!(v,axis,theta)
    return v
end
function hash2arr(hash::Dict)
    ka=Array(Any,length(keys(hash)))
    i=1
    for key in keys(hash)
        ka[i]=key
        i+=1
    end
    sort!(ka)
    arr=Array(Float64,length(ka),length(hash[ka[1]]))
    for i=1:length(ka)
        arr[:,i]=hash[ka[1]]
    end
end
function proj2UpHem!(p)
  n=size(p)[2]
  for i=1:n
    if p[3,i]<0.
        p[:,i]= -p[:,i]
      end
    end
  return p
  end


function getRandc(n)
  p=rand(Distributions.Normal(),(3,n))
  for i=1:n
    p[:,i]=p[:,i]/norm(p[:,i])
    end
    return p
  end

function deviator(M)
    m=size(M,1)
    M-trace(M)*eye(m)/m
end

function distMat(ps1,ps2)
    m=size(ps1,2)
    n=size(ps2,2)
    costs=Array(Float64,m,n)
    for i=1:m
        for j=1:n
            dp=abs(dot(ps1[:,i],ps2[:,j]))
            if dp <=1.
                costs[i,j]=acos(dp)
            elseif 1.<dp
                costs[i,j]=0.
            end
        end
    end
   return costs
end

#align fabric so that it is in the principal ref frame,
#with big eigenvalue vertical
function rot2vert!(ps)
    ev=eigfact(ps*ps')
    for i=1:size(ps,2)
        ps[:,i]=ev[:vectors]'*ps[:,i]
    end
    return true
end

#find the horizontal axis rotation with the optimal
#alignment between p1 and p2.
#Returns (theta,p2_rotated,distance)
function alignFabrics(p1,p2)
  svp1=svd(p1)[1][:,3]
  obj(theta)=alignFabricsObj(theta,svp1,p2)
  res=Optim.optimize(obj,[0.],method=:simulated_annealing)
  return(-res.minimum,res.f_minimum,rotp(-res.minimum[1],p2))
  end
function alignFabricsObj(thetab,svp1,p2)
  theta=thetab[1];
  p2_rot=rotp(theta,p2)
  tc=svd(p2_rot)[1][:,3]
  return acos(abs(dot(tc,svp1)))
  end

function rotp(theta,p)
  rotM=zeros(3,3);rotM[1,1]=cos(theta);rotM[2,2]=cos(theta);
  rotM[3,3]=1;rotM[1,2]=-sin(theta);rotM[2,1]=-rotM[1,2];
  p_rot=rotM*p;
  return p_rot
  end
#This uses the Munkres algorithm to find 
#the emd between ps1 and ps2, where psj[spatial coors,point index]
#Munkres assigns n1 workers (ps2) to n2 jobs (ps1)
#according to cost matrix.
#If size(ps1) != size(ps2) then one needs must deal with weights. Spse p1 bigger than p2. Then have
#wts wts1 and wts2, st sum(wts1)=sum(wts2);
#Th

#function earthMoversDist(ps1::Array{Float64,2},ps2::Array{Float64,2})
#  (dim,n)=size(ps1)
#  costs=Array(Float64,n,n)
#  for i=1:n
#    for j=1:n
#      dp=abs(dot(ps1[:,i],ps2[:,j]))
#      if dp <=1.
#        costs[i,j]=acos(dp)
#        elseif 1.<dp
#          costs[i,j]=0.
#        end
#      end
#  end
#  tc=assignment(costs)
#  return tc
#  end

#function emd2sm(p)
#  n=size(p)[2]
#  sm=repmat([0.,0.,1.]',n)'
#  return earthMoversDist(p,sm)
#  end
    
function assignDiffN(srtwts1,srtwts2)
  nwts1=Array(Float64,m) #New wts1 array. Some
  # of the mass from the largest m-n moved to the on
  #ones at the end m-n elements, corresponding
  #to extra pts 1:m-n
  n=length(srtwts1);m=length(srtwts2)
  sw1=sum(srtwts1);sw2=sum(srtwts2)
  assert(n<m); #sum of wts must also equal
  dn=m-n; 
  wts=[srtwts1,srtwts1[1:m-n+1]]/sw1*sw2
  end

#stuff for bootstrapping
function bootstrap_ps(ps,n=1000)
    m=length(ps)
    variance=0;bmean=0.
    for i=1:n
        inds=rand(1:m,m)
        stdev+=std(ps[inds])
        bmean+=mean(ps[inds])
    end
    return stdev/n,bmean/n
end
  
function filterZeros(x::Array{Float64,2},y::Array{Float64,1},ind::Int64)
  m,n=size(x)
  length(y)==n?nothing:error("Dimension mismatch")
  res=Array(Float64,0)
  ind==2?x=x':nothing
  for i=1:n
     y[i]==0?nothing:append!(res,x[:,i])
  end
  ind==2?x=x':nothing
  return reshape(res,(m,int(length(res)/m)))
  end
function rk4!(f::Function,h::Float64,n::Int64,x,vort,epsdot,m)
   for i=1:n
      k1=f(x,vort,epsdot)
      k2=f(x+k1*h/2,vort,epsdot)
      k3=f(x+k2*h/2,vort,epsdot)
      k4=f(x+k3*h/2,vort,epsdot)
      x+=(1/6)*h*(k1+2*k2+2*k3+k4)
      end
   return x
   end

function voigt2Tensor2(v)
  return [ v[1] v[6] v[5]
           v[6] v[2] v[4]
           v[5] v[4] v[3]]
  end

function tensor2Voigt2(v)
    return [v[1,1],v[2,2],v[3,3],v[2,3],v[1,3],v[1,2]]
end

function size(pd::Dict{Any,Any})
    k=0
    for j in pd
        k+=1
    end
    return k
end

function make_asym(v)
    asm=voigt2Tensor2(v)
    asm[2,1]*=-1
    asm[3,1]*=-1
    asm[3,2]*=-1
    return asm
end

function flatten_to_2d(A,index)
    si=size(A)
    order=length(si)
    return reshape(A,(si, int(prod(si[2:end])/si[1])))
end

#function voigt2Tensor(v)
#  x=zeros(3,3)
#  x[1,1]=v[1];x[1,2]=v[6];x[1,3]=v[5]
#  x[2,3]=v[4];x[2,2]=v[2];x[3,3]=v[3]
#  return x+x'
#  end
voigt2Tensor=voigt2Tensor2 
tensor2Voigt=tensor2Voigt2

#function tensor2Voigt(v)
#  x=zeros(6)
#  x[1]=v[1,1];x[2]=v[2,2];x[3]=v[3,3]
#  x[4]=v[2,3];x[5]=v[1,3];x[6]=v[1,2]
#  return x
#  end
function binBoolInd(x,fn,n)
  ind=1
  for i=2:n
    if fn(x[ind],x[i])==true
      ind=i
      end
    end
  return ind
  end

function vdc(n,base)
  x,denom=0.,1.
  while n>0
    x+=n%base/(denom*=base)
    n=floor(n/base)
    end
    return x   
  end
  
function halton(n::Int,dim::Int,skip::Int=1000)
  bases=[2,3,5,7,11,13,17,19,23,29]
  xs=zeros(dim,n+skip)
  for i=1:dim
    b=bases[i]
    for j=1:n+skip
      xs[(j-1)*dim+i]=vdc(j,b)
      end
    end
  return xs[:,skip:end]
  end

function unifmesh(x,y)
  m=length(x)
  n=length(y)
  r=Array(Float64,m*n,2)
  for i=1:n
    r[m*(i-1)+1:i*m,:]=[x fill(y[i],m)]
    end
    return r
  end

#Get the second invariant of a 3x3 symmetric tensor in Voigt notation.
function secondInv(x::Array{Float64,1})
  return x[1]*x[2]+x[1]*x[3]+x[2]*x[3]-x[4]^2-x[5]^2-x[6]^2
  end

secondInv(x::Array{Float64,2})=secondInv([x[1,1],x[2,2],x[3,3],x[2,3],x[1,3],x[1,2]])

function randir(dims,lo::Int64,hi::Int64)
  x=rand(Int64,dims)
  x=mod(x,hi-lo+1)+lo
  end

#creates an array of random Ints in range [lo,hi] inclusive. 
function randi(len::Int,lo::Int,hi::Int)
  x=mod(rand(Int64,len),hi-lo+1)+lo
  end
#scalar version
function randi(lo::Int,hi::Int)
  x=mod(rand(Int64),hi-lo+1)+lo
  end
#choose an Int in the range lo high but not this.
function diffrandi(this,lo,hi)
  x=this
  while x==this
    x=randi(lo,hi)
    end
  return x
  end
function randir(lo::Int64,hi::Int64)
  x=rand(Int64)
  x=mod(x,hi-lo+1)+lo
  end

function diffrandi(this,lo,hi)
  x=this
  while x==this
    x=randir(lo,hi)
    end
  return x
  end

end #module
