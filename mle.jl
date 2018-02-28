include("lebedev.jl")
include("../Utils.jl")
using PyCall,Optim,Gadfly

@pyimport matplotlib.pyplot as plt
@pyimport matplotlib.font_manager as fm
include("../newts.jl")
plt.style[:use]("seaborn-paper")
function dinh(p,bs)
    b3=1/(bs[1]*bs[2])
    return 1/(sqrt(bs[1]*bs[2]*b3)*(bs[1]*p[1]*p[1] + bs[2]*p[2]*p[2]
              +b3*p[3]*p[3])^1.5)
end

function tsdf2arr(ts)
    ngr=size(ts)
    ps=[ts[:c1] ts[:c2] ts[:c3]]
    area=ts[:area]
    return (ps,area)
end

function fisher(p, kappa)
    if abs(kappa) - 0 < 1e-6
        return 1/(4*pi)
    else
        return 1.0/(4*pi)*kappa*exp(kappa*p[3])/(exp(kappa)-1.0)
    end
end

function fisher_mle(p,areas=false)
    ps = p'
    println("shape")
    println(size(ps))
	V=eigfact!(ps * ps')
    println(size(V[:vectors]))
    lams = V[:values]
    V=V[:vectors]
    #swap for girdle
    if lams[2] > 1/3
        V[:,1],V[:,3]=V[:,3],V[:,1]
    end
    q=V'*ps
    ng=size(ps,2)
    if areas == false
        areas=ones(ng)/ng
    end
     function fisher_obj(kappa)
        log_p=0.0
#        if (any(bs.<-1)); return Inf;end
        for i=1:ng
            if q[3,i]<0
                q[:,i] *= -1
            end
            log_p += log(fisher(q[:,i],kappa[1]))*areas[i]
        end
        return -log_p# + kappa[2]^2
    end
    res = optimize(fisher_obj, [-100.,0.])
	println(res)
    kappa_opt = res.minimizer
    println(kappa_opt)
    return (res.minimum,-res.minimizer[1])
end

function dinh_mle_B(ps,areas=false,ret_b=false)
    V=eigfact!(ps*ps')[:vectors]
    q=V'*ps
    ng=size(ps,2)
    if areas == false
        areas=ones(ng)/ng
    end
    function dinh_obj(bs)
        log_p=0.0
#        if (any(bs.<-1)); return Inf;end
        if (any(bs.<0)); return Inf;end
        for i=1:ng
            log_p -= log(dinh(q[:,i],bs))*areas[i]
        end
        return log_p
    end

    res=optimize(dinh_obj, [100.0,100])
    bs=res.minimizer
    B=diagm([bs[1],bs[2],1/(bs[1]*bs[2])])
    B=V*B*V'
    if ret_b
        return (bs,-res.minimum)
    else
        return (B,-res.minimum)
    end
end

function get_a2_area(ps,area)
    ng=size(ps,2)
    a2=zeros(3,3)
    for i=1:ng
        a2+=ps[:,i]*ps[:,i]'*area[i]
    end
    a2
end

function ps_bingham_mle(ps,areas=false)
    ng=size(ps,2)
    if areas == false
        areas=ones(ng)/ng
    end
    a2=get_a2_area(ps,areas)
    V=eigfact!(a2)[:vectors]
    q=V'*ps     
    function bingham_obj(kappa)
       log_l=0
       norm_c=d2(kappa,eye(3),leb_X)*4*pi
       for i=1:ng
           log_l -= log(bingham_unnorm_density(kappa,eye(3),q[:,i])/norm_c)*areas[i]
       end
       return log_l
   end
   optimize(bingham_obj,[0.0,-0.0])
end

function get_mles(fabric::DataFrame)
    nts=length(unique(fabric[:,:Depth]))
    ml_bing=zeros(nts)
    ml_dinh=zeros(nts)
    ml_fisher=zeros(nts)
    for i=1:nts
        pd=Array(fabric[fabric[:Depth].==dr[i],2:4])'
        area=Array(fabric[fabric[:Depth].==dr[i],:area])
        ml_bing[i]=ps_bingham_mle(pd,area).minimum
        ml_dinh[i]=dinh_mle_B(pd,area)[2]
        ml_fisher[i]=fisher_mle(pd',area)[1]
        println("iter $i done")
        #println("bing $ml_bing[i], dinh $ml_dinh[i], fish $ml_fisher[i]")
    end
    return (ml_bing,ml_dinh,ml_fisher)
end

function get_B(fabric)
    nts=length(unique(fabric[:,:Depth]))
    ml_dinh=Dict()
    for i=1:nts
        pd=Array(fabric[fabric[:Depth].==dr[i],2:4])'
        area=Array(fabric[fabric[:Depth].==dr[i],:area])
        ml_dinh[i]=dinh_mle_B(pd,area)[2]
    end
    return ml_dinh
end 

function get_mles(fabric::Dict)
    n=length(fabric)
    ml_bing=zeros(n)
    ml_dinh=zeros(n)
    ml_fisher=zeros(n)
    for i=1:n
        ml_bing[i]=ps_bingham_mle(fabric[i]).minimum
        ml_dinh[i]=dinh_mle_B(fabric[i])[2]
        ml_fisher[i]=fisher_mle(fabric[i]')[1]
    end
    return (ml_bing,ml_dinh,ml_fisher)
end

function plot_ll_diff(fabric_wais,fabic_siple,dr,dr_siple,img_file)
    (ml_bing,ml_dinh)=get_mles(fabric_wais)
    (siple_bing,siple_dinh)=get_mles(fabric_siple)
    #dr_siple=get_depths()    
    lls=DataFrame(ll=[-ml_bing-ml_dinh;-siple_bing-siple_dinh],
                  depth=[dr;dr_siple]/1000,core=[repeat("WAIS Divide",length(ml_bing)),repeat("Siple Dome",length(siple_bing))])
    names!(lls,[symbol("Log likelihood difference"),symbol("Depth (km)"),symbol("Core")])
#    img=
    #plot(lls,x=symbol("Depth (km)"),y=symbol("Log likelihood difference"),color=:Core,Theme(default_color=color("black")))
    #(fig,ax)=plt.subplots()
    plt.scatter(dr_siple, -siple_bing - siple_dinh,c="k",marker="x", label="Siple Dome")
    plt.scatter(dr, -ml_bing - ml_dinh,c="k",label="WAIS divide")
    plt.xlim([0.0,maximum(dr)+100])
    plt.xlabel("Depth (m)",fontsize=16)
    plt.ylabel("Average log likelihood difference",fontsize=16)
    plt.legend()
    plt.tight_layout()
    #    draw(SVG(img_file,20cm,7.5cm),img)
    plt.show()
end

function plot_ll(fabric_wais,fabic_siple,dr,dr_siple,img_file)
    (ml_bing,ml_dinh,ml_fisher)=get_mles(fabric_wais)
    (siple_bing,siple_dinh,siple_fisher)=get_mles(fabric_siple)
    #dr_siple=get_depths()    
#    lls=DataFrame(ll=[-ml_bing-ml_dinh;-siple_bing-siple_dinh],
#                  depth=[dr;dr_siple]/1000,core=[rep("WAIS Divide",length(ml_bing)),rep("Siple Dome",length(siple_bing))])
#    names!(lls,[symbol("Log likelihood difference"),symbol("Depth (km)"),symbol("Core")])
#    img=
    #plot(lls,x=symbol("Depth (km)"),y=symbol("Log likelihood difference"),color=:Core,Theme(default_color=color("black")))
    fig = plt.figure()
    ax0=fig[:add_subplot](211)
    ax1=fig[:add_subplot](212)

#    fontP = fm.FontProperties()
#    fontP[:set_size]("small")
    ax0[:scatter](dr, -ml_bing,c="#1b9e77",label="Bingham",marker="^",lw=0,s=13)
    ax0[:scatter](dr, ml_dinh,c="#d95f02",label="Dinh-Armstrong",marker="*",lw=0,s=13)
    ax0[:scatter](dr, -ml_fisher,c="#7570b3",label="Fisherian",marker="o",lw=0,s=13)
    ax1[:scatter](dr_siple, -siple_bing,c="#1b9e77",label="Bingham",marker="^",lw=0,s=13)
    ax1[:scatter](dr_siple, siple_dinh,c="#d95f02",label="Dinh-Armstrong",marker="*",lw=0,s=13)
    ax1[:scatter](dr_siple, -siple_fisher,c="#7570b3",label="Fisherian",marker="o",lw=0,s=13)
    ax0[:set_xlim]([0.0,maximum(dr)+100])
    ax0[:set_xlabel]("Depth (m)",fontsize=12)
    ax0[:set_ylabel]("Average log likelihood",fontsize=12)
    ax0[:legend](fontsize=10,loc=2)#prop=fontP)
    ax0[:set_title]("WAIS Divide")
    ax0[:text](3400,0.92, "A",
          fontsize=16, fontweight="bold", va="top")

    ax1[:set_xlim]([0.0,maximum(dr_siple)+30])
    ax1[:set_xlabel]("Depth (m)",fontsize=12)
    ax1[:set_ylabel]("Average log likelihood",fontsize=12)
    ax1[:legend](fontsize=10,loc=2)#prop=fontP)
    fig[:set_size_inches](9,9.0)
    ax1[:set_title]("Siple Dome")
    ax1[:text](990,0.92, "B",
          fontsize=16, fontweight="bold", va="top")



    fig[:tight_layout](h_pad=1.2)
    #    draw(SVG(img_file,20cm,7.5cm),img)
    return (fig,ax0,ax1)
end

function plot_ll_wais(fabric_wais,fabic_siple,dr,dr_siple,img_file)
    (ml_bing,ml_dinh,ml_fisher)=get_mles(fabric_wais)
 #   (siple_bing,siple_dinh,siple_fisher)=get_mles(fabric_siple)
    #dr_siple=get_depths()    
#    lls=DataFrame(ll=[-ml_bing-ml_dinh;-siple_bing-siple_dinh],
#                  depth=[dr;dr_siple]/1000,core=[rep("WAIS Divide",length(ml_bing)),rep("Siple Dome",length(siple_bing))])
#    names!(lls,[symbol("Log likelihood difference"),symbol("Depth (km)"),symbol("Core")])
#    img=
    #plot(lls,x=symbol("Depth (km)"),y=symbol("Log likelihood difference"),color=:Core,Theme(default_color=color("black")))
    fig = plt.figure()
    ax=fig[:add_subplot](111)
#    fontP = fm.FontProperties()
#    fontP[:set_size]("small")
    ax[:scatter](dr, -ml_bing,c="#000000",label="Bingham",marker="^",lw=0,s=40)
    ax[:scatter](dr, ml_dinh,c="#000000",label="Dinh-Armstrong",marker="*",lw=0,s=40)
    ax[:scatter](dr, -ml_fisher,c="#000000",label="Fisherian",marker="o",lw=0,s=40)
#    ax[:scatter](dr_siple, -siple_bing,c="#4c509a",label="Bingham MLL at Siple D.",marker="^",lw=0,s=40)
#    ax[:scatter](dr_siple, siple_dinh,c="#d66a00",label="Dinh MLL at Siple D.",marker="^",lw=0,s=40)
#    ax[:scatter](dr_siple, -siple_fisher,c="green",label="Fisher MLL at Siple D.",marker="^",lw=0,s=40)

    ax[:set_xlim]([0.0,maximum(dr)+100])
    ax[:set_xlabel]("Depth (m)",fontsize=12)
    ax[:set_ylabel]("Average log likelihood",fontsize=12)
    ax[:legend](fontsize=10)#prop=fontP)
    fig[:tight_layout]()
    fig[:set_size_inches](9,5.0)
#    ax[:tight_layout]()
    #    draw(SVG(img_file,20cm,7.5cm),img)
    return (fig,ax)
end


function plot_ll_siple(fabric_wais,fabic_siple,dr,dr_siple,img_file)
#    (ml_bing,ml_dinh,ml_fisher)=get_mles(fabric_wais)
    (siple_bing,siple_dinh,siple_fisher)=get_mles(fabric_siple)
    #dr_siple=get_depths()    
#    lls=DataFrame(ll=[-ml_bing-ml_dinh;-siple_bing-siple_dinh],
#                  depth=[dr;dr_siple]/1000,core=[rep("WAIS Divide",length(ml_bing)),rep("Siple Dome",length(siple_bing))])
#    names!(lls,[symbol("Log likelihood difference"),symbol("Depth (km)"),symbol("Core")])
#    img=
    #plot(lls,x=symbol("Depth (km)"),y=symbol("Log likelihood difference"),color=:Core,Theme(default_color=color("black")))
    fig = plt.figure()
    ax=fig[:add_subplot](111)
#    fontP = fm.FontProperties()
#    fontP[:set_size]("small")
#    ax[:scatter](dr, -ml_bing,c="#4c509a",label="Bingham",marker="*",lw=0,s=40)
#    ax[:scatter](dr, ml_dinh,c="#d66a00",label="Dinh-Armstrong",marker="*",lw=0,s=40)
#    ax[:scatter](dr, -ml_fisher,c="green",label="Fisherian",marker="*",lw=0,s=40)
    ax[:scatter](dr_siple, -siple_bing,label="Bingham",marker="^",lw=0,s=40,c="#000000")
    ax[:scatter](dr_siple, siple_dinh,label="Dinh-Armstrong",marker="*",lw=0,s=40,c="000000")
    ax[:scatter](dr_siple, -siple_fisher,label="Fisherian",marker="o",lw=0,s=40,c="000000")

    ax[:set_xlim]([0.0,maximum(dr)+50])
    ax[:set_xlabel]("Depth (m)",fontsize=12)
    ax[:set_ylabel]("Average log likelihood",fontsize=12)
    ax[:legend](fontsize=10)#prop=fontP)
    fig[:tight_layout]()
    fig[:set_size_inches](9,5.0)
#    ax[:tight_layout]()
    #    draw(SVG(img_file,20cm,7.5cm),img)
    return (fig,ax)
end



import Utils.cddo
include("../parse_siple.jl")
fabric = readtable("../wais_fabric_v4.csv")
fabric_siple = read_siple("../siple") 
dr_siple = get_depths("../siple")
(ps,areas)=tsdf2arr(fabric)
dr=sort(unique(fabric[:Depth]))
#(mlb,mld)=plot_mles(fabric,dr)
#pl = plot_ll_diff(fabric,fabric_siple,dr,dr_siple,"mle_4-6.svg")
