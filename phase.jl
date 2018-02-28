using Gadfly
using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport matplotlib.pyplot as plt
@pyimport phase2

D_0 = zeros(3,3);D_0[1,3]=1.;D_0[3,1]=1
V_0 = zeros(3,3);V_0[1,3]=1.;V_0[3,1]=-1

function adot(A,D=D_0,V=V_0)
    return V*A-A*V-D*A-A*D + 2*A*sum(A.*D)
end

function ratiodot(ratio,A13,tot_radius=0.1,D=D_0,V=V_0)
    A22 = tot_radius - ratio# 1
    A11 = ratio
    #A11 *= tot_radius/(1+ratio)
    #A22 *= tot_radius/(1+ratio)
    A=diagm([A11,A22,1-A11-A22])
    A[1,3]=A13
    A[3,1]=A13
    return adot(A,D,V)
end

function plot_ratio()
    ratios = collect(logspace(0.0001,2.,100))
    A13s = collect(linspace(-.2,0,100))
    plot(x=ratios,y=map(z->norm(eigvals(ratiodot(z,-0.01))), A13s ))
end

function plt_contour()
    A22s = collect(linspace(0.0001,0.1,100))
    A13s = reverse(collect(linspace(-.2,0,100)))
    x=[a22 for a22 in A22s, a13 in A13s]
    y=[a13 for a22 in A22s, a13 in A13s]
    z=[sum(abs(ratiodot(a22,a13))) for a22 in A22s, a13 in A13s]
    levels = linspace(0,maximum(z),100)
    #plt.contourf(x,y,z,levels)
    #plt.xlabel(r"||\dot{\mathbf{A}}||")
   #plt.colorbar()
#    return phase2.conplot(x,y,z, levels)
    return phase2.conplot(x,y,z,levels)#plt.contourf(x,y,z)
end
#plt.contour()
