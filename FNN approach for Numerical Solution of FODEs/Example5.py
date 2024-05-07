import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
import scipy.special as sp

def adfibo(alpha, n, x):
    s = 0*x
    for i in range(int(np.ceil(alpha)),n+1):
        if (i+n)%2==0:
            s=s+0
        else:
            s=s+(sp.gamma((n+i+1)/2))/(sp.gamma(i-alpha+1)*sp.gamma((n-i+1)/2))*x**(i-alpha)
    return s

def Input(x,degree,alpha):
    inp_x = []
    for i in range(1,degree+1):
        inp_x.append(adfibo(alpha,i,x))
    return np.array([inp_x])

def Neural_Network(input,degree,alpha,weights):
    z=[]
    z.append(Input(input, degree, alpha))
    z.append(np.dot(z[0],weights.T))
    return z[1], z[0]

def function(x):
    return 2+4*np.sqrt(x/np.pi)+x**2

def prob(input,degree,weights):
    tri_w = np.zeros_like(weights)
    h = np.zeros((len(weights[0]),len(weights[0])))
    error1  = 0
    error2  = 0
    z_int, dw_int = Neural_Network(0,degree,0,weights)
    z_int2, dw_int2 = Neural_Network(0,degree,1,weights)
    for l in input:
        z, dw = Neural_Network(l,degree,0,weights)
        z_alp, dw_alp = Neural_Network(l,degree,1.5,weights)
        z_1, dw_1 = Neural_Network(l,degree,2,weights)
        
        tri_w =tri_w +2*(z_1+z_alp+z-function(l))*(dw_1+dw_alp+dw)
        
        for i in range(len(weights[0])):
                 for j in range(len(weights[0])):
                     h[i,j]=h[i,j]+2*(dw_1[0,i]+dw_alp[0,i]+dw[0,i])*(dw_1[0,j]+dw_alp[0,j]+dw[0,j])
    
        error1=error1+(z_1+z_alp+z-function(l))**2
        error2=error2+(z_1+z_alp+z-function(l))**2
        
    tri_w = tri_w/(2*len(input)) + (z_int)*dw_int + (z_int2)*dw_int2
    for i in range(len(weights[0])):
                 for j in range(len(weights[0])):
                      h[i,j]=h[i,j]/(2*len(input))+ dw_int[0,i]*dw_int[0,j] + dw_int2[0,i]*dw_int2[0,j]
                     
    error1=error1/(2*len(input)) +1/2*(z_int)**2+1/2*(z_int2)**2
        
    return error1, error2/(2*len(input)),tri_w,h


def feed_farward(input, degree,accuracy,iter):
    w = r.random_sample((1,degree))
    k=0
    lem = 10**4
    while k<iter:
        error1,error2, tri_w, h,  = prob(input,degree,w)
        
        if error1<accuracy:
            break
        else:
            w_next = w.T - np.dot(np.linalg.inv(h+lem*np.identity(len(h))),tri_w.T)
            error11,error22,_,_ = prob(input,degree,w_next.T)
            
            if error11<error1:
                w = w_next.T
                k= k+1
                lem = lem/4
                print(f"Number of iteration is {k} and error is {error11}") 
            else:
                lem = 2*lem
                print(f"lemda is increasing {lem}") 
            
    return w, error11, error22
       
    
input = np.linspace(0,1,11)

degree = 4

w , error1, error2 = feed_farward(input,degree,10**(-31),500)  

def Nsol(Input,degree, weights):
    z1, z2 = Neural_Network(Input,degree,0,weights)
    return z1[0,0]

def Exsol(t):
    return t**2

exacSol = Exsol(input)

NumSol = []

for i in input:
    NumSol.append(Nsol(i,degree,w))

NumSol=np.array(NumSol)
    

error=abs(NumSol-exacSol)

plt.plot(input,NumSol, 'or',label="Numerical Solution")
plt.plot(input, exacSol, '-b',label="Exact Solution")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),ncol=5, fancybox=True)    
#plt.savefig('line_plot.pdf',bbox_inches='tight')
plt.show()

plt.plot(input, error)
plt.xlabel("x")
plt.ylabel(" Absolute Error")
#plt.savefig('pic3.pdf',bbox_inches='tight')  
plt.show()
