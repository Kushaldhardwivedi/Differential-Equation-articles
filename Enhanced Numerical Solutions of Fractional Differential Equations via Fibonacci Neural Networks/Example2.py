import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
import scipy.special as sp
np.random.seed(42)

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
    return x**3+6*x+(3.2)/(sp.gamma(0.5))*x**(2.5)

def prob(input,degree,weights):
    tri_w = np.zeros_like(weights)
    h = np.zeros((len(weights[0]),len(weights[0])))
    error1  = 0
    error2  = 0
    z_int, dw_int = Neural_Network(0,degree,0,weights)
    z_int2, dw_int2 = Neural_Network(0,degree,1,weights)
    for l in input:
        z, dw = Neural_Network(l,degree,0,weights)
        z_alp, dw_alp = Neural_Network(l,degree,1/2,weights)
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
                if lem>10**(20):
                    print("Lemda is too large, stopping the process")
                    break
            
    return w, error11, error22
       
    
#input = np.linspace(0,1,11)

input = np.random.uniform(0, 1, 10)

degree = 4

w , error1, error2 = feed_farward(input,degree,10**(-31),500)  

def Nsol(Input,degree, weights):
    z1, z2 = Neural_Network(Input,degree,0,weights)
    return z1[0,0]

def Exsol(t):
    return t**3

input_plot = np.linspace(0, 1, 101)

exacSol = Exsol(input_plot)

NumSol = []

for i in input_plot:
    NumSol.append(Nsol(i,degree,w))

NumSol=np.array(NumSol)
    

error=abs(NumSol-exacSol)

relative_error = [error[i]/exacSol[i] if exacSol[i] != 0 else 0 for i in range(len(input_plot))]


plt.plot(input_plot, exacSol, '-b',label="Exact Solution")
plt.plot(input_plot,NumSol, linestyle="--",color='red', label="Numerical Solution")
plt.xlabel("x")
plt.ylabel("y(x)")
#plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),ncol=5, fancybox=True) 
plt.legend()   
plt.savefig('line_plot.png')
plt.show()

plt.plot(input_plot, error)
plt.xlabel("x")
plt.ylabel(" Absolute Error")
plt.savefig('pic3.png',bbox_inches='tight', dpi=300)  
plt.show()

plt.plot(input_plot, relative_error)
plt.xlabel("x")
plt.ylabel(" Relative Error")
plt.savefig('pic4.png')  
plt.show()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('"x"')
ax1.set_ylabel('Relative Error', color=color, )
ax1.plot(input_plot, relative_error, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Absolute Error', color=color)  # we already handled the x-label with ax1
ax2.plot(input_plot, error, color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.savefig('error.png')
plt.show()

print("")
print( np.array([error[j] for j in range(0,101,10)])) 
print(" ")