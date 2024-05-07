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



def f(t,alp):
    return 1-4*t+5*t**2-(4*t**(1-alp))/(sp.gamma(2-alp))+(10*t**(2-alp))/(sp.gamma(3-alp))


def prob(input,degree,weights):
    tri_w = np.zeros_like(weights)
    h = np.zeros((len(weights[0]),len(weights[0])))
    error1  = 0
    error2  = 0
    z_int, dw_int = Neural_Network(0,degree,0,weights)
    for l in input:
        z_1, dw_1 = Neural_Network(l,degree,0.75,weights)
        z, dw = Neural_Network(l,degree,0,weights)
        
        tri_w =tri_w +2*(z_1 + z -f(l,0.75))*(dw_1 + dw )
        
        for i in range(len(weights[0])):
                 for j in range(len(weights[0])):
                     h[i,j]=h[i,j]+2*(dw_1[0,i] + dw[0,i] )*(dw_1[0,j] + dw[0,j] )
    
        error1=error1+(z_1 +z -f(l,0.75))**2
        error2=error2+(z_1 +z -f(l,0.75))**2
        
    tri_w = tri_w/(2*len(input)) + (z_int-1)*dw_int
    for i in range(len(weights[0])):
                 for j in range(len(weights[0])):
                      h[i,j]=h[i,j]/(2*len(input))+ dw_int[0,i]*dw_int[0,j]
                     
    error1=error1/(2*len(input)) +1/2*(z_int-1)**2
        
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
                lem = lem/2
                print(f"Number of iteration is {k} and error is {error11}") 
            else:
                lem = 2*lem
                print(f"lemda is increasing {lem}") 
            
    return w, error11, error22
       
    
input = np.linspace(0,1,11)

degree = 3

w , error1, error2 = feed_farward(input,degree,10**(-27),1000)  

def Nsol(Input,degree, weights):
    z1, z2 = Neural_Network(Input,degree,0,weights)
    return z1[0,0]

def Exsol(t):
    return 1-4*t+5*t**2

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





#for alpha =0.25
err1=np.array([8.43769499e-15, 1.22124533e-15, 8.32667268e-15, 1.33781874e-14,
       1.69864123e-14, 1.59872116e-14, 1.26565425e-14, 8.88178420e-15,
       1.33226763e-15, 9.32587341e-15, 2.13162821e-14])

#for alpha =0.5
err2=np.array([4.44089210e-16, 5.55111512e-16, 5.55111512e-16, 5.55111512e-17,
       9.99200722e-16, 0.00000000e+00, 6.66133815e-16, 0.00000000e+00,
       4.44089210e-16, 1.33226763e-15, 8.88178420e-16])

#for alpha =0.75
err3=np.array([2.22044605e-15, 3.33066907e-16, 2.10942375e-15, 3.60822483e-15,
       4.55191440e-15, 4.44089210e-15, 3.77475828e-15, 3.55271368e-15,
       2.22044605e-15, 1.33226763e-15, 1.77635684e-15])

#for alpha =1
err4=np.array([3.10862447e-15, 3.33066907e-16, 2.99760217e-15, 4.49640325e-15,
       6.32827124e-15, 6.21724894e-15, 6.43929354e-15, 6.21724894e-15,
       4.88498131e-15, 2.22044605e-15, 0.00000000e+00])

plt.plot(input, err1,   label="\u03B1=0.25")
plt.plot(input, err2,   label="\u03B1=0.5")
plt.plot(input, err3,   label ="\u03B1=0.75")
plt.plot(input, err4,  label ="\u03B1=1.0")
plt.xlabel("x")
plt.ylabel("Error")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),ncol=5, fancybox=True)    
#plt.savefig('line_plot.pdf',bbox_inches='tight')
plt.show()



