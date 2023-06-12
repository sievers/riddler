import numpy as np
import ratfit
from matplotlib import pyplot as plt

def stir_err(n):
    truth=0.0*n
    stir=0.0*n
    for i in range(len(n)):
        truth[i]=np.sum(np.log(np.arange(1,n[i]+1)))
        stir[i]=n[i]*(np.log(n[i])-1)
    return np.exp(truth-stir)
    

plt.ion()




n=200
lnx=np.linspace(np.log(1),np.log(n),1001)
x=np.exp(lnx)
x=np.round(x)
y=stir_err(x)

nn=4
mm=3

lnxx=np.linspace(np.log(1),np.log(n),nn+mm)
xx=np.exp(lnxx)
xx=np.round(xx)
yy=stir_err(xx)
aa,bb=ratfit.ratfit_exact(xx,yy,nn,mm)

aa2,bb2=ratfit.ratfit_lsqr(x,y,aa,bb)

y2=ratfit.rateval(aa2,bb2,x)

print('rational function numerator: ',aa2)
print('rational function denominator: ',bb2)


plt.clf();
plt.plot(x,y2/y)
#plt.plot(x,y2)
plt.show()

assert(1==0)
