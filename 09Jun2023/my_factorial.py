import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def myfact(n,do_log=False):
    
    top=1.30899952+2.39491933*n+0.29909052*n**2+0.00287671*n**3
    bot=1.0+4.57014176e-01*n+1.66387218e-02*n**2+3.02339218e-05*n**3

    if do_log==False:
        return top/bot*(n/np.e)**n
    else:
        return n*(np.log(n)-1)+np.log(top)-np.log(bot)

def myfact2(n,do_log=False,nmax=200):
    fac=np.sqrt(2*np.pi*np.sqrt(nmax))
    if do_log==False:
        return fac*(n/np.e)**n
    else:
        return n*(np.log(n)-1)+np.log(fac)

nvec=np.arange(1,201)
truth=np.cumsum(np.log(nvec))  #get the log of the factorial
pred=myfact2(nvec,True)
plt.clf()
plt.plot(nvec,np.exp(pred-truth))
plt.show()
