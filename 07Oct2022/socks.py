import numpy as np
import numba as nb
import time
from matplotlib import pyplot as plt
plt.ion()

@nb.njit
def calc_prob(npair=14,nchair=9):
    p=np.zeros((2*npair,nchair+1))
    p[0,1]=1.0
    for i in range(1,2*npair):
        p[i,1]=p[i-1,0] #if we have no socks, we are going to guaranteed have one sock
        for j in range(1,nchair+1):
            nleft=2*npair-i
            p_match=j/nleft
            p[i,j-1]=p[i,j-1]+p_match*p[i-1,j]
            if j<nchair:
                p[i,j+1]=p[i,j+1]+(1-p_match)*p[i-1,j]
    return np.sum(p[-1,:])

def find_targ(npair,targ,guess=None):
    if guess is None:
        guess=npair//2
    p0=calc_prob(npair,guess)
    if p0>guess:
        while p0>targ:
            guess=guess-1
            p0=calc_prob(npair,guess)
            #print(guess,p0)
        return guess,guess+1
    else:
        while (p0<targ):
            guess=guess+1
            p0=calc_prob(npair,guess)
            #print(guess,p0)
        return guess-1,guess
    
def get_width(npair,pl=0.25,pr=0.75):
    nn=npair//2
    p0=calc_prob(npair,nn)
    il=nn
    if p0>pl:
        while True:
            il=il-1
            p=calc_prob(npair,il)
            if p<pl:
                break
    else:
        while True:
            il=il+1
            p=calc_prob(npair,il)
            if p>pl:
                il=il-1
                break
    print('il is ',il,' with prob ',calc_prob(npair,il))
    ir=np.max([nn,il])
    while True:
        ir=ir+1
        p=calc_prob(npair,ir)
        if p>pr:
            break
    print('ir is ',ir,' with prob ',calc_prob(npair,ir))
    return il,ir
    

def gaussgrad(pars,x):
    amp=pars[0]
    x0=pars[1]
    a=pars[2]
    y=np.exp(-0.5*a*(x-x0)**2)*amp
    grad=np.zeros([len(x),3])
    grad[:,0]=y/amp
    grad[:,1]=y*a*(x-x0)
    grad[:,2]=y*(-0.5*(x-x0)**2)
    return y,grad

def fit_newton(pars,fun,x,y,niter=10):
    for i in range(niter):
        pred,grad=gaussgrad(pars,x)
        r=y-pred
        lhs=grad.T@grad
        rhs=grad.T@r
        dp=np.linalg.inv(lhs)@rhs
        #print('dp is ',dp)
        pars=pars+dp
    print('final iteration shift was ',dp)
    return pars

def get_pars(n):
    width=int(np.sqrt(n))
    xx=np.arange(n//2-2*width,n//2+3*width)
    pp=np.zeros(len(xx))
    for i in range(len(xx)):
        pp[i]=calc_prob(n,xx[i])
    y=np.diff(pp)
    pars=np.asarray([y.max(),xx[np.argmax(y)],4.5/n])
    fitp=fit_newton(pars,gaussgrad,xx[:-1],y)
    return pp,xx,fitp


assert(1==0)
    
npair=14
nchair=9

print('basic riddler answer is ',calc_prob(npair,nchair))
t1=time.time()
val=calc_prob(npair,nchair)
t2=time.time()
#print('elapsed time is ',t2-t1)
npair_max=100
nchair_max=npair_max
pmat=np.ones([npair_max,nchair_max])
for i in range(1,npair_max):
    for j in range(1,i-1): 
        pmat[i,j]=calc_prob(i,j)
pmat[:,0]=0 #we can't ever match socks if we have zero chair slots

plt.figure(1)
plt.clf()
plt.imshow(pmat)
plt.show()
plt.xlabel('Room on Chair')
plt.ylabel('Pairs of Socks')
plt.colorbar()
plt.title('Probability of Sockcess')
plt.savefig('sockcess.png')



nmax=2000
ltarg=0.32/2
rtarg=1-ltarg
if True:
    il=np.zeros(nmax,dtype='int')
    ir=np.zeros(nmax,dtype='int')
    istart=10
    il[istart],ir[istart]=get_width(istart,ltarg,rtarg)
    for i in range(istart+1,nmax):
        il[i],tmp=find_targ(i-1,ltarg,il[i-1])
        tmp,ir[i]=find_targ(i-1,rtarg,ir[i-1])
width=ir-il
plt.figure(2)
plt.clf()
x=np.arange(nmax)
vec=width[istart:]/np.sqrt(x[istart:])
plt.semilogx(x[istart:],vec,'.')
plt.show()
plt.xlabel('Pairs of Socks')
plt.ylabel('50% probability width for chair/sqrt(pairs)')
plt.title('Width of 50% Range, Scaled by 1/sqrt(npair)')
plt.savefig('scaled_width.png')
plt.figure(3)
plt.clf();
plt.plot(x[istart:],il[istart:]/x[istart:],'.');
plt.plot(x[istart:],ir[istart:]/x[istart:],'.');
plt.xlabel('pairs of socks')
plt.ylabel('Room on chair/npair')
plt.legend(['25% probability','75% probability'])
plt.title('25/75% Room on Chair for 25/75% probability of winning')
plt.show()
plt.savefig('win_probs.png')


w=ir[-1]-il[-1]
xx=np.arange(il[-1]-w,ir[-1]+w+1)
pvec=np.zeros(len(xx))
for i in range(len(xx)):
    pvec[i]=calc_prob(nmax,xx[i])


x0=nmax/2+np.sqrt(nmax)*2/7
pred=np.exp(-0.5*(xx-x0)**2/nmax*4.5)
pred=pred/pred.sum()
psum=np.cumsum(pred)
plt.figure(4)
plt.clf()
plt.plot(np.diff(pvec))
plt.plot(np.diff(psum),'.')
plt.show()

pars=np.asarray([pred.max(),x0,4.5/nmax])
fitp=fit_newton(pars,gaussgrad,xx[:-1],np.diff(pvec))
pred,grad=gaussgrad(fitp,xx[:-1])
plt.clf();plt.plot(xx[:-1],np.diff(pvec),'.');plt.plot(xx[:-1],pred);plt.show()

#y1,grad1=gaussgrad(pars,xx)
#pp=pars.copy()
#pp[2]=pp[2]+1e-6
#y2,grad2=gaussgrad(pp,xx)
#yy=(y2-y1)/1e-6
#gg=(grad1[:,2]+grad2[:,2])/2


#>>> nn=5000;aa,bb,fitp=get_pars(nn);print(1/fitp[-1]/nn,fitp[1]-nn/2)
#final iteration shift was  [ 7.62827211e-19  3.36005027e-14 -1.87115071e-20]
#0.23239793267181116 15.18388034225336
