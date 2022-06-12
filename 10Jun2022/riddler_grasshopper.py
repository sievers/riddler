import numpy as np
from matplotlib import pyplot as plt
plt.ion()


jump_len=0.2
ruler_len=1.0
npt=1001

try:
    print(vals.keys())
except:
    vals={}

dx=ruler_len/(npt-1)
nn=int(jump_len/dx)

x=np.linspace(0,ruler_len,npt)
mat=np.zeros([npt,npt])

for i in range(npt):
    imin=i-nn
    imax=i+nn+1
    if imin<0:
        imin=0
    if imax>npt:
        imax=npt
    mat[imin:imax,i]=1.0/(imax-imin)

#xx=np.zeros(npt)
#xx[npt//10]=1.0
#y=mat@xx

ee,vv=np.linalg.eig(mat)
ind=np.argmax(np.real(ee))
val=np.real(vv[0,ind]/vv[npt//2,ind])
vals[npt]=val
print(val,val-(0.5+1.25/npt))
plt.clf();
vec=np.real(vv[:,ind])
vec=vec/vec.max()
plt.clf();
plt.plot(x,vec)
plt.title('relative PDF')
plt.show()
print('largest eigenvalue is ',np.real(ee[ind]),', expected 1.')
if False:  #this was because I was bored and wanted to look at the higher order Taylor series terms
    xvec=[kk for kk in vals.keys()]
    yvec=[vals[kk] for kk in xvec]
    xvec=1/np.asarray(xvec)
    yvec=np.asarray(yvec)
    pp=np.polyfit(xvec,yvec,2)
    print(pp)
