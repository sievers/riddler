import numpy as np


def numgen(pair,n):
    num=0
    for i in range(n):
        ii=i%2
        num=num+pair[ii]*10**i
    return num
    
def ndigit(n):
    #return int(np.floor(np.log10(n)))
    ndig=0
    while n>0:
        n=n//10
        ndig=ndig+1
    return ndig

def check_num(n):
    digits=[]
    while n>0:
        digits.append(n%10)
        n=n//10
    v=[]
    for i in range(len(digits)):
        tmp=0
        for k in range(len(digits)):
            if k==i:
                tmp=tmp+np.arange(10)*10**k
            else:
                tmp=tmp+10**k*digits[k]
        v.append(tmp)
    v=np.asarray(v)
    vv=np.unique(v%11)
    if len(vv)==11:
        return 0
    elif 0 in vv:
        return 1
    else:
        return 2

def check_num_safe(n):
    digits=[]
    while n>0:
        digits.append(n%10)
        n=n//10
    v=[]
    for i in range(len(digits)):
        tmp=[0]
        for k in range(len(digits)):
            if k==i:
                #tmp=tmp+np.arange(10)*10**k
                tmp=[tmp[0]+j*10**k for j in range(10)]
            else:
                #tmp=tmp+10**k*digits[k]
                tmp=[num+10**k*digits[k] for num in tmp]
        for x in tmp:
            v.append(x)
        #v.append(x for x in tmp)
    #print(type(v))
    #return v
    v=[x%11 for x in v]
    #v=np.asarray(v%11)
    vv=np.unique(v)
    if len(vv)==11:
        return 0
    elif 0 in vv:
        return 1
    else:
        return 2

def check_num_recursive(n,all_bads=[],nmax=1e17):
    val=check_num_safe(n)
    if val==2:
        #print('we have a loser at: ',n)
        all_bads.append(n)
    if val>0:
        #print('candidate at ',n)
        ndig=ndigit(n)
        for i in np.arange(1,10):
            nn=i*(10**(ndig))+n
            #print(nn)
            #num=i*10**(ndig+1)+n
            #print(new_nums)
            if nn<nmax:
                check_num_recursive(nn,all_bads,nmax)


all_bads=[]
for i in np.arange(1,10):
    check_num_recursive(i,all_bads,nmax=10**17)
#print(all_bads)
all_bads.sort()
for num in all_bads:
    print(num)

pairs=[[1,8],[2,7],[3,6],[4,5],[5,4],[6,3],[7,2],[8,1]]

new_bads=[]
ndig_max=100
for pair in pairs:
    for i in range(1,ndig_max):
        num=numgen(pair,i)
        val=check_num_safe(num)
        if val==2:
            #print(num)
            new_bads.append(num)
        if val==0:
            print('pair ',pair,' stopped.')
            break
new_bads.sort()
print('all non-11 numbers with up to ',ndig_max,' digits are: ')
for x in new_bads:
    print(x)

    
