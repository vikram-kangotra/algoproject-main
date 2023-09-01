import numpy as np

def seq_extr(data,a,length,ctr=0,n=25,ac=[]):
    ar=[]
    ac=[]
    for sym in data:
        ar.append(sym)
    ab=np.asarray(ar)
    for sym in data:
        ctr=ctr+1
        if ctr == int(a) and length >= 2*n+1:
            if ctr < n:
                print("hello")
                b = n-ctr+1
                for i in range(0,b):
                    ac.append('X')
                    ctr = ctr+1
                for i in range(b,ctr+n):
                    ac.append(ab[i-b])
            elif length < ctr+n+1:
                print('hiii')
                for i in range(ctr-n-1,length):
                    ac.append(ab[ctr-n-1])
                    ctr = ctr+1
                for i in range(int(length)-int(a)+n+1,2*n+1):
                    ac.append('X')
            else:
                for i in range(ctr-(n+1), ctr + n):
                    ac.append(ab[i])
        y = "".join(ac)
    return y
