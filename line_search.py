import numpy as np
import matplotlib.pyplot as plt
import math
def fx(x):
    return (x-3)**2

def grad_f(x):
    return 2*(x-3)

def sgd(iter=300,eta=0.1,x=0):
    err = 1.0
    it = 0
    ys = [fx(x)]
    while err > 1e-4 and it < iter:
        x = x - grad_f(x) * eta
        # print(x)
        new_y = fx(x)
        err = math.fabs(new_y - ys[it])
        ys.append(new_y)
        it += 1
    return ys

def back_tracking(iter=300,gamma=0.25,c=0.8,x=0):
    err = 1.0
    it = 0
    ys = [fx(x)]
    while err > 1e-4 and it < iter:
        g = grad_f(x) 
        step = 1.0
        newf = fx(x - step*g)
        df = c*g*g
        # print(x-t*g)
        while newf > ys[it] - step*df:
            step *= gamma
            newf = fx(x - step*g)
            
        
        x = x-step*g
        print(x)
        new_y = fx(x)
        err = math.fabs(new_y - ys[it])
        ys.append(new_y)
        it += 1
    return ys


plt.figure(figsize=(12,8))
ys = back_tracking()
ys2 = sgd()
print(ys)

# print(np.arange(len(ys)))
plt.plot(np.arange(len(ys)),ys,label='ys1')
plt.plot(np.arange(len(ys2)),ys2,label='ys2',marker='o'  )
plt.legend()
# plt.plot([1,2],[2,3])
plt.show()


        