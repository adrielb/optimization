import numpy as np

c= 1e-3

def loss( Q ):
    return Q[0] ** 2 + Q[1] ** 2

def SPSA( Q, d1, d2 ):
    d = np.array( [d1,d2] )
    L1   = loss( Q + c * d )
    L2   = loss( Q - c * d )
    return (L1 - L2) / ( 2*c*d )

def SPSAsign( Q ):
    d = [
        SPSA( Q, d1, d2)
        for d1 in [-1, 1]
        for d2 in [-1, 1]
    ]
    print Q
    d.insert(0,  loss(Q) )
    return d


a = np.array( [ [ (q1, q2, np.array( [q1, q2] ) )
  for q1 in np.linspace(-6,6,10) ]
  for q2 in np.linspace(-6,6,10) ] )

print a[0,0]
import sys
sys.exit(0)

a = np.array( [ [ (q1, q2, SPSAsign( np.array( q1, q2) ) )
  for q1 in np.linspace(-6,6,100) ]
  for q2 in np.linspace(-6,6,100) ] )

x  = a[:,:,0]
y  = a[:,:,1]
l  = a[:,:,2,0]
s0 = a[:,:,2,1,0]
s1 = a[:,:,2,1,1]
import matplotlib.pyplot as plt
r = 2
c = 2
plt.subplot(r,c,1)
plt.pcolormesh(x,y,l)
