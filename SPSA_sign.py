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
        np.sign( SPSA( Q, d1, d2) )
        for d1 in [-1, 1]
        for d2 in [-1, 1]
    ]
    return d

#import sys
#sys.exit(0)
r1 = np.linspace(-6,6,10)
r2 = np.linspace(-6,6,10)
a = np.array( [ [ (q1, q2, loss([q1,q2]) )
        for q1 in r1 ]
        for q2 in r2 ] )
x  = a[:,:,0]
y  = a[:,:,1]
l  = a[:,:,2]
s  = np.array( [ SPSAsign( Q[0:2] ) for Q in a.reshape(-1, 3) ] )
print s.shape

s00 = s[:,0,0].reshape( x.shape )
s01 = s[:,0,1].reshape( x.shape )
s10 = s[:,1,0].reshape( x.shape )
s11 = s[:,1,1].reshape( x.shape )

import matplotlib.pyplot as plt
r = 2
c = 2
plt.subplot(r,c,1)
plt.title( '0 x' )
plt.pcolormesh(x,y,s00)
plt.subplot(r,c,2)
plt.title( '0 y' )
plt.pcolormesh(x,y,s01)
plt.subplot(r,c,3)
plt.title( '1 x' )
plt.pcolormesh(x,y,s00+s10)
plt.subplot(r,c,4)
plt.title( '1 y' )
plt.pcolormesh(x,y,s11)
plt.show()
