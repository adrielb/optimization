import numpy as np
from test_func import *

c= 1e-3

loss =  test_function

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

def DerivSign( Q ):
    g = np.zeros_like(Q)
    d = np.zeros_like(Q)

    for i in xrange( g.size ):
        d *= 0
        d[i] = 1

        L1   = loss( Q + c * d )
        L2   = loss( Q - c * d )
        g[i] = (L1 - L2) / (2*c)
    return g

#import sys
#sys.exit(0)
dq = 50
a = np.array( [ [ ( q1, q2, loss( [q1,q2] ))
        for q1 in np.linspace(-6,6,dq) ]
        for q2 in np.linspace(-6,6,dq) ] )
x  = a[:,:,0]
y  = a[:,:,1]
l  = a[:,:,2]

s  = np.array( [ SPSAsign( Q[0:2] ) for Q in a.reshape(-1, 3) ] )
ds  = np.array( [ DerivSign( Q[0:2] ) for Q in a.reshape(-1, 3) ] )
print s.shape

dx  = ds[:,0].reshape( x.shape )
dy  = ds[:,1].reshape( x.shape )
s00 = s[:,0,0].reshape( x.shape )
s01 = s[:,0,1].reshape( x.shape )
s10 = s[:,1,0].reshape( x.shape )
s11 = s[:,1,1].reshape( x.shape )

import matplotlib.pyplot as plt
r = 4
c = 2
plt.subplot(r,c,1)
plt.title( '0 x' )
plt.pcolormesh(x,y,s00)
plt.subplot(r,c,2)
plt.title( '0 y' )
plt.pcolormesh(x,y,s01)
plt.subplot(r,c,3)
plt.title( '1 x' )
plt.pcolormesh(x,y,s10)
plt.subplot(r,c,4)
plt.title( '1 y' )
plt.pcolormesh(x,y,s11)
plt.subplot(r,c,5)
plt.title( 'x0+1' )
plt.pcolormesh(x,y,s00+s10)
plt.subplot(r,c,6)
plt.title( 'y0+1' )
plt.pcolormesh(x,y,s01+s11)
plt.subplot(r,c,7)
plt.title( 'dx' )
plt.pcolormesh(x,y,dx)
plt.subplot(r,c,8)
plt.title( 'dy' )
plt.pcolormesh(x,y,dy)

plt.show()

plot_test_function()
plt.show()
