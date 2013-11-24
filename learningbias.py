import matplotlib.pyplot as plt
import numpy as np

def loss( q ):
    return q*q

sigma = 1e-2
a     = 0.01
b     = 0.1

def search( p ):
    Qold, Lold = p
    dQ = sigma * np.random.randn()
    Qnew = Qold + dQ
    Lnew = loss( Qnew )
    if Lnew < Lold:
        Lold = Lnew
        Qold = Qnew
    return ( Qold, Lold )


def search2(p):
    Qold, Lold, bias = p
    dQ = sigma * np.random.randn() + bias
    Qnew = Qold + dQ
    Lnew = loss( Qnew )
    if Lnew < Lold:
        Lold = Lnew
        Qold = Qnew
        bias = bias + a * dQ
    else:
        bias = bias - a * dQ
    return ( Qold, Lold, bias )

def search3(p):
    Qold, Lold, bias, sigma = p
    dQ = sigma * np.random.randn() + bias
    Qnew = Qold + dQ
    Lnew = loss( Qnew )
    if Lnew < Lold:
        Lold = Lnew
        Qold = Qnew
        bias  = (1-a) * bias + a * dQ
        sigma = (1+b) * sigma
    else:
        bias  = (1-10*a) * bias - 10*a * dQ
        sigma = (1-b) * sigma
    return ( Qold, Lold, bias, sigma )

#if __name__ == "__main__":
Q0 = 10
L0 = loss( Q0 )
maxiter = 300
sol= np.zeros( (maxiter, 4) )
sol[0] = (Q0, L0, 0, sigma)
for i in xrange( 1, maxiter ):
    sol[i] = search3( sol[i-1] )

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)

ax0.set_ylim(-11,11)
ax0.set_ylabel( 'Q0' )
ax0.plot( sol[:,0] )

ax1.set_yscale('log')
ax1.set_ylabel( 'L0' )
ax1.plot( sol[:,1] )

ax2.set_ylabel( 'bias' )
ax2.plot( sol[:,2] )

ax3.set_yscale('log')
ax3.set_ylabel( 'sigma' )
ax3.plot( sol[:,3] )
plt.show()

