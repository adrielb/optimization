'''
Simultaneous Perturbation Stochastic Approximation
'''
import numpy as np

alpha = 0.05
c = 1e-3
momentum_rate = 0.9
learning_rate = 1.0
max_iter = 10
loss = None


def delta_gen( p ):
    '''
    returns {-1, 1}
    '''
    return 2 * np.random.randint(low=0, high=2, size=p ) - 1

def SPSAk( Lold, Qold, learning_rate ):
    d = delta_gen( p )
    L1 = loss( Qold + c * d )
    L2 = loss( Qold - c * d )
    grad = (L1 - L2) / ( 2*c*d )

    dQnew = momentum_rate * dQold - learning_rate * grad

    Qnew = Qold + dQnew
    Qnew = constraints( Qnew )
    Lnew = loss( Qnew )

    if( Lnew < Lold ):
        Lold = Lnew
        Qold = Qnew
        learning_rate = (1+alpha) * learning_rate
    else:
        learning_rate = (1-alpha) * learning_rate

    return ( Lnew, Qnew, learning_rate )

def constraints( Q ):
    pass

def run( Q0 ):
    sol = [0] * max_iter
    sol[0] = ( loss(Q0), Q0, learning_rate )
    for i in xrange( max_iter ):
        sol[i] = SPSAk( *sol[i-1] )
    return sol




