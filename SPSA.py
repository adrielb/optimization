'''
Simultaneous Perturbation Stochastic Approximation
'''
import numpy as np

def delta_gen( p ):
    '''
    returns {-1, 1}
    '''
    return 2 * np.random.randint(low=0, high=2, size=p ) - 1

def SPSAk( ):
    c = ck( k )
    d = delta_gen( p )
    L1 = loss( Qk + c * d )
    L2 = loss( Qk - c * d )
    grad = (L1 - L2) / ( 2*c*d )

    g =  w
    dQnew = -a* g

    Qnew = Qk + dQ
    Qnew = constraints( Qnew )
    Lnew = loss( Qnew )

    if( Lnew > Lold ):
        Lold = Lnew
        Qk   = Qnew
        a = (1+alpha) * a
    else:
        a = (1-alpha) * a


