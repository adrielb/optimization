'''
Simultaneous Perturbation Stochastic Approximation
'''
import numpy as np
import pandas as pd

alpha_p = 0.05
alpha_n = 0.05
c = 1e-3
momentum_rate = 0.8
learning_rate = 1.0
fail_thres = 3e3
max_iter = 100
loss = None
Qsize = None


def delta_gen( ):
    '''
    returns {-1, 1}
    '''
    return 2 * np.random.randint(low=0, high=2, size=Qsize ) - 1

def SPSAk(params={}):

    p = params.copy()
    Lold          = p['Lold']
    Qold          = p['Qold']
    dQold         = p['dQold']
    learning_rate = p['learning_rate']

    d    = delta_gen()
    L1   = loss( Qold + c * d )
    L2   = loss( Qold - c * d )
    grad = (L1 - L2) / ( 2*c*d )

    if( p['fail_count'] >= fail_thres ):
        dQold *= 0.0

    dQnew = momentum_rate * dQold - learning_rate * grad

    Qnew = Qold + dQnew
    Qnew = constraints( Qnew )
    Lnew = loss( Qnew )

    if( Lnew < Lold ):
        Lold = Lnew
        learning_rate = (1+alpha_p) * learning_rate
        p['fail_count'] = 0
    else:
        learning_rate = (1-alpha_n) * learning_rate
        p['fail_count'] += 1

    Qold = Qnew
    p['Lnew']          = Lnew
    p['Qnew']          = Qnew
    p['Lold']          = Lold
    p['Qold']          = Qold
    p['dQold']         = dQnew
    p['learning_rate'] = learning_rate
    return p

def constraints( Q ):
    return Q

def run( Q0 ):
    global Qsize
    Qsize = Q0.size
    sol = [0] * max_iter
    init = {'Lold' : loss(Q0),
            'Qold' : Q0,
            'Qnew' : Q0,
            'dQold': np.zeros(Qsize),
            'fail_count' : 0,
            'learning_rate' : learning_rate }
    sol[0] = init
    for i in xrange( 1, max_iter ):
        sol[i] = SPSAk( sol[i-1] )

    return pd.DataFrame( sol )

def learn_c():
    c = 1e-3 * np.ones( Qsize )
    d = np.zeros_like( c )
    for i in xrange( Qsize ):
        idx.fill( 0 )
        idx[i] = 1
        L1 = loss( Q0 + c[i] * d )
        L2 = loss( Q0 - c[i] * d )

        dL = L2 - L1
        print dL
