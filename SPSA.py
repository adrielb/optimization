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

    if( p['fail_count'] >= 3 ):
        dQold *= 0.0

    dQnew = momentum_rate * dQold - learning_rate * grad

    Qnew = Qold + dQnew
    Qnew = constraints( Qnew )
    Lnew = loss( Qnew )

    if( Lnew < Lold ):
        Lold = Lnew
        Qold = Qnew
        learning_rate = (1+alpha_p) * learning_rate
        p['fail_count'] = 0
    else:
        learning_rate = (1-alpha_n) * learning_rate
        p['fail_count'] += 1


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

eta_plus  = 1.2
eta_minus = 0.5
ddMax = 1e3
ddMin = 1e-3
def SPSArprop_minus( params={} ):

    p = params.copy()
    Lnew  = p['Lnew']
    Lold  = p['Lold']
    Qnew  = p['Qnew'].copy()
    Qold  = p['Qold'].copy()
    Gold  = p['Gnew'].copy()
    ddold = p['dd'].copy()
    if( Lnew < Lold ):
        Lold = Lnew
        Qold = Qnew
        p['fail_count'] = 0
    else:
        dd = eta_minus * dd
        p['fail_count'] += 1

    d    = delta_gen()
    L1   = loss( Qold + c * d )
    L2   = loss( Qold - c * d )
    grad = (L1 - L2) / ( 2*c*d )

    Gnew = momentum_rate * Gold + (1-momentum_rate) * grad
    Gsign = np.sign( Gnew )

    for i in xrange( dd.size ):
        if Gold[i] * Gnew[i] > 0:
            dd1[i] = min( eta_plus  * dd0[i], ddMax )
        else:
            dd1[i] = max( eta_minus * dd0[i], ddMin )

    Qnew = Qold +  -Gsign * dd
    Qnew = constraints( Qnew )
    Lnew = loss( Qnew )

    p['Lold'] = Lold
    p['Lnew'] = Lnew
    p['Qnew'] = Qnew
    p['Qold'] = Qold
    p['dd']   = dd
    p['Gold'] = Gold
    p['Gnew'] = Gnew
    return p

def SPSArprop_minus_init( Q0 ):
    global Qsize
    Qsize = Q0.size
    sol = [0] * max_iter
    L0 = loss( Q0 )
    init = {'Lold' : L0,
            'Lnew' : L0,
            'Qold' : Q0,
            'Qnew' : Q0,
            'Gold' : np.zeros(Qsize),
            'Gnew' : np.zeros(Qsize),
            'dd'   : np.ones( Qsize ),
            'fail_count' : 0
           }
    sol[0] = init
    for i in xrange( 1, max_iter ):
        sol[i] = SPSArprop_minus( sol[i-1] )

    return pd.DataFrame( sol )
