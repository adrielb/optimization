'''
SPSA iRprop
'''
import numpy as np
import pandas as pd

c = 1e-3
max_iter = 10
eta_plus  = 1.2
eta_minus = 0.5
ddMax = 1e3
ddMin = 1e-5
momentum_rate = 0.0

def SPSArprop_minus( params={} ):

    p = params.copy()
    Lnew  = p['Lnew']
    Lold  = p['Lold']
    Qnew  = p['Qnew'].copy()
    Qold  = p['Qold'].copy()
    Gold  = p['Gnew'].copy()
    dd    = p['dd'].copy()
    dQ    = p['dQ'].copy()
    sign_flip = list(p['sign_flip'])

    Qold = Qnew

    if( Lnew < Lold ):
        Lold = Lnew
        p['fail_count'] = 0
    else:
        p['fail_count'] += 1

    d    = delta_gen()
    L1   = loss( Qold + c * d )
    L2   = loss( Qold - c * d )
    grad = (L1 - L2) / ( 2*c*d )

    Gnew = momentum_rate * Gold + (1-momentum_rate) * grad
    Gsign = np.sign( Gnew )

    for i in xrange( dd.size ):
        if Gold[i] * Gnew[i] > 0:
            if not sign_flip[i]:
                dd[i]   = min( eta_plus  * dd[i], ddMax )
            sign_flip[i] = False
            dQ[i]   = -Gsign[i] * dd[i]
            Qnew[i] = Qold[i] + dQ[i]
        else:
            sign_flip[i] = True
            dd[i]   = max( eta_minus * dd[i], ddMin )
            Qnew[i] = Qold[i] - dQ[i]

    #Qnew = Qold +  -Gsign * dd
    Qnew = constraints( Qnew )
    Lnew = loss( Qnew )

    p['Lold'] = Lold
    p['Lnew'] = Lnew
    p['Qnew'] = Qnew
    p['Qold'] = Qold
    p['dd']   = dd
    p['Gold'] = Gold
    p['Gnew'] = Gnew
    p['dQ']   = dQ
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
            'dQ'   : np.zeros(Qsize),
            'sign_flip' : [False] * Qsize,
            'fail_count' : 0
           }
    sol[0] = init
    for i in xrange( 1, max_iter ):
        sol[i] = SPSArprop_minus( sol[i-1] )

    return pd.DataFrame( sol )

def delta_gen( ):
    '''
    returns {-1, 1}
    '''
    return 2 * np.random.randint(low=0, high=2, size=Qsize ) - 1
def constraints( Q ):
    return Q
