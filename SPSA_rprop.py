'''
SPSA Rprop
'''
import numpy as np
import pandas as pd

c  = 1e-3
eta_plus  = 1.5
eta_minus = 0.5
ddMax = 1e3
ddMin = 1e-2
fail_thres = 5

def SPSArprop_minus( params={} ):
    global ddMin

    p    = params.copy()
    Lmin = p['Lmin']
    Lnew = p['Lnew']
    Qmin = p['Qmin'].copy()
    Qold = p['Qnew'].copy()
    Gold = p['Gnew'].copy()
    dd   = p['dd'].copy()

    if p['fail_count'] == fail_thres:
        p['fail_count'] = 0
        Qold = Qmin
        ddMin *= 0.5
        Gold *= 0

    d    = delta_gen()
    L1   = loss( Qold + c * d )
    L2   = loss( Qold - c * d )
    grad = (L1 - L2) / ( 2*c*d )

    Gnew = momentum_rate * Gold + (1-momentum_rate) * grad
    Gsign = np.sign( Gnew )

    for i in xrange( dd.size ):
        if Gold[i] * Gnew[i] > 0:
            dd[i] = min( eta_plus  * dd[i], ddMax )
        else:
            dd[i] = max( eta_minus * dd[i], ddMin )

    Qnew = Qold +  -Gsign * dd
    Qnew = constraints( Qnew )
    Lnew = loss( Qnew )

    if Lnew < Lmin:
        Lmin = Lnew
        Qmin = Qnew
        p['fail_count'] = 0
    else:
        p['fail_count'] += 1

    p['Lmin'] = Lmin
    p['Lnew'] = Lnew
    p['Qmin'] = Qmin
    p['Qnew'] = Qnew
    p['Qold'] = Qold
    p['dd']   = dd
    p['Gold'] = Gold
    p['Gnew'] = Gnew
    return p

def run( Q0 ):
    global Qsize
    Qsize = Q0.size
    sol = [0] * max_iter
    L0 = loss( Q0 )
    init = {'Lmin' : L0,
            'Lnew' : L0,
            'Qmin' : Q0,
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

def delta_gen( ):
    '''
    returns {-1, 1}
    '''
    return 2 * np.random.randint(low=0, high=2, size=Qsize ) - 1

def constraints( Q ):
    return Q
