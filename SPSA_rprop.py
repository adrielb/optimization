'''
SPSA Rprop
'''
import numpy as np
import pandas as pd
import csv
import os.path

c  = 1e-3
eta_plus  = 1.5
eta_minus = 0.5
ddMax = 1e3
ddMin = 1e-1
ddMin_reduction = 0.5
fail_thres = 5
momentum_rate = 0.1
max_iter = 10
logfile = '/tmp/opt.csv'

def SPSArprop_minus( params={} ):
    global ddMin

    p    = params.copy()
    Lmin = p['Lmin']
    Lnew = p['Lnew']
    Qmin = p['Qmin'].copy()
    Qold = p['Qnew'].copy()
    Gold = p['Gnew'].copy()
    dd   = p['dd'].copy()
    ddMin= p['ddMin']

    if p['fail_count'] >= fail_thres:
        p['fail_count'] = 0
        Qold = Qmin
        ddMin *= ddMin_reduction
        Gold *= 0

    d  = delta_gen()
    Q1 = constraints( Qold + c * d )
    L1 = loss( Q1 )
    Q2 = constraints( Qold - c * d )
    L2 = loss( Q2 )

    grad = (L1 - L2) / ( 2*c*d )

    Gnew = momentum_rate * Gold + (1-momentum_rate) * grad
    Gsign = np.sign( Gnew )

    for i in xrange( dd.size ):
        if Gold[i] * Gnew[i] > 0:
            dd[i] = min( eta_plus  * dd[i], ddMax )
        elif Gold[i] * Gnew[i] < 0:
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
    p['dd']   = dd
    p['ddMin']= ddMin
    p['Gold'] = Gold
    p['Gnew'] = Gnew
    p['idx'] += 1
    return p

def init( Q0 ):
    global Qsize
    Qsize = Q0.size
    L0 = loss( Q0 )
    return {'Lmin'      : L0,
           'Lnew'       : L0,
           'Qmin'       : Q0,
           'Qnew'       : Q0,
           'Gold'       : np.zeros(Qsize),
           'Gnew'       : np.zeros(Qsize),
           'dd'         : np.ones( Qsize ),
           'ddMin'      : ddMin,
           'fail_count' : 0,
           'idx'        : 0
           }

def init_from_log():
    global Qsize
    df = readlog()
    init = df.iloc[-1].to_dict()
    Qsize = init['Qmin'].size
    init['Lmin'] = loss( init['Qmin'] )
    return init

def run_df( Q0 ):
    sol = [0]*max_iter
    sol[0] = init( Q0 )

    for i in xrange( 1, max_iter ):
        sol[i] = SPSArprop_minus( sol[i-1] )

    return pd.DataFrame( sol )

def run_log( Q0=None ):
    if Q0 == None:
        sol = init_from_log()
        mode = 'a'
    else:
        sol = init( Q0 )
        mode = 'w'
    with open( logfile, mode ) as log:
        csvwriter = csv.DictWriter( log, sorted( sol.keys() ) )
        if Q0 != None:
            csvwriter.writerow( dict( (k,k) for k in sol.keys() ) )
            csvwriter.writerow( sol )
        for i in xrange( 1, max_iter ):
            sol = SPSArprop_minus( sol )
            csvwriter.writerow( sol )

def readlog():
    df = pd.read_csv(logfile)
    for col in ['Qmin', 'Qnew', 'Gold', 'Gnew', 'dd' ]:
        df[col] = pd.Series( [ np.array(
            [float(f) for f in row.strip('[]').split() ]
        ) for row in df[col]] )
    return df

def delta_gen( ):
    '''
    returns {-1, 1}
    '''
    return 2 * np.random.randint(low=0, high=2, size=Qsize ) - 1

def constraints( Q ):
    return Q
