import matplotlib.pyplot as plt
import numpy as np
import SPSA_rprop as spsa
from test_func import *

reload( spsa )
#spsa.eta_plus = 1.0
#spsa.eta_minus= 1.0
spsa.max_iter = 100
spsa.momentum_rate = 0.5
spsa.fail_thres = 5
spsa.loss = test_function
Q0 = np.array( [-4.1,4.1] )

#import sys
#sys.exit(0)

def multirun():
    numruns = 100

    dfs = [ spsa.run_df(Q0) for i in xrange( numruns ) ]

    plt.axes( aspect='equal' )
    for i in xrange( numruns ):
        qm  = np.array( dfs[i]['Qmin'].tolist() ).transpose()
        plt.plot( qm[0] , qm[1] , linewidth=1.0 , marker='.' )
    plot_test_function()
    plt.show()

    for i in xrange( numruns ):
        plt.semilogy( dfs[i]['Lmin'] )
    plt.show()

    #for i in xrange( numruns ):
        #if dfs[i]['Lmin'].iloc[-1] > 1e0:
            #df = dfs[i]
            #break
#multirun()

def constraints( Q ):
    Q[0] = 2
    return np.clip( Q, 0, 10 )

spsa.constraints = constraints
spsa.run_log(Q0)
spsa.run_log()

#df = spsa.run_df( Q0 )
df = spsa.readlog()

qn = np.array( df['Qnew'].tolist() ).transpose()
qm = np.array( df['Qmin'].tolist() ).transpose()
dd = np.array( df['dd'].tolist() ).transpose()
go = np.array( df['Gold'].tolist() ).transpose()
gn = np.array( df['Gnew'].tolist() ).transpose()

plt.axes( aspect='equal' )
plt.plot( qn[0] , qn[1] , color='white' , linewidth=5.0 , marker='o' )
plt.plot( qm[0] , qm[1] , color='red'   , linewidth=0.0 , marker='o' )
plot_test_function()
plt.show()


r = 5
c = 1
plt.subplot(r,c,1)
plt.ylabel( 'Q' )
plt.plot( qm[0] )
plt.plot( qm[1] )
plt.plot( qn[0] , color='blue'  , linestyle='dashed' )
plt.plot( qn[1] , color='green' , linestyle='dashed' )
plt.subplot(r,c,2)
plt.ylabel( 'dd' )
plt.semilogy( dd[0] )
plt.semilogy( dd[1] )
plt.subplot(r,c,3)
plt.ylabel( 'G' )
plt.plot( go[0], marker='o')
plt.plot( go[1], marker='.')
plt.plot( gn[0] , color='blue'  , linestyle='dashed', marker='o' )
plt.plot( gn[1] , color='green' , linestyle='dashed', marker='.' )
plt.subplot(r,c,4)
plt.ylabel( 'L' )
plt.semilogy( df['Lmin'] )
plt.semilogy( df['Lnew'] )
plt.subplot(r,c,5)
plt.ylabel( 'fail' )
plt.plot( df['fail_count'] )
plt.show()
