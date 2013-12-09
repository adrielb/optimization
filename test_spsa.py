import matplotlib.pyplot as plt
import numpy as np
import SPSA as spsa
from test_func import *

spsa.alpha_p = 0.1
spsa.alpha_n = 0.1
spsa.learning_rate = 1e-2
spsa.momentum_rate = 0.1
spsa.max_iter = 500
spsa.loss = test_function
Q0 = np.array( [-4.5,-3] )


def single_run():
    df = spsa.run( Q0 )
    df.head( 50 )
    qn  = np.array( df['Qnew'].tolist() ).transpose()
    qo  = np.array( df['Qold'].tolist() ).transpose()
    dqo = np.array( df['dQold'].tolist() ).transpose()
    plot_iter( (df, qn, qo, dqo) )
    plot_spatial( (df, qn, qo, dqo) )

#import sys
#sys.exit(0)
def plot_spatial( p ):
    (df, qn, qo, dqo) = p
    plt.axes( aspect='equal' )
    plt.plot( qo[0] , qo[1] , color='white'  , linewidth=5.0 , marker='o' )
    plt.plot( qn[0] , qn[1] , color='yellow' , linewidth=0.0 , marker='o' )
    for i in xrange( qo.shape[1] ):
        plt.arrow( qo[0,i], qo[1,i], dqo[0,i], dqo[1,i], color='pink', lw=2, width=0.005, head_starts_at_zero=False )
    plot_test_function()
    plt.show()

def plot_iter( p ):
    (df, qn, qo, dqo) = p
    r = 5
    c = 1
    plt.subplot(r,c,1)
    plt.ylabel( 'Q' )
    plt.plot( qo[0] )
    plt.plot( qo[1] )
    plt.plot( qn[0], color='blue', linestyle='dashed' )
    plt.plot( qn[1], color='green', linestyle='dashed' )
    plt.subplot(r,c,2)
    plt.ylabel( 'dQ' )
    plt.plot( dqo[0] )
    plt.plot( dqo[1] )
    plt.subplot(r,c,3)
    plt.ylabel( 'L' )
    plt.semilogy( df['Lold'] )
    plt.semilogy( df['Lnew'] )
    plt.subplot(r,c,4)
    plt.ylabel('learning rate')
    plt.semilogy( df['learning_rate'], marker='.')
    plt.subplot(r,c,5)
    plt.ylabel('fail_count')
    plt.plot( df['fail_count'], marker='.' )
    plt.show()



#single_run()
numruns = 100

dfs = [ spsa.run(Q0) for i in xrange( numruns ) ]

plt.axes( aspect='equal' )
for i in xrange( numruns ):
    qo  = np.array( dfs[i]['Qold'].tolist() ).transpose()
    plt.plot( qo[0] , qo[1] , linewidth=1.0 , marker='.' )
plot_test_function()
plt.show()

for i in xrange( numruns ):
    plt.semilogy( dfs[i]['Lold'] )
plt.show()
