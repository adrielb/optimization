import matplotlib.pyplot as plt
import numpy as np
import SPSA as spsa

def test_function( X ):
    l1 = 1e-3
    l2 = 1e-3
    A = np.array( [[l1 , 0],
                   [0  ,l2]] )
    return np.dot( X, np.dot( A, X ) )

def test_function2( X ):
    l1 = 1e-3
    l2 = 1e-3
    A = np.array( [[ 0.5*l1 + 0.5*l2, -0.5*l1 + 0.5*l2],
                   [-0.5*l1 + 0.5*l2,  0.5*l1 + 0.5*l2]] )
    return np.dot( X, np.dot( A, X ) )

def plot_test_function():
    dx = 0.1
    z = np.array( [ [
        (x, y, test_function( np.array((x, y))) )
        for x in np.arange( -5, 5, dx)]
        for y in np.arange( -5, 5, dx)] )
    #y, x = np.mgrid[-5:5:dx, -5:5:dx]
    x = z[:,:,0]
    y = z[:,:,1]
    z = z[:,:,2]

    plt.pcolormesh(x,y,z)


reload( spsa )
spsa.alpha = 1e-1
spsa.learning_rate = 1e-4
spsa.max_iter = 50
spsa.momentum_rate = 0.1
spsa.loss = test_function
Q0 = np.array( [-4.5,-3] )
Q0 = np.array( [-0.005,-3] )
df = spsa.run( Q0 )

df.head( 50 )

qn  = np.array( df['Qnew'].tolist() ).transpose()
qo  = np.array( df['Qold'].tolist() ).transpose()
dqo = np.array( df['dQold'].tolist() ).transpose()

#import sys
#sys.exit(0)
def plot_spatial():
    plt.axes( aspect='equal' )
    plt.plot( qo[0] , qo[1] , color='white'  , linewidth=5.0 , marker='o' )
    plt.plot( qn[0] , qn[1] , color='yellow' , linewidth=0.0 , marker='o' )
    for i in xrange( qo.shape[1] ):
        plt.arrow( qo[0,i], qo[1,i], dqo[0,i], dqo[1,i], color='pink', lw=2, width=0.005, head_starts_at_zero=False )
    plot_test_function()
    plt.show()

def plot_iter():
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
    plt.plot( df['fail_count'], marker='.' )
    plt.show()

plot_iter()
plot_spatial()
