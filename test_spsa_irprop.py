import SPSA_irprop as spsa
import matplotlib.pyplot as plt
import numpy as np

# test function {{{
def test_function( X ):
    q = np.pi / 1;
    l1 = 1e0
    l2 = 1e-3
    D = np.array( [[l1 , 0],
                   [0  ,l2]] )
    R = np.array( [[np.cos(q),-np.sin(q)],
                   [np.sin(q), np.cos(q)]] )
    # Xt.Rt.D.R.X
    A = np.dot( R.transpose(), np.dot( D, R ) )
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
# }}}

reload( spsa )
spsa.loss = test_function
spsa.max_iter = 100
spsa.momentum_rate = 0.9
Q0 = np.array( [-4.5,-3] )
df = spsa.SPSArprop_minus_init(Q0)
print df.head(50)

qn  = np.array( df['Qnew'].tolist() ).transpose()
qo  = np.array( df['Qold'].tolist() ).transpose()
dd  = np.array( df['dd'].tolist() ).transpose()
go  = np.array( df['Gold'].tolist() ).transpose()
gn  = np.array( df['Gnew'].tolist() ).transpose()

plt.axes( aspect='equal' )
plt.plot( qo[0] , qo[1] , color='white'  , linewidth=5.0 , marker='o' )
plt.plot( qn[0] , qn[1] , color='yellow' , linewidth=0.0 , marker='o' )
plot_test_function()
plt.show()

r = 5
c = 1
plt.subplot(r,c,1)
plt.ylabel( 'Q' )
plt.plot( qo[0] )
plt.plot( qo[1] )
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
plt.semilogy( df['Lold'] )
plt.semilogy( df['Lnew'] )
plt.subplot(r,c,5)
plt.ylabel( 'fail' )
plt.plot( df['fail_count'] )
plt.show()
