import numpy as np
import matplotlib.pyplot as plt

def test_function( X ):
    q = np.pi / 5
    l1 = 1e-3
    l2 = 1e0
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

    plt.axes( aspect='equal' )
    plt.pcolormesh(x,y,z)
