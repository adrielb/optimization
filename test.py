import matplotlib.pyplot as plt
import numpy as np
import SPSA as spsa

def test_function( x1, x2 ):
    l1 = 3
    l2 = 1
    A = np.array( [[ 0.5*l1 + 0.5*l2, -0.5*l1 + 0.5*l2],
                   [-0.5*l1 + 0.5*l2,  0.5*l1 + 0.5*l2]] )
    X = np.array( [x1, x2] )
    return (x1, x2, np.dot( X, np.dot( A, X ) ) )

def plot_test_function():
    dx = 0.1
    z = np.array( [
        [test_function( x, y)
        for x in np.arange( -5, 5, dx)]
        for y in np.arange( -5, 5, dx) ] )
    #y, x = np.mgrid[-5:5:dx, -5:5:dx]
    x = z[:,:,0]
    y = z[:,:,1]
    z = z[:,:,2]

    plt.pcolormesh(x,y,z)
    plt.show()

spsa.loss = test_function
Q0 = [-4,-3]
spsa.run( Q0 )
