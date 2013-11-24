import numpy as np

logfile = open( logfilename, 'w' )

Qlen = Qk.shape[0]

dQ = sigma * np.random.randn( Qlen )
Qnew = Qk + dQ
Qnew = apply_constraints( Qnew )
Lnew = LossFunction( Qnew )

if Lnew < Lold:
    Lold = Lnew
    Qk   = Qnew






'''
while True:
    pass
'''
