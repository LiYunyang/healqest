import utils
import weights
import numpy as np

def qest(est,lmax,clfile,almbar1,almbar2):
    retglm  = 0
    retclm  = 0
    nside    = utils.get_nside(lmax)
    w        = weights.weights(est,lmax,clfile)
    for i in range (0,w.ntrm):
        wX,wY,wP,sX,sY,sP = w[i][0],w[i][1],w[i][2],s[i][0],s[i][1],s[i][2]
        walmbar1          = hp.almxfl(almbar1,wX)#########
        walmbar2          = hp.almxfl(almbar2,wY)#########
        
        ### input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
        qmapX, umapX = hp.alm2map_spin( [walmbar1,np.zeros_like(walmbar1)], nside, np.abs(sX),lmax)
        qmapY, umapY = hp.alm2map_spin( [walmbar2,np.zeros_like(walmbar2)], nside, np.abs(sY),lmax)
        
        X  = qmapX+np.sign(sX)*1j*umapX#*(-1)**(sX+1)
        Y  = qmapY+np.sign(sY)*1j*umapY#*(-1)**(sY+1)
        XY = X*Y 

        glm,clm  = hp.map2alm_spin([XY.real,Y.imag], np.abs(sP), lmax)
        retglm  += glm
        retclm  += clm

    return retglm,retclm

