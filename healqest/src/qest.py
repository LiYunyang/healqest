import sys
import utils
import weights
import numpy as np
import healpy as hp
np.seterr(all='ignore')

class qest(object):

    def __init__(est,qe,almbar1,almbar2,config):
        '''
        Sets up the quatratic estimator calculation 
    
        Parameters
        ----------
        est : str
          Define what the estimator is reconstructing. Should be one of 'lens'/'src'/'prof'.  
        qe  : str
          quadratic estimator type: 'TT'/'EE'/'TE'/'EB'/'TB'
        almbar1: complex
          first filtered alms
        almbar2: complex
          second filtered alms
        config : dictionary of settings
        '''

        assert(est=='lens' or est=='src' or est=='prof', "est expected to be lens/src/prof, got: %s"%est)

        clfile = config['clfile']

        print('estimator used: %s'%est)
        self.retglm  = 0
        self.retclm  = 0
        self.nside   = utils.get_nside(Lmax)
        print("projecting to nside=%d"%nside)
        self.lmax1   = hp.Alm.getlmax(almbar1.shape[0])
        self.lmax2   = hp.Alm.getlmax(almbar2.shape[0])
        self.q       = weights.weights(est,max(lmax1,lmax2),clfile,u=u)
        print("lmax=%d"%max(lmax1,lmax2))
        print("Lmax=%d"%Lmax)

    def eval(self):
        '''
        Compute equatratic estimator 
    
        Returns
        ----------
        glm: complex
          Gradient component of the plm
        clm: 
          Curl component of the plm

        '''      

        if qe=='TB' or qe=='EB':
            # hack to get TB/EB working. currently not understanding some factors of j
            print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')

            wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
            wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

            walmbar1          = hp.almxfl(almbar1,wX1) #T1/E1
            walmbar3          = hp.almxfl(almbar1,wX3) #T3/E3
            walmbar2          = hp.almxfl(almbar2,wY1) #B2

            SpX1, SmX1   = hp.alm2map_spin( [walmbar1,np.zeros_like(walmbar1)], nside , 1,lmax1)
            SpX3, SmX3   = hp.alm2map_spin( [walmbar3,np.zeros_like(walmbar3)], nside , 3,lmax1)
            SpY2, SmY2   = hp.alm2map_spin( [np.zeros_like(walmbar2),-1j*walmbar2], nside, 2,lmax2)

            SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
            SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

            glm,clm  = hp.map2alm_spin([SpZ,SmZ],1, Lmax)

            if est=='TT' or est=='EE' or est=='TE' or est=='ET':
                nrm=0.5
            elif est=='EB':
                nrm=-1
            else:
                nrm=1

            glm = hp.almxfl(glm,nrm*wP1)
            clm = hp.almxfl(clm,nrm*wP1)
            return glm,clm

        else:
            # More traditional quicklens style calculation
            
            for i in range(0,q.ntrm):
                
                wX,wY,wP,sX,sY,sP = q.w[i][0],q.w[i][1],q.w[i][2],q.s[i][0],q.s[i][1],q.s[i][2]
                print("computing term %d/%d sj=[%d,%d,%d]"%(i+1,q.ntrm,sX,sY,sP))
                walmbar1          = hp.almxfl(almbar1,wX)
                walmbar2          = hp.almxfl(almbar2,wY)

                #print(sP,u[i])

                ### input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
                #-----------------------------------------------------------------------------------------------
                if est[0]=='B':
                    SpX, SmX = hp.alm2map_spin( [np.zeros_like(walmbar1),1j*walmbar1], nside, np.abs(sX),lmax1)
                else:
                    SpX, SmX = hp.alm2map_spin( [walmbar1,np.zeros_like(walmbar1)], nside, np.abs(sX),lmax1)
                    
                X  = SpX+1j*SmX # Complex map _{+s}S or _{-s}S
                
                if sX<0:
                    X = np.conj(X)*(-1)**(sX)
                #-----------------------------------------------------------------------------------------------
                if est[1]=='B':
                    SpY, SmY = hp.alm2map_spin( [np.zeros_like(walmbar2),1j*walmbar2], nside, np.abs(sY),lmax2)
                else:
                    SpY, SmY = hp.alm2map_spin( [walmbar2,np.zeros_like(walmbar2)], nside, np.abs(sY),lmax2)
                    
                Y  = SpY+1j*SmY
                
                if sY<0:
                    Y = np.conj(Y)*(-1)**(sY)
                #-----------------------------------------------------------------------------------------------
                
                XY = X*Y
                
                if sP<0:
                    XY = np.conj(XY)*(-1)**(sP)

                glm,clm  = hp.map2alm_spin([XY.real,XY.imag], np.abs(sP), Lmax)
                
                    

                glm = hp.almxfl(glm,0.5*wP)
                clm = hp.almxfl(clm,0.5*wP)

                retglm  += glm
                retclm  += clm

            return retglm,retclm


    def get_aresp(self,flm1,flm2):
        '''
        Compute analytical response function

        Parameters
        ----------
        flm1 : float, array
          file containing plm1 dictionary. Should have entries 'glm' and 'analytical_resp'
        flm2 : float, array
          file containing plm2 dictionary. Should have entries 'glm' and 'analytical_resp'
        
        Returns
        -------
        resp:
          Analytical response function
        '''

        aresp   = resp.fill_resp(weights.weights(qe,dict_lrange['Lmax'],cambcls,u=u_ell),
                                 np.zeros(dict_lrange['Lmax']+1, dtype=np.complex_), flm1, flm2)



    def harden(self,u=None):
        '''cross-estimator response needed for source hardening
        TO DO: make it pass arrays directly

        Parameters
        ----------
        file_plm1 : dict
          file containing plm1 dictionary. Should have entries 'glm' and 'analytical_resp'
        file_plm2 : dict
          file containing plm2 dictionary. Should have entries 'glm' and 'analytical_resp'
        qe1 : str   
          quadratic estimator 
        qe2 : str   
          quadratic estimator 
        cambcls: 
          Cl file produced by camb (fix soon)
        dict_cls:
          Dictionary containing various Cls
        dict_lrange:
          Dictionary containing lcuts
        u : float 
          Array containing profile shape

        Returns
        -------
          source hardened glm and response function
        '''
        tmp        = np.load(file_plm1)
        plm1,resp1 = tmp['glm'], tmp['analytical_resp']

        tmp   = np.load(file_plm2)
        plm2,resp2 = tmp['glm'], tmp['analytical_resp']

        resp12     = resp_xest(qe1,qe2,cambcls,dict_cls,dict_lrange,u=u_ell)

        weight     = -1*resp12 / resp2
        plm        = plm1 + hp.almxfl(plm2, weight)
        resp       = srchard_weighting(resp1,resp12,resp2,weight)
        return plm, resp
