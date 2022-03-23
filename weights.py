import numpy as np
import utils 

class weights():
    def __init__(self,est,lmax,clfile):
        l  = np.arange(lmax+1,dtype=np.float_)
        ell,sltt,slee,slbb,slte = utils.get_lensedcls(clfile,lmax=lmax)

        if est=='TT':
            self.sltt = sltt
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 = -0.5*np.ones_like(l)
            f2 = np.nan_to_num(np.sqrt(l*(l+1)))
            f3 = np.nan_to_num(np.sqrt(l*(l+1)))*sltt[:lmax+1]
            self.w[0][0]=f3; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=+1; self.s[0][1]=+0; self.s[0][2]=+1
            self.w[1][0]=f3; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-1; self.s[1][1]=+0; self.s[1][2]=-1
            self.w[2][0]=f1; self.w[2][1]=f3; self.w[2][2]=f2; self.s[2][0]=+0; self.s[2][1]=-1; self.s[2][2]=-1
            self.w[3][0]=f1; self.w[3][1]=f3; self.w[3][2]=f2; self.s[3][0]=+0; self.s[3][1]=+1; self.s[3][2]=+1
        
        if est=='EE':
            self.slee = slee
            self.ntrm = 8
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25*np.ones_like(l)
            f2 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f3 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            self.w[0][0]=f3; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f4; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f4; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f3; self.w[3][1]=f1; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f1; self.w[4][1]=f3; self.w[4][2]=f2; self.s[4][0]=-2; self.s[4][1]=+1; self.s[4][2]=-1
            self.w[5][0]=f1; self.w[5][1]=f4; self.w[5][2]=f2; self.s[5][0]=-2; self.s[5][1]=+3; self.s[5][2]=+1
            self.w[6][0]=f1; self.w[6][1]=f4; self.w[6][2]=f2; self.s[6][0]=+2; self.s[6][1]=-3; self.s[6][2]=-1
            self.w[7][0]=f1; self.w[7][1]=f3; self.w[7][2]=f2; self.s[7][0]=+2; self.s[7][1]=-1; self.s[7][2]=+1
            
        if est=='TE':
            self.slte = slte
            self.ntrm = 6
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            f5 = -0.50*np.ones_like(l,dtype=np.float_)
            f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*slte[:lmax+1]
            self.w[0][0]=f3; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f4; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f4; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f3; self.w[3][1]=f1; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f5; self.w[4][1]=f7; self.w[4][2]=f6; self.s[4][0]=+0; self.s[4][1]=-1; self.s[4][2]=-1
            self.w[5][0]=f5; self.w[5][1]=f7; self.w[5][2]=f6; self.s[5][0]=+0; self.s[5][1]=+1; self.s[5][2]=+1

        if est=='TB':
            self.slte = slte
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            f5 = +0.25j*np.ones_like(l,dtype=np.float_)
            #self.w[0][0]=f4; self.w[0][1]=f5; self.w[0][2]=f2; self.s[0][0]=+3; self.s[0][1]=-2; self.s[0][2]=+1 #from QL
            #self.w[1][0]=f4; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            #self.w[2][0]=f3; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=-1; self.s[2][1]=+2; self.s[2][2]=+1
            #self.w[3][0]=f3; self.w[3][1]=f5; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[0][0]=f3; self.w[0][1]=f5; self.w[0][2]=f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1 # self derived
            self.w[1][0]=f4; self.w[1][1]=f5; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f4; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f3; self.w[3][1]=f1; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1

        if est=='BT':
            self.slte = slte
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = +0.25*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            f5 = -0.25*np.ones_like(l,dtype=np.float_)
            self.w[0][1]=f4; self.w[0][0]=f1; self.w[0][2]=f2; self.s[0][1]=+3; self.s[0][0]=-2; self.s[0][2]=+1
            self.w[1][1]=f4; self.w[1][0]=f5; self.w[1][2]=f2; self.s[1][1]=-3; self.s[1][0]=+2; self.s[1][2]=-1
            self.w[2][1]=f3; self.w[2][0]=f5; self.w[2][2]=f2; self.s[2][1]=-1; self.s[2][0]=+2; self.s[2][2]=+1
            self.w[3][1]=f3; self.w[3][0]=f1; self.w[3][2]=f2; self.s[3][1]=+1; self.s[3][0]=-2; self.s[3][2]=-1

        if est=='EB':
            self.slee = slee
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 =  -0.25j*np.ones_like(l)
            f2 =  +0.25j*np.ones_like(l)
            f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            f5 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            self.w[0][0]=f4; self.w[0][1]=f1; self.w[0][2]=f3; self.s[0][0]=+1; self.s[0][1]=-2; self.s[0][2]=-1
            self.w[1][0]=f5; self.w[1][1]=f1; self.w[1][2]=f3; self.s[1][0]=+3; self.s[1][1]=-2; self.s[1][2]=+1
            self.w[2][0]=f5; self.w[2][1]=f2; self.w[2][2]=f3; self.s[2][0]=-3; self.s[2][1]=+2; self.s[2][2]=-1
            self.w[3][0]=f4; self.w[3][1]=f2; self.w[3][2]=f3; self.s[3][0]=-1; self.s[3][1]=+2; self.s[3][2]=+1
        '''
        if est=='EB':
            self.slee = slee
            self.ntrm = 6
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 =  (+1/(4j)*np.ones_like(l))
            f2 =  (+1/(4j)*np.ones_like(l))
            f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            f5 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slbb[:lmax+1]
            f7 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slbb[:lmax+1]         
            self.w[0][0]=f4; self.w[0][1]=f1; self.w[0][2]=f3; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f5; self.w[1][1]=f1; self.w[1][2]=f3; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f5; self.w[2][1]=f2; self.w[2][2]=f3; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f4; self.w[3][1]=f2; self.w[3][2]=f3; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f6; self.w[4][1]=f2; self.w[4][2]=f3; self.s[4][0]=-2; self.s[4][1]=+1; self.s[4][2]=-1
            self.w[5][0]=f7; self.w[5][1]=f2; self.w[5][2]=f3; self.s[5][0]=-2; self.s[5][1]=+3; self.s[5][2]=+1
            self.w[6][0]=f7; self.w[6][1]=f1; self.w[6][2]=f3; self.s[6][0]=+2; self.s[6][1]=-3; self.s[6][2]=-1
            self.w[7][0]=f6; self.w[7][1]=f1; self.w[7][2]=f3; self.s[7][0]=+2; self.s[7][1]=-1; self.s[7][2]=+1
        '''

'''
def weights_TT(idx,sltt,lmax):
    f1 = -0.5*np.ones_like(l)
    f2 = np.sqrt(l*(l+1))
    f3 = np.sqrt( l*(l+1) )*sltt[:lmax+1]; f3[:3]=0
    if idx==0: w1=f3; w2=f1; wL=f2; s1=+1; s2=+0; sL=+1
    if idx==1: w1=f3; w2=f1; wL=f2; s1=-1; s2=+0; sL=-1
    if idx==2: w1=f3; w2=f1; wL=f2; s1=+0; s2=-1; sL=-1
    if idx==3: w1=f3; w2=f1; wL=f2; s1=+0; s2=+1; sL=+1
    return w1,w2,wL,s1,s2,sL
    
def weights_EE(idx,slee,lmax):
    l  = np.arange(lmax+1,dtype=np.float_)
    f1 = -0.25*np.ones_like(l)
    f2 = +np.sqrt(l*(l+1))
    f3 = +np.sqrt((l+2.)*(l-1.))*slee[:lmax+1]; f3[:3]=0
    f4 = -np.sqrt((l+3.)*(l-2.))*slee[:lmax+1]; f4[:3]=0
    if idx==0: w1=f3; w2=f1; wL=f2; s1=-1; s2=+2; sL=+1
    if idx==1: w1=f4; w2=f1; wL=f2; s1=-3; s2=+2; sL=-1
    if idx==2: w1=f4; w2=f1; wL=f2; s1=+3; s2=-2; sL=+1
    if idx==3: w1=f3; w2=f1; wL=f2; s1=+1; s2=-2; sL=-1
    if idx==4: w1=f1; w2=f3; wL=f2; s1=-2; s2=+1; sL=-1
    if idx==5: w1=f1; w2=f4; wL=f2; s1=-2; s2=+3; sL=+1
    if idx==6: w1=f1; w2=f4; wL=f2; s1=+2; s2=-3; sL=-1
    if idx==7: w1=f1; w2=f3; wL=f2; s1=+2; s2=-1; sL=+1
    return w1,w2,wL,s1,s2,sL

def weights_TE(idx,slte,lmax):
    l  =  np.arange(lmax+1,dtype=np.float_)
    f1 = -0.25*np.ones_like(l)
    f2 =  np.sqrt(l*(l+1))
    f3 =  np.sqrt((l+2.)*(l-1.))*slte[:lmax+1]; f3[:3]=0
    f4 = -np.sqrt((l+3.)*(l-2.))*slte[:lmax+1]; f4[:3]=0
    f5 = -0.5*np.ones_like(l,dtype=np.float_)
    f6 =  np.sqrt(l*(l+1))
    f7 =  np.sqrt(l*(l+1))*slte[:lmax+1]
    if idx==0: w1=f3; w2=f1; wL=f2; s1=-1; s2=+2; sL=+1
    if idx==1: w1=f4; w2=f1; wL=f2; s1=-3; s2=+2; sL=+1
    if idx==2: w1=f4; w2=f1; wL=f2; s1=+3; s2=-2; sL=+1
    if idx==3: w1=f3; w2=f1; wL=f2; s1=+1; s2=-2; sL=-1
    if idx==4: w1=f5; w2=f7; wL=f6; s1=+0; s2=-1; sL=-1
    if idx==5: w1=f5; w2=f7; wL=f6; s1=+0; s2=+1; sL=+1
    return w1,w2,wL,s1,s2,sL
''';    
