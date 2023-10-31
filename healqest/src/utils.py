import os,sys,git,uuid
import numpy as np
import healpy as hp
from pathlib import Path
import logging,yaml,pickle

def reduce_lmax(alm, lmax=4000):
    """
    Reduce the lmax of input alm
    """
    lmaxin  = hp.Alm.getlmax(alm.shape[0])
    print( "-- Reducing lmax: lmax_in=%g -> lmax_out=%g"%(lmaxin,lmax) )
    ell,emm = hp.Alm.getlm(lmaxin)
    almout  = np.zeros(hp.Alm.getsize(lmax),dtype=np.complex_)
    oldi=0
    oldf=0
    newi=0
    newf=0
    dl = lmaxin-lmax
    for i in range(0,lmax+1):
        oldf=oldi+lmaxin+1-i
        newf=newi+lmax+1-i
        almout[newi:newf]=alm[oldi:oldf-dl]
        oldi=oldf
        newi=newf
    return almout

def get_nside(lmax):
    """calculates the most appropriate nside based on lmax"""
    nside = np.array([8,16,32,64,128,256,512,1024,2048,4096,8192,16384])
    idx   = np.argmin(np.abs(nside-lmax))
    return nside[idx]

def zeropad(cl):
    """add zeros for L=0,1"""
    cl=np.insert(cl,0,0)
    cl=np.insert(cl,0,0)
    return cl

def get_lensedcls(file,lmax=2000,dict=False):
    ell,sltt,slee,slbb,slte=np.loadtxt(file,unpack=True)
    # Removing the ell factors and padding with zeros (since the file starts with l=2)
    sltt=sltt/ell/(ell+1)*2*np.pi; sltt=zeropad(sltt)
    slee=slee/ell/(ell+1)*2*np.pi; slee=zeropad(slee)
    slte=slte/ell/(ell+1)*2*np.pi; slte=zeropad(slte)
    slbb=slbb/ell/(ell+1)*2*np.pi; slbb=zeropad(slbb)
    ell=np.insert(ell,0,1); ell=np.insert(ell,0,0)
    ell  = ell[:lmax+1]
    sltt = sltt[:lmax+1]
    slee = slee[:lmax+1]
    slbb = slbb[:lmax+1]
    slte = slte[:lmax+1]
    if dict==False:
        return ell,sltt,slee,slbb,slte
    else:
        d={}
        d['tt']=sltt
        d['ee']=slee
        d['bb']=slbb
        d['te']=slte
        return d

def get_unlensedcls(file,lmax=2000):
    ell,sltt,slee,slbb,slte,slpp,sltp,slep=np.loadtxt(file,unpack=True)
    # Removing the ell factors and padding with zeros (since the file starts with l=2)
    sltt=sltt/ell/(ell+1)*2*np.pi;             sltt=zeropad(sltt)
    slee=slee/ell/(ell+1)*2*np.pi;             slee=zeropad(slee)
    slbb=slbb/ell/(ell+1)*2*np.pi;             slbb=zeropad(slbb)
    slte=slte/ell/(ell+1)*2*np.pi;             slte=zeropad(slte)
    slpp=slpp/ell/ell/(ell+1)/(ell+1)*2*np.pi; slpp=zeropad(slpp)
    sltp=sltp/(ell*(ell+1))**(1.5)*2*np.pi;    sltp=zeropad(sltp)
    slep=slep/(ell*(ell+1))**(1.5)*2*np.pi;    slep=zeropad(slep)
    ell=np.insert(ell,0,1); ell=np.insert(ell,0,0)
    ell  = ell[:lmax+1]
    sltt = sltt[:lmax+1]
    slee = slee[:lmax+1]
    slbb = slbb[:lmax+1]
    slte = slte[:lmax+1]
    slpp = slpp[:lmax+1]
    sltp = sltp[:lmax+1]
    slep = slep[:lmax+1]
    return ell,sltt,slee,slbb,slte,slpp,sltp,slep
 
def setup_logger(nolog,file_log='test.log'):

    dir_log = str(Path(file_log).parent)
    Path(dir_log).mkdir(parents=True, exist_ok=True)

    if nolog==True:
        logging.basicConfig(
                            format   = '[%(asctime)s] %(message)s',
                            datefmt  = '%H:%M:%S',
                            level    = logging.WARNING)
    else:
        logging.basicConfig(filename = file_log ,
                            filemode = 'w+',
                            format   = '[%(asctime)s] %(message)s',
                            datefmt  = '%H:%M:%S',
                            level    = logging.WARNING)

def add_clsdict(d,key,cltt,clee,clbb,clte=None):
    d[key]  = {}
    d[key]['tt'] = cltt
    d[key]['ee'] = clee
    d[key]['bb'] = clbb

    if clte is not None:
        d[key]['te'] = clte

    return d

#def get_fl(config,use_unlCls=False):
#    lmaxTP = config['lmaxTP']
#    if use_unlCls:
#        flT = 1.0/(config['cls']['ucmb']['tt'][:lmaxTP+1]+config['cls']['totres']['tt'][:lmaxTP+1])
#        flE = 1.0/(config['cls']['ucmb']['ee'][:lmaxTP+1]+config['cls']['totres']['ee'][:lmaxTP+1])
#        flB = 1.0/(config['cls']['ucmb']['bb'][:lmaxTP+1]+config['cls']['totres']['bb'][:lmaxTP+1])
#    else:
#        flT = 1.0/(config['cls']['lcmb']['tt'][:lmaxTP+1]+config['cls']['totres']['tt'][:lmaxTP+1])
#        flE = 1.0/(config['cls']['lcmb']['ee'][:lmaxTP+1]+config['cls']['totres']['ee'][:lmaxTP+1])
#        flB = 1.0/(config['cls']['lcmb']['bb'][:lmaxTP+1]+config['cls']['totres']['bb'][:lmaxTP+1])
#    flT[lmaxT+1:] = 0; flT[:lmin] = 0
#    flE[lmaxP+1:] = 0; flE[:lmin] = 0
#    flB[lmaxP+1:] = 0; flB[:lmin] = 0
#    return flT, flE, flB

#def get_almbar(qetype,alms1,alms2,config,use_unlCls=True):
#    if alms1.ndim == 1:
#        tlm1,tlm2 = alms1,alms2
#    else:
#        tlm1,elm1,blm1 = alms1[0],alms1[1],alms1[2]
#        tlm2,elm2,blm2 = alms2[0],alms2[1],alms2[2]
    
#    flT,flE,flB = get_fl(config,use_unlCls=use_unlCls)

#    lmaxTP = config['lmaxTP']
#    print('Preparing input almbars')
#    if qetype[0]=='T': tlm1 = reduce_lmax(tlm1,lmax=lmaxTP); almbar1 = hp.almxfl(tlm1,flT); flm1= flT
#    if qetype[0]=='E': elm1 = reduce_lmax(elm1,lmax=lmaxTP); almbar1 = hp.almxfl(elm1,flE); flm1= flE
#    if qetype[0]=='B': blm1 = reduce_lmax(blm1,lmax=lmaxTP); almbar1 = hp.almxfl(blm1,flB); flm1= flB
#    if qetype[1]=='T': tlm2 = reduce_lmax(tlm2,lmax=lmaxTP); almbar2 = hp.almxfl(tlm2,flT); flm2= flT
#    if qetype[1]=='E': elm2 = reduce_lmax(elm2,lmax=lmaxTP); almbar2 = hp.almxfl(elm2,flE); flm2= flE
#    if qetype[1]=='B': blm2 = reduce_lmax(blm2,lmax=lmaxTP); almbar2 = hp.almxfl(blm2,flB); flm2= flB
#    return almbar1,almbar2,flm1,flm2


def get_fl(config,mtype,use_unlCls=False):

    sdict  = {'T':'tt', 'E':'ee', 'B':'bb' }
    
    lmaxTP = config['lmaxTP']
    lmax   = config['lmax%s'%('T' if mtype=='T' else 'P')]
    lmin   = config['lmin']

    if use_unlCls:
        fl = 1.0/(config['cls']['ucmb'][sdict[mtype]][:lmaxTP+1]+config['cls']['totres'][sdict[mtype]][:lmaxTP+1])
    else:
        fl = 1.0/(config['cls']['lcmb'][sdict[mtype]][:lmaxTP+1]+config['cls']['totres'][sdict[mtype]][:lmaxTP+1])
    fl[:lmin] = 0
    if lmax < lmaxTP: fl[lmax:] = 0

    return fl

def get_almbar(config,mtype,cmbid,seed,use_unlCls=True):

    hdudict = {_mtype: c+1 for c,_mtype in enumerate(['T','E','B'])}

    lmaxTP  = config['lmaxTP']
    alm     = hp.read_alm(config['iqu']['dir']+config['iqu']['prefix'].format(cmbid=cmbid,seed=seed),hdu=hdudict[mtype]) 
    alm     = reduce_lmax(alm,lmax=lmaxTP);

    fl      = get_fl(config,mtype,use_unlCls=use_unlCls)

    almbar  = hp.almxfl(alm,fl)

    return almbar,fl

def parse_yaml(file_yaml):
    '''
    Loading all settinsg stored in the yaml and also
    setting certain dictionary keys based on it.
    
    Params
    
    Returns
    
    '''
    dict = yaml.safe_load(Path(file_yaml).read_text())
   
    repo    = git.Repo(dict['base']['dir_healqest'],search_parent_directories=True)
    sha     = repo.head.object.hexsha

    dir_out = dict['base']['dir_out']
    
    dict = dict['lensing']
    dict['healqest_githash'] = sha
    dict['dir_out']          = dir_out

    # ----- Setting lmax of T/P -----
    dict['lmaxTP']=max(dict['lmaxt'],dict['lmaxp'])

    runhash = uuid.uuid4().hex
    dict['runhash'] = runhash
    with open(dir_out+'config_%s.pkl'%runhash, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # ----- Check if we need pol -----
    #if all('TT' in qe for qe in dict['plm']['qes']):
    #    dict['need_pol']=0
    #else:
    #    dict['need_pol']=1

    # ----- Setting Cls ------
    if "cls" in dict:

        if 'file_lcmb' in dict['cls']:
            try:
                ell = np.loadtxt(dict['cls']['file_lcmb'], usecols=[0])
                dict['cls']['lcmb'] = {f: zeropad(np.loadtxt(dict['cls']['file_lcmb'], usecols=[c+1])/ell/(ell+1)*2*np.pi)[:dict['lmaxTP']+1] for c, f in enumerate(['tt','ee','bb','te']) }
                print("Setting CMB cls")    
            except:
                print("Couldn't load CMB cls -- not setting CMB Cls")
         
        if 'file_ucmb' in dict['cls']:
            try:
                ell = np.loadtxt(dict['cls']['file_ucmb'], usecols=[0])
                dict['cls']['ucmb'] = {f: zeropad(np.loadtxt(dict['cls']['file_ucmb'], usecols=[c+1])/ell/(ell+1)*2*np.pi)[:dict['lmaxTP']+1]  for c, f in enumerate(['tt','ee','bb','te']) }
                dict['cls']['ucmb']['pp'] = zeropad(np.loadtxt(dict['cls']['file_ucmb'], usecols=[5])/ell/ell/(ell+1)/(ell+1)*2*np.pi)[:dict['lmaxTP']+1]
                dict['cls']['ucmb']['tp'] = zeropad(np.loadtxt(dict['cls']['file_ucmb'], usecols=[6])/(ell*(ell+1))**(1.5)*2*np.pi)[:dict['lmaxTP']+1]
                dict['cls']['ucmb']['ep'] = zeropad(np.loadtxt(dict['cls']['file_ucmb'], usecols=[7])/(ell*(ell+1))**(1.5)*2*np.pi)[:dict['lmaxTP']+1]
                print("Setting CMB unlensed cls")
            except:
                print("Couldn't load CMB cls -- not setting CMB Cls")

        if 'file_gcmb' in dict['cls']:
            try:
                #TODO: This will change if we use a different file! Currently configured for Abhi's file
                ell = np.loadtxt(dict['cls']['file_gcmb'], usecols=[0])
                dict['cls']['gcmb'] = {f: zeropad(np.loadtxt(dict['cls']['file_gcmb'], usecols=[c+1])/ell/(ell+1)*2*np.pi)[:dict['lmaxTP']+1]  for c, f in enumerate(['tt','ee','bb']) }
                dict['cls']['gcmb']['te'] = zeropad(np.loadtxt(dict['cls']['file_gcmb'], usecols=[5])/ell/(ell+1)*2*np.pi)[:dict['lmaxTP']+1]
                print("Setting CMB gradient cls")
            except:
                print("Couldn't load CMB cls -- not setting CMB Cls")

        if 'file_noise' in dict['cls']:
            try:
                dict['cls']['noise'] = {f: np.loadtxt(dict['cls']['file_noise'], usecols=[c+1])[:dict['lmaxTP']+1]  for c, f in enumerate(['tt','ee','bb','te']) }
                print("Setting noise cls")
            except:
                print("Couldn't load noise cls -- not setting noise Cls")
                
        if 'file_foreground' in dict['cls']:
            try:
                dict['cls']['foreground'] = {f: np.loadtxt(dict['cls']['file_foreground'], usecols=[c+1])[:dict['lmaxTP']+1]  for c, f in enumerate(['tt','ee','bb','te']) }
                print("Setting foreground cls")
            except:
                print("Couldn't load foreground cls -- not setting foreground Cls")

        if (('file_foreground' in dict['cls']) and ('file_noise' in dict['cls'])):
            try:
                dict['cls']['totres'] = {f: np.loadtxt(dict['cls']['file_noise'], usecols=[c+1])[:dict['lmaxTP']+1] +np.loadtxt(dict['cls']['file_foreground'], usecols=[c+1])[:dict['lmaxTP']+1] for c, f in enumerate(['tt','ee','bb','te']) }
                print("Setting noise cls")
            except:
                print("Couldn't load noise cls -- not setting noise Cls")


    else:
        sys.exit("Need to provide cls")
        
    return dict
