import healpy as hp
from tabulate import tabulate
from healqest import healqest_utils as hutils

class maps:
    def __init__(self, sim_dict, lmax = None):
        '''
        sim_dict: dictionary containing settings for cinv filtering 
        '''
        if ~all(name in sim_dict for name in ['nside','file_map','file_mask']):
            print('some keys missing in sim_dict')

        self.nside      = sim_dict['nside']
        #self.file_noise = sim_dict['file_noise']
        self.file_cmb   = sim_dict['file_signal']
        #self.file_mask  = sim_dict['file_mask']

        table = [["nside"     ,self.nside],
                 ["file_dmap" ,self.file_cmb],
                 #["file_noise",self.file_cmb],
                 #["file_mask" ,self.file_mask]
                 ]
        print(tabulate(table))

        #sys.exit()
        
    
    def hashdict(self):
        return {'cmbs':self.file_cmb, 'noise':self.file_noise}
    
    def change_alm_lmax(self, alm, lmax):
        alm = hutils.reduce_lmax(alm,lmax=lmax)
        return alm
    
    def get_mask(self):
        return hp.read_map(self.file_mask)

    def get_tmap(self,seed,add_noise=True,apply_tf=False):
        '''Load sim temperature signal and noise map separately and add'''

        print("loading %s"%(self.file_cmb.format(seed=seed)))
        almt = hp.read_alm(self.file_cmb.format(seed=seed), hdu=1)

        if add_noise:
            print('Adding noise')
            print("loading %s"%(self.noise.format(seed=seed)))
            nlmt  = hp.read_alm(self.noise.format(seed=seed), hdu=1)
            almt += nlmt
            del nlmt
        else:
            print('not adding noise')

        if apply_tf:
            print('Applying tf')
            assert tf2d is not None
            lmax   = hp.Alm.getlmax(len(self.tf2d))
            lmaxin = hp.Alm.getlmax(len(almt))
            if lmaxin > lmax: almt = self.change_alm_lmax(almt,lmax)
            almt *= self.tf2d
        else:
            print('Not applying tf')

        return hp.alm2map(almt, self.nside)

    def get_pmap(self,seed,add_noise=True,apply_tf=False):
        '''Load sim polarization signal and noise map separately and add'''

        print("loading %s"%(self.file_cmb.format(seed=seed)))
        alme,almb = hp.read_alm(self.file_cmb.format(seed=seed), hdu=[2,3])
        lmaxin    =  hp.Alm.getlmax(len(alme))

        if add_noise:
            print('Adding noise')
            print("loading %s"%(self.file_noise.format(seed=seed)))
            nlme,nlmb = hp.read_alm(self.file_noise.format(seed=seed), hdu=[2,3])
            alme += nlme
            almb += nlmb
            del nlme, nlmb
        else:
            print('Not adding noise')

        if apply_tf:
            print('Applying tf')
            assert tf2d is not None
            lmax   = hp.Alm.getlmax(len(self.tf2d))
            lmaxin = hp.Alm.getlmax(len(alme))
            if lmaxin > lmax: alme = self.change_alm_lmax(alme,lmax)
            if lmaxin > lmax: almb = self.change_alm_lmax(almb,lmax)
            alme *= self.tf2d
            almb *= self.tf2d
        else:
            print('Not applying tf')

        return hp.alm2map_spin([alme, almb], self.nside, spin=2, lmax=lmaxin)
