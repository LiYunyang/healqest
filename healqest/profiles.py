'''
Profile class inspired by pyccl. More flexible than just feeding 
in a txt file with ell/b(ell), and can use all sorts of profiles.
'''

import numpy as np


class Profile(object):
    '''Profile to null out for profile hardening'''

    def __init__(self):
        self.ells = np.arange(self.lmax+1)

    def fourier(self):
        if getattr(self, '_fourier', None):
            f_k = self._fourier()
        return f_k

class profileGaussian(Profile):
    '''
    Returns Gaussian profile in harmonic space
    
    Parameters
    ----------
    fwhm : float
      Full width half max in arcmins.
    lmax: int, scalar, optional
      Maximum l of the power spectrum. Default: 12000
      
    Returns
    -------
    p(ell) : array
      profile as a function fo ell.
      
    Examples
    --------
    >>> p_l = profileGaussian(1.2).fourier()
    array([1.        , 0.99999998, 0.99999993, ..., 0.20564428, 0.20559007,
       0.20553587])
    '''

    name = 'profileGaussian'

    def __init__(self, fwhm, lmax=12000):
        self.fwhm_rad = fwhm*0.00029088
        self.lmax        = lmax
        
        super(profileGaussian, self).__init__()

    def _real(self):
        '''Compute realspace and then convert to Fourier'''
        # To be implemented

    def _fourier(self):
        '''Compute Fourier space directly'''
        sigma = self.fwhm_rad/(np.sqrt(8*np.log(2)))
        return np.exp(-0.5*self.ells*(self.ells+1) * sigma**2)

