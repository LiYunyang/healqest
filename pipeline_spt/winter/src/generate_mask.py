import healpy as hp
import numpy as np
import argparse
import pandas

parser = argparse.ArgumentParser()
parser.add_argument('file_yaml'  , default=None, type=str, help='yaml file with all the configurations')
args = parser.parse_args()

p       = yaml.safe_load(open(args.file_yaml))

dir_mask = p['masks']['dir_mask']
pcut     = p['masks']['pthres']
ccut     = p['masks']['cthres']

fluxcut  = pcut

# CIB and radio maps to construct masks from
dir_mdpl = '/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.7/'
cibfile  = dir_mdpl+'/cib/cib_bestfit_081322_run2_200_3000_nocc/len/spt3g_150/param0/mdpl2_len_mag_cibmap_spt3g_150.fits'
radfile  = dir_mdpl+'/rad/len/mdpl2_radiomap_len_universemachine_trinity_150ghz_randflux_datta2018_truncgauss.0.fits'

# Read maps
cib      = hp.read_map(cibfile,field=0)
rad      = hp.read_map(radfile,field=0)
cibflux  = hp.ud_grade(cib*hp.nside2pixarea(nside)*1000,nside_out=4096,power=-2)
radflux  = hp.ud_grade(rad*hp.nside2pixarea(nside)*1000,nside_out=4096,power=-2)

# Want to create a mask from the total flux map
totflux  = cibflux + radflux
del cibflux,radflux

fc      =  np.where(totflux>fluxcut)[0]

mask     = np.ones(hp.nside2npix(4096))
mask[fc] = 0
mask     = hp.ud_grade(mask,nside_out=8192)

# save ptsrc mask
mask    = hp.ud_grade(mask,nside_out=nside)

hp.write_map(dir_mask+'/mask%d_mdpl2_lenmag_cibmap_radmap_%s_%s_fluxcut%.1fmjy_singlepix.fits'%(nside,expt,freqname,fluxcut),mask,overwrite=True,dtype=np.float32)

#----------------------------------------------------------------------------------------------

######################## cluster mask #############################
''' Create ymask from looking at the Compton-y map'''

fsky         =  0.03636 # fsky from Lindsey
file_ymaxima = '/lcrc/project/SPT3G/users/ac.yomori/sims/mdpl2/v0.7/tsz/len/bahamas80/maxima_mdpl2_ltszNG_bahamas80_rot_sum_4_176_bnd_unb_1.0e+12_1.0e+18_v103021_lmax24000_nside8192_interp1.0_method1_1_lensed_map.npy'
file_cluscat = '/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software/simulations/python/data/foregrounds/spt3g_1500d_cluster_list_lb_Feb17_2020.txt'
clus_sigma   = ccut
capod        = 3

nside = 8192
maxima = np.load(file_ymaxima)

# Precut so that sorting is faster
idx    = np.where(maxima[:,1]>1e-5)[0]
maxima = maxima[idx,:]

# Sort from largest -> lowest Compton-y
maxima = maxima[maxima[:,1].argsort()[::-1]]

# Compare with the number density with 3g catalog
# simulations/python/data/foregrounds/spt3g_1500d_cluster_list_lb_Feb17_2020.txt
# May need to tweak in the furture depending on the file used

######### cluster mask ##########
tmp      = pandas.read_csv('/lcrc/project/SPT3G/users/ac.yomori/repo/spt3g_software/sources/python/main_lists/SPT3G_Nov422_masked_run_plus_SPTSZ_ACT_unmasked.txt',delimiter='\t',skiprows=25)
clist1 = {}
clist1['index'] = tmp['index'].values
clist1['conf']  = tmp['SPT_CONFIDENCE'].values # SPT confidence
clist1['ra']    = tmp['RA(deg)'].values
clist1['dec']   = tmp['DEC(deg)'].values
clist1['size']  = tmp['SPT_SIZE(arcmin)'].values
clist1['xcoin'] = tmp['XRAY_COINCIDENCE'].values # Xray coincidens
clist1['masksn']= tmp['MASKSN'].values

Nclus_3g     = len(np.where(clist1['conf']>clus_sigma)[0])
print("Basing clusters on %d clusters"%(Nclus_3g))

# Convert to fullsky number and pick the top X clusters
cat      = maxima[:int(Nclus_3g/fsky),:]
print("Number of clusters over fullsky: %d"%(int(Nclus_3g/fsky)))

sys.exit()
# Convert pixels to directions and apply masking
tht,phi  = hp.pix2ang(nside,np.int64(cat[:,0]) )

# Pick masking radius to use
R = np.zeros_like(tht)
R[cat[:,1]>1e-5]=4
R[cat[:,1]>1e-4]=8

# Generate mask
maskc = np.ones(hp.nside2npix(nside))
vec   = hp.ang2vec(tht,phi)

yy=np.copy(y)
for i in range(0,len(tht)):
    #print(i)

    f=1
    R=cat[i,1]*1e4

    while f>0.95:

        pixs=hp.query_disc(nside,vec[i],radius=(R)*0.000290888)

        f = len(np.where(yy[pixs]>cat[i,1]*0.05)[0])/len(pixs)
        R+=1
        print(f)


    maskc[pixs]=0



# Apodize mask
maskc_apod = utils.apodize_mask(process_mask,maskc,capod)

#Save masks
hp.write_map(dir_mask+'/mask%d_mdpl2_clusters_spt3g_%dsigma.fits'%(nside,clus_sigma),maskc,overwrite=True,dtype=np.float32)
hp.write_map(dir_mask+'/mask%d_mdpl2_clusters_spt3g_%dsigma_apod%d.fits'%(nside,clus_sigma,capod),maskc_apod,overwrite=True,dtype=np.float32)

