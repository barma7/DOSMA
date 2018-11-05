
from utils.dicom_utils import load_dicom
from scan_sequences.dess import Dess
from tissues.femoral_cartilage import FemoralCartilage
from utils.quant_vals import QuantitativeValues
from models.get_model import get_model
import matplotlib.pyplot as plt
from utils.io_utils import load_h5, load_nifti
from os.path import join
import numpy as np
from dit.divergences import jensen_shannon_divergence
from dit import Distribution
import pandas as pd
from scipy.stats import wasserstein_distance, kstest

#=============================== CONFIGURATIONS STEPS ==================================================================

repo_folder = '/bmrNAS/people/barma7/bilateral/dess_study/data/marco_study_copy/rep_07/'

dicom_path_l = repo_folder + '005/LEFT/'
dicom_path_r = repo_folder + '005/RIGHT/'

joint_save_folder =  repo_folder + 'data/005/manual/'
left_knee_save_folder = repo_folder + 'data/005/manual/LEFT/'
right_knee_save_folder = repo_folder + 'data/005/manual/RIGHT/'

t2_filepath_left = repo_folder + 'data/005/LEFT/dess_data/t2.nii.gz'
mask_filepath_left = repo_folder + 'data/005/LEFT/fc/fc_manual.nii.gz'

t2_filepath_right = repo_folder + 'data/005/RIGHT/dess_data/t2.nii.gz'
mask_filepath_right = repo_folder + 'data/005/RIGHT/fc/fc_manual.nii.gz'

sub_id = 7
scan_id = 1


#weights_dir = '/home/marco.b215/msk_pipeline_data/weights'


## inizializzare mappa t2 e femoral cartilage

t2_par = QuantitativeValues(2)

dess_l = Dess(dicom_path_l)
fc_l = FemoralCartilage()
fc_l.pid = sub_id
fc_l.scan = scan_id

dess_r = Dess(dicom_path_r)
fc_r = FemoralCartilage()
fc_r.pid = sub_id
fc_r.scan = scan_id
fc_r.medial_to_lateral = True

dess_l.t2map = load_nifti(t2_filepath_left)
dess_r.t2map = load_nifti(t2_filepath_right)

fc_l.__mask__ = load_nifti(mask_filepath_left)
fc_r.__mask__ = load_nifti(mask_filepath_right)

fc_l.calc_quant_vals(dess_l.t2map, t2_par)
fc_r.calc_quant_vals(dess_r.t2map, t2_par)

#save data

fc_l.__save_quant_data__(left_knee_save_folder)
fc_r.__save_quant_data__(right_knee_save_folder)


# Compute Histogram and KL divergence

seg_t2_l = np.multiply(fc_l.__mask__.volume, dess_l.t2map.volume)
seg_t2_r = np.multiply(fc_r.__mask__.volume, dess_r.t2map.volume)

seg_t2_l = seg_t2_l[seg_t2_l>0]
seg_t2_r = seg_t2_r[seg_t2_r>0]

fig = plt.figure()
ax = fig.add_subplot(111)
h_l = ax.hist(seg_t2_l.flatten(), 100, range = (0, 100), alpha=1, label='LEFT')
h_r = ax.hist(seg_t2_r.flatten(), 100, range = (0, 100), alpha=0.7, label='RIGHT')
ax.set_xlabel('T2 (ms)')
ax.set_ylabel('# COUNTS')
ax.legend()
plt.draw()
plt.savefig(join(joint_save_folder, 'T2_hist.pdf'), format='pdf', dpi=300)


# Compute Jensen-Shannon divergence

pmf_l = np.divide(h_l[0],np.sum(h_l[0]))
pmf_r = np.divide(h_r[0],np.sum(h_r[0]))


d_l = Distribution.from_ndarray(pmf_l)
d_r = Distribution.from_ndarray(pmf_r)

JSD = jensen_shannon_divergence([d_l, d_r])

pd_header = ['#counts', 'JS_Divergence']
pd_list = [[np.sum(h_l[0]), JSD],[np.sum(h_r[0]), JSD]]

df = pd.DataFrame(pd_list, columns=pd_header)

df.to_csv(join(joint_save_folder,'histogram_calculation.csv'))

