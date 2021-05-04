import os, sys, argparse, cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.datahandler import get_dataset_object

'''
sythesize AWGN on the BSD datset
'''
# CBSD432_15 = get_dataset_object('BSD432')(add_noise='gau-15.', mask=None, aug=None, crop_size=None)
# CBSD432_15.save_all_image('./dataset/BSD432', clean=True)

# CBSD432_25 = get_dataset_object('BSD432')(add_noise='struc_gau-25.', mask=None, aug=None, crop_size=None)
# CBSD432_25.save_all_image('./dataset/synthesized_BSD/BSD432_25_s', syn_noisy=True)

# CBSD432_50 = get_dataset_object('CBSD432')(add_noise='gau-50.', mask=None, aug=None, crop_size=None)
# CBSD432_50.save_all_image('./dataset/synthesized_BSD/CBSD432_50', syn_noisy=True)

# CBSD68_15 = get_dataset_object('CBSD68')(add_noise='gau-15.', mask=None, aug=None, crop_size=None)
# CBSD68_15.save_all_image('./dataset/synthesized_BSD/CBSD68_15', syn_noisy=True)

# CBSD68_25 = get_dataset_object('CBSD68')(add_noise='gau-25.', mask=None, aug=None, crop_size=None)
# CBSD68_25.save_all_image('./dataset/synthesized_BSD/CBSD68_25', syn_noisy=True)

# CBSD68_50 = get_dataset_object('CBSD68')(add_noise='gau-50.', mask=None, aug=None, crop_size=None)
# CBSD68_50.save_all_image('./dataset/synthesized_BSD/CBSD68_50', syn_noisy=True)

'''
pixel-shuffle down-sampling on AWGN BSD dataset
'''
clipping_number = 12

# for ds in ['CBSD68', 'CBSD432']:
#     for nlf in [15, 25, 50]:
#         for pd in [1,2,3,4]:
for ds in ['CBSD68']:
    for nlf in [25]:
        for pd in [1,2,3,4]:
            dataset = get_dataset_object(ds)(add_noise='gau-%d'%nlf, mask=None, aug=None, crop_size=None, multiple_cliping=clipping_number, pd=pd)
            dataset.save_all_image('./dataset/synthesized_PD_BSD/pd%d/%s_%d/C'%(pd, ds, nlf), clean=True)
            dataset.save_all_image('./dataset/synthesized_PD_BSD/pd%d/%s_%d/N'%(pd, ds, nlf), syn_noisy=True)
