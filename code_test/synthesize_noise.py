import os, sys, argparse, cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.datahandler.denoise_dataset import get_dataset_object

# CBSD432_15 = get_dataset_object('BSD432')(add_noise='gau-15.', mask=None, aug=None, crop_size=None)
# CBSD432_15.save_all_image('./dataset/BSD432', clean=True)

CBSD432_25 = get_dataset_object('BSD432')(add_noise='struc_gau-25.', mask=None, aug=None, crop_size=None)
CBSD432_25.save_all_image('./dataset/synthesized_BSD/BSD432_25_s', syn_noisy=True)

# CBSD432_50 = get_dataset_object('CBSD432')(add_noise='gau-50.', mask=None, aug=None, crop_size=None)
# CBSD432_50.save_all_image('./dataset/synthesized_BSD/CBSD432_50', syn_noisy=True)

# CBSD68_15 = get_dataset_object('CBSD68')(add_noise='gau-15.', mask=None, aug=None, crop_size=None)
# CBSD68_15.save_all_image('./dataset/synthesized_BSD/CBSD68_15', syn_noisy=True)

# CBSD68_25 = get_dataset_object('CBSD68')(add_noise='gau-25.', mask=None, aug=None, crop_size=None)
# CBSD68_25.save_all_image('./dataset/synthesized_BSD/CBSD68_25', syn_noisy=True)

# CBSD68_50 = get_dataset_object('CBSD68')(add_noise='gau-50.', mask=None, aug=None, crop_size=None)
# CBSD68_50.save_all_image('./dataset/synthesized_BSD/CBSD68_50', syn_noisy=True)
