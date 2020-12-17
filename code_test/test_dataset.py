import os, sys, argparse, cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.dataset.denoise_dataset import get_dataset_object

train_dataset = get_dataset_object('CBSD432')(crop_size=[64, 64], add_noise='gau-15.', n_repeat=1)
data = train_dataset.__getitem__(1)
cv2.imwrite('clean.png', cv2.cvtColor(data['clean'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
cv2.imwrite('noisy.png', cv2.cvtColor(data['syn_noisy'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
