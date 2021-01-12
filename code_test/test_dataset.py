import os, sys, argparse, cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.datahandler.denoise_dataset import get_dataset_object

train_dataset = get_dataset_object('SIDD')(add_noise='gau-15.', mask='rnd_32-rnd', aug=['hflip', 'rot'], crop_size=None, n_repeat=1)
data = train_dataset.__getitem__(1)
cv2.imwrite('clean.png', cv2.cvtColor(data['clean'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
cv2.imwrite('synthetic_noisy.png', cv2.cvtColor(data['syn_noisy'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
cv2.imwrite('real_noisy.png', cv2.cvtColor(data['real_noisy'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
cv2.imwrite('masked.png', cv2.cvtColor(data['masked'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
print(data['data_aug'])
