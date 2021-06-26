import os, sys, argparse, cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

from src.datahandler import get_dataset_object
from src.util.util import pixel_shuffle_up_sampling

# train_dataset = get_dataset_object('PD4_CBSD68_50')()
train_dataset = get_dataset_object('prep_SIDD')()
# train_dataset = get_dataset_object('CBSD68')(add_noise='gau-25.', multiple_cliping=4)

#data = train_dataset.__getitem__(166)
# if data['masked'].shape[0] == 3:
#cv2.imwrite('clean.png', cv2.cvtColor(data['clean'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
# cv2.imwrite('synthetic_noisy.png', cv2.cvtColor(data['syn_noisy'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
#cv2.imwrite('real_noisy.png', cv2.cvtColor(data['real_noisy'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
#t = pixel_shuffle_up_sampling(data['real_noisy'], 2)
#cv2.imwrite('real_noisy2.png', cv2.cvtColor(t.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
    #cv2.imwrite('masked.png', cv2.cvtColor(data['masked'].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
# else:
    #cv2.imwrite('clean.png', data['clean'].squeeze().numpy())
    #cv2.imwrite('synthetic_noisy.png', data['syn_noisy'].squeeze().numpy())
    #cv2.imwrite('real_noisy.png', data['real_noisy'].squeeze().numpy())
    #cv2.imwrite('masked.png', data['masked'].squeeze().numpy())
    #cv2.imwrite('mask.png', 255*data['mask'].squeeze().numpy())
    #cv2.imwrite('diff.png', (data['masked'].squeeze().numpy()-data['syn_noisy'].squeeze().numpy()))

# data = train_dataset.__getitem__(1)
# cv2.imwrite('synthetic_noisy1.png', data['syn_noisy'].squeeze().numpy())
# data = train_dataset.__getitem__(1)
# cv2.imwrite('synthetic_noisy2.png', data['syn_noisy'].squeeze().numpy())
# data = train_dataset.__getitem__(1)
# cv2.imwrite('synthetic_noisy3.png', data['syn_noisy'].squeeze().numpy())

# train_dataset = get_dataset_object('DND')()
# train_dataset.prep_save(img_size=512, overlap=256, real_noisy=True)

for i in range(train_dataset.__len__()):
    data = train_dataset.__getitem__(i)
    print('%06d : '%i, data['real_noisy'].mean())