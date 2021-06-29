import time

import tqdm


pbar = tqdm.tqdm(total=1000, ncols=100)

for i in range(1000):
    time.sleep(0.1)
    pbar.update()
