import os, sys, argparse, cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from src.model.EBSN import EBSN

if __name__ == "__main__":
    model = EBSN()

    t = torch.randn((16,1,64,64))

    print(model(t).shape)