import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.dataset.custom_dataset import CustomDataSet


train_dataset = CustomDataSet(name="CBSD432")
data = train_dataset.__getitem__(0)
print(data['img'])