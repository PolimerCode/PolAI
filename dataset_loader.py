import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MinecraftDataset(Dataset):
    def __init__(self, images_dir, actions_dir, transform=None):
        self.images_dir = images_dir
        self.actions_dir = actions_dir
        self.transform = transform

        self.image_files = sorted(os.listdir(images_dir))
        self.action_files = sorted(os.listdir(actions_dir))
        assert len(self.image_files) == len(self.action_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        act_path = os.path.join(self.actions_dir, self.action_files[idx])

        img = cv2.imread(img_path)
        img = cv2.resize(img, (160, 120))  # resize for speed increasement
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        img = torch.tensor(img).permute(2, 0, 1)  # HWC -> CHW
        action = torch.tensor(np.load(act_path), dtype=torch.float32)

        return img, action
