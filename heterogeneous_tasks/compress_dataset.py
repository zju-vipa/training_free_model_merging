from torch.utils.data import DataLoader, Dataset
import os
import skimage
from tqdm import tqdm
import matplotlib.pyplot as plt
from eval_utils import load_raw_image, resize_image, taskonomy_data_dir
import os
import numpy as np
import cv2

new_dims = (256, 256)


class CompressDataset(Dataset):
    """
    Compress all data to 256*256.
    """

    def __init__(self, root, task) -> None:
        super().__init__()
        self.task_root = os.path.join(root, task)
        self.files = os.listdir(self.task_root)
        self.files_path = [os.path.join(self.task_root, p) for p in self.files]
        task_handle_dict = {
            "rgb": self.handle_color_uint8,
            "normal": self.handle_color_uint8,
            'keypoints2d': self.handle_gray_uint16,
            'keypoints3d': self.handle_gray_uint16,
            'reshading': self.handle_gray_float,
            'edge_texture': self.handle_gray_uint16,
            'edge_occlusion': self.handle_gray_uint16,
        }

        self.task_handle = task_handle_dict[task]

    def handle_color_uint8(self, path):
        img = load_raw_image(path, color=True)
        if img.shape[0] == new_dims[0]:
            return
        img = skimage.img_as_float(img)
        img = resize_image(img, new_dims, interp_order=1)
        skimage.io.imsave(path, skimage.img_as_ubyte(img))

    def handle_gray_uint16(self, path):
        img = load_raw_image(path, color=False)
        if img.shape[0] == new_dims[0]:
            return
        img = skimage.img_as_float(img)
        img = resize_image(img, new_dims, interp_order=1)
        cv2.imwrite(path, (img * 65535).astype(np.uint16))

    def handle_gray_float(self, path):
        img = load_raw_image(path, color=False)
        if img.shape[0] == new_dims[0]:
            return
        img = skimage.img_as_float(img)
        img = resize_image(img, new_dims, interp_order=1)
        cv2.imwrite("test.png", (img * 255).repeat(3, axis=2))

    def __getitem__(self, index):
        try:
            self.task_handle(self.files_path[index])
        except Exception as e:
            print(e)
        return 0

    def __len__(self):
        return len(self.files)


domains = ["muleshoe", "ihlen", "mcdade", "noxapater", "uvalda"]

rbg_domains = [
    "allensville", "beechwood", "benevolence", "coffeen", "cosmos", "forkland",
    "hanson", "hiteman"
]
root = taskonomy_data_dir


for i, d in enumerate(domains + rbg_domains):
    print(f"current domain {d} {i}/{len(domains) + len(rbg_domains)}")
    ds = CompressDataset(os.path.join(root, d), "rgb")
    dl = DataLoader(ds, 100, shuffle=False, num_workers=8)

    for _ in tqdm(dl, desc=f"domain {d} task rgb", total=len(dl)):
        pass

for t in [
        "normal",
        'keypoints2d',
        'keypoints3d',
]:
    for i, d in enumerate(domains):
        print(f"current task {t} current domain {d} {i}/{len(domains)}")
        ds = CompressDataset(os.path.join(root, d), t)
        dl = DataLoader(ds, 100, shuffle=False, num_workers=8)

        for _ in tqdm(dl, desc=f"domain {d} task {t}", total=len(dl)):
            pass

for t in [
        'reshading',
        'edge_texture',
        'edge_occlusion',
]:
    for i, d in enumerate(domains):
        print(f"current task {t} current domain {d} {i}/{len(domains)}")
        ds = CompressDataset(os.path.join(root, d), t)
        dl = DataLoader(ds, 100, shuffle=False, num_workers=8)

        for _ in tqdm(dl, desc=f"domain {d} task {t}", total=len(dl)):
            pass