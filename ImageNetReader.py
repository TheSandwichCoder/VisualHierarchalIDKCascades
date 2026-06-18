"""
This code is taken from the github repo for using ImageNetV2 for pytorch
https://github.com/modestyachts/ImageNetV2_pytorch/blob/master/imagenetv2_pytorch/ImageNetV2_dataset.py
"""

import pathlib
import tarfile
import zipfile
import requests
import shutil

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

URLS = {"matched-frequency" : "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
        "threshold-0.7" : "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz",
        "top-images": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz",
        "val": "https://imagenet2val.s3.amazonaws.com/imagenet_validation.tar.gz"}

FNAMES = {"matched-frequency" : "imagenetv2-matched-frequency-format-val",
        "threshold-0.7" : "imagenetv2-threshold0.7-format-val",
        "top-images": "imagenetv2-top-images-format-val",
        "val": "imagenet_validation"}


V2_DATASET_SIZE = 10000
VAL_DATASET_SIZE = 50000

class ImageNetValDataset(Dataset):
    def __init__(self, transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/imagenet_validation/")
        self.tar_root = pathlib.Path(f"{location}/imagenet_validation.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.JPEG"))
        self.transform = transform
        if not self.dataset_root.exists() or len(self.fnames) != VAL_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f"Dataset imagenet-val not found on disk, downloading....")
                response = requests.get(URLS["val"], stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f"Downloading from {URLS[variant]} failed"
            print("Extracting....")
            tarfile.open(self.tar_root).extractall(f"{location}")
            shutil.move(f"{location}/{FNAMES['val']}", self.dataset_root)

        self.dataset = ImageFolder(self.dataset_root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img, label = self.dataset[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class ImageNetV2Dataset(Dataset):
    def __init__(self, variant="matched-frequency", transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/ImageNetV2-{variant}/")
        self.tar_root = pathlib.Path(f"{location}/ImageNetV2-{variant}.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        self.transform = transform
        assert variant in URLS, f"unknown V2 Variant: {variant}"
        if not self.dataset_root.exists() or len(self.fnames) != V2_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f"Dataset {variant} not found on disk, downloading....")
                response = requests.get(URLS[variant], stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f"Downloading   from {URLS[variant]} failed"
            print("Extracting....")
            tarfile.open(self.tar_root).extractall(f"{location}")
            shutil.move(f"{location}/{FNAMES[variant]}", self.dataset_root)
            self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = img.convert("RGB")
            img = self.transform(img)   
        return img, label

class ImageNetSubset(Dataset):
    def __init__(self, transform=None, location=".", name="imagenet_subtrain", normalize_class_names=True):
        self.location = pathlib.Path(location)
        self.dataset_root = self.location / name
        self.zip_root = self.location / f"{name}.zip"
        self.transform = transform
        self.normalize_class_names = normalize_class_names
        self.fnames = self._find_images()

        if not self.dataset_root.exists() or not self.fnames:
            if not self.zip_root.exists():
                raise FileNotFoundError(
                    f"Dataset not found. Expected either {self.dataset_root} "
                    f"or {self.zip_root}"
                )
                
            print("Extracting....")
            with zipfile.ZipFile(self.zip_root, "r") as archive:
                archive.extractall(self.dataset_root)

            if self.normalize_class_names:
                self._normalize_class_folders()

            self.fnames = self._find_images()
            if not self.fnames:
                raise FileNotFoundError(
                    f"No image files were found under {self.dataset_root} "
                    f"after extracting {self.zip_root}"
                )
        elif self.normalize_class_names:
            self._normalize_class_folders()
            self.fnames = self._find_images()
        
    def _find_images(self):
        image_paths = []
        for extension in ("*.jpeg", "*.jpg", "*.JPEG", "*.JPG"):
            image_paths.extend(self.dataset_root.glob(f"**/{extension}"))
        return sorted(image_paths)

    def _normalize_class_folders(self):
        class_dirs = sorted({path.parent for path in self._find_images()}, key=self._class_sort_key)
        if not class_dirs:
            return

        expected_names = {str(index) for index in range(len(class_dirs))}
        current_names = {path.name for path in class_dirs}
        if current_names == expected_names:
            return

        temp_dirs = []
        for index, class_dir in enumerate(class_dirs):
            temp_dir = class_dir.with_name(f".__imagenet_tmp_{index:04d}")
            if temp_dir.exists():
                raise FileExistsError(f"Temporary rename path already exists: {temp_dir}")
            class_dir.rename(temp_dir)
            temp_dirs.append(temp_dir)

        for index, temp_dir in enumerate(temp_dirs):
            target_dir = temp_dir.with_name(str(index))
            if target_dir.exists():
                raise FileExistsError(f"Target class folder already exists: {target_dir}")
            temp_dir.rename(target_dir)

    def _class_sort_key(self, path):
        if path.name.isdigit():
            return (0, int(path.name))
        return (1, path.name.lower())

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = img.convert("RGB")
            img = self.transform(img)   
        return img, label


