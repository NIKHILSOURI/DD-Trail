from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union
from natsort import natsorted
from glob import glob
import pickle

from transformers import AutoProcessor
def identity(x):
    return x
def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP
    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)
    v_split = normalize(v_split)
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img



#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'npy'# type: ignore

class eeg_pretrain_dataset(Dataset):
    def __init__(self, path=None, roi='VC', patch_size=16, transform=identity, aug_times=2, 
                num_sub_limit=None, include_kam=False, include_hcp=True, eeg_pth_path=None):
        super(eeg_pretrain_dataset, self).__init__()
        # Default to repo root if path not provided
        if path is None:
            _CODE_DIR = Path(__file__).parent.absolute()
            _REPO_ROOT = _CODE_DIR.parent.absolute()
            path = os.path.join(str(_REPO_ROOT), 'datasets', 'mne_data')
        data = []
        images = []
        self.input_paths = [str(f) for f in sorted(Path(path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]
        self._eeg_from_pth = None  # In-memory fallback from eeg_5_95_std.pth

        if len(self.input_paths) == 0:
            # Fallback: load from eeg_5_95_std.pth (ImageNet-EEG) when no MNE .npy files exist
            pth_path = eeg_pth_path or os.path.join(str(Path(path).parent), 'eeg_5_95_std.pth')
            if os.path.isfile(pth_path):
                loaded = torch.load(pth_path, map_location='cpu', weights_only=False)
                self._eeg_from_pth = [d['eeg'] for d in loaded['dataset']]
                self.input_paths = list(range(len(self._eeg_from_pth)))
                if len(self._eeg_from_pth) > 0:
                    print(f'[eeg_pretrain_dataset] No .npy files in {path}; using {len(self._eeg_from_pth)} samples from {pth_path}')
            if len(self.input_paths) == 0:
                raise AssertionError(
                    f'No data found. Expected either:\n'
                    f'  - .npy files in {path}\n'
                    f'  - or eeg_5_95_std.pth in {Path(path).parent}\n'
                    f'Set DREAMDIFFUSION_DATA_ROOT or place eeg_5_95_std.pth in datasets/.'
                )

        self.data_len  = 512
        self.data_chan = 128

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, index):
        if self._eeg_from_pth is not None:
            data = np.array(self._eeg_from_pth[index].float().numpy())
            # EEG in .pth is (channels, time); eeg_pretrain expects (channels, time)
            if data.ndim == 2 and data.shape[0] < data.shape[1]:
                data = data.T  # (time, channels) -> (channels, time)
            # Match EEGDataset: crop time axis to 440 points [20:460] when available
            if data.shape[-1] >= 460:
                data = data[:, 20:460]
        else:
            data_path = self.input_paths[index]
            data = np.load(data_path)

        if data.shape[-1] > self.data_len:
            idx = np.random.randint(0, int(data.shape[-1] - self.data_len)+1)

            data = data[:, idx: idx+self.data_len]
        else:
            x = np.linspace(0, 1, data.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, data)
            data = f(x2)
        ret = np.zeros((self.data_chan, self.data_len))
        if (self.data_chan > data.shape[-2]):
            for i in range((self.data_chan//data.shape[-2])):

                ret[i * data.shape[-2]: (i+1) * data.shape[-2], :] = data
            if self.data_chan % data.shape[-2] != 0:

                ret[ -(self.data_chan%data.shape[-2]):, :] = data[: (self.data_chan%data.shape[-2]), :]
        elif(self.data_chan < data.shape[-2]):
            idx2 = np.random.randint(0, int(data.shape[-2] - self.data_chan)+1)
            ret = data[idx2: idx2+self.data_chan, :]
        # print(ret.shape)
        elif(self.data_chan == data.shape[-2]):
            ret = data
        ret = ret/10 # reduce an order
        # torch.tensor()
        ret = torch.from_numpy(ret).float()
        return {'eeg': ret } #,



def get_img_label(class_index:dict, img_filename:list, naive_label_set=None):
    img_label = []
    wind = []
    desc = []
    for _, v in class_index.items():
        n_list = []
        for n in v[:-1]:
            n_list.append(int(n[1:]))
        wind.append(n_list)
        desc.append(v[-1])

    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(wind, desc)):
            if name in w:
                img_label.append((c, d, nl))
                break
    return img_label, naive_label

class base_dataset(Dataset):
    def __init__(self, x, y=None, transform=identity):
        super(base_dataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.transform(self.x[index])
        else:
            return self.transform(self.x[index]), self.transform(self.y[index])
    
def remove_repeats(fmri, img_lb):
    assert len(fmri) == len(img_lb), 'len error'
    fmri_dict = {}
    for f, lb in zip(fmri, img_lb):
        if lb in fmri_dict.keys():
            fmri_dict[lb].append(f)
        else:
            fmri_dict[lb] = [f]
    lbs = []
    fmris = []
    for k, v in fmri_dict.items():
        lbs.append(k)
        fmris.append(np.mean(np.stack(v), axis=0))
    return np.stack(fmris), lbs


def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]

EEG_EXTENSIONS = [
    '.mat'
]


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in EEG_EXTENSIONS)


def make_dataset(dir):

    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir, topdown=False)):#
        for fname in fnames:
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

from PIL import Image
from PIL import UnidentifiedImageError
import numpy as np
 


class EEGDataset(Dataset):

    # Constructor
    def __init__(self, eeg_signals_path, imagenet_path, image_transform=identity, subject = 4):
        if not imagenet_path or not str(imagenet_path).strip():
            raise RuntimeError(
                "EEGDataset requires imagenet_path for real GT images. "
                "No fallback to random noise. Set --imagenet_path to the ImageNet root (e.g. /path/to/ILSVRC2012)."
            )
        # Load EEG signals
        loaded = torch.load(eeg_signals_path, map_location='cpu', weights_only=False)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        # else:
        # print(loaded)
        if subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==subject]
        else:
            self.data = loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = imagenet_path
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 512
        # Indices whose image file exists and is readable (skip corrupt/empty files)
        self.valid_indices = self._build_valid_indices()
        # Compute size
        self.size = len(self.data)
        try:
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        except (OSError, ConnectionError, Exception) as e:
            if "ProxyError" in type(e).__name__ or "connection" in str(e).lower():
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        "openai/clip-vit-large-patch14", local_files_only=True, use_fast=True
                    )
                    print("[dataset] Loaded CLIP processor from local cache (offline mode)")
                except Exception as e2:
                    raise RuntimeError(
                        "Could not load CLIP processor. Ensure openai/clip-vit-large-patch14 "
                        "is cached (run once with internet) or disable proxy: "
                        f"{e2}"
                    ) from e2
            else:
                raise

    def _image_readable(self, idx):
        """Return True if the image file for data index idx exists and can be opened."""
        image_name = self.images[self.data[idx]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split("_")[0], image_name + ".JPEG")
        image_path = os.path.normpath(os.path.abspath(image_path))
        if not os.path.isfile(image_path):
            return False
        try:
            with Image.open(image_path) as im:
                im.load()
        except (UnidentifiedImageError, OSError):
            return False
        return True

    def _build_valid_indices(self):
        """Build set of indices whose image file is present and readable."""
        n = len(self.data)
        valid = []
        for i in range(n):
            if self._image_readable(i):
                valid.append(i)
        n_bad = n - len(valid)
        if n_bad > 0:
            print("[dataset] Skipping %d indices with missing/corrupt images (valid=%d)." % (n_bad, len(valid)))
        return set(valid)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):

        eeg = self.data[i]["eeg"].float().t()

        eeg = eeg[20:460,:]

        eeg = np.array(eeg.transpose(0,1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()

        label = torch.tensor(self.data[i]["label"]).long()

        # Get image: require real file, no random noise fallback
        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name + '.JPEG')
        image_path = os.path.normpath(os.path.abspath(image_path))
        if i not in self.valid_indices:
            raise RuntimeError(
                "Index %d excluded: image missing or corrupt at %s (valid_indices built at init)."
                % (i, image_path)
            )
        if not os.path.isfile(image_path):
            raise RuntimeError("Image file missing: %s (imagenet_path=%s, image_name=%s)" % (
                image_path, self.imagenet, image_name))
        img_debug = os.environ.get('IMG_DEBUG') == '1'
        if img_debug:
            try:
                fsize = os.path.getsize(image_path)
            except OSError:
                fsize = -1
            print("[IMG_DEBUG] idx=%d label=%s image_id=%s path=%s exists=True size=%s" % (
                i, label.item(), image_name, image_path, fsize))
        try:
            image_raw = Image.open(image_path).convert('RGB')
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(
                "Image file is corrupt or unreadable: %s (remove or replace the file and re-run). Original: %s"
                % (image_path, e)
            ) from e
        # Float32 [0, 1] before transform (transform may resize and normalize to [-1,1])
        image = np.array(image_raw, dtype=np.float32) / 255.0
        if img_debug:
            print("[IMG_DEBUG] after load shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f" % (
                image.shape, float(image.min()), float(image.max()), float(image.mean()), float(image.std())))
        image_raw = self.processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)

        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}
        # Return
        # return eeg, label

class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train", subject=4):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path, map_location='cpu', weights_only=False)
        if "splits" not in loaded:
            raise KeyError(f"splits_path must contain 'splits' key. Keys found: {list(loaded.keys())}")
        splits = loaded["splits"]
        if split_num >= len(splits):
            raise IndexError(f"split_num={split_num} but splits has only {len(splits)} split(s).")
        if split_name not in splits[split_num]:
            raise KeyError(f"split_name '{split_name}' not in splits[{split_num}]. Keys: {list(splits[split_num].keys())}")
        self.split_idx = list(splits[split_num][split_name])
        # Filter data (skip indices that would cause index or key errors)
        max_idx = len(self.dataset.data) - 1
        filtered = []
        for i in self.split_idx:
            if i < 0 or i > max_idx:
                continue
            try:
                eeg = self.dataset.data[i].get("eeg")
                if eeg is None:
                    continue
                L = eeg.size(1) if hasattr(eeg, 'size') else eeg.shape[1]
                if 450 <= L <= 600:
                    filtered.append(i)
            except (KeyError, IndexError, TypeError):
                continue
        self.split_idx = filtered
        if hasattr(self.dataset, 'valid_indices'):
            self.split_idx = [idx for idx in self.split_idx if idx in self.dataset.valid_indices]
        # Compute size

        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


def create_EEG_dataset(eeg_signals_path='../dreamdiffusion/datasets/eeg_5_95_std.pth', 
            splits_path = '../dreamdiffusion/datasets/block_splits_by_image_single.pth',
            imagenet_path = None,
            image_transform=identity, subject = 0):

    if isinstance(image_transform, list):
        dataset_train = EEGDataset(eeg_signals_path, imagenet_path, image_transform[0], subject )
        dataset_test = EEGDataset(eeg_signals_path, imagenet_path, image_transform[1], subject)
    else:
        dataset_train = EEGDataset(eeg_signals_path, imagenet_path, image_transform, subject)
        dataset_test = EEGDataset(eeg_signals_path, imagenet_path, image_transform, subject)
    split_train = Splitter(dataset_train, split_path = splits_path, split_num = 0, split_name = 'train', subject= subject)
    split_test = Splitter(dataset_test, split_path = splits_path, split_num = 0, split_name = 'test', subject = subject)
    return (split_train, split_test)


class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img



def normalize2(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img



def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')


if __name__ == '__main__':
    import scipy.io as scio
    import copy
    import shutil


