import cv2
import os
import sys
sys.path.insert(0, '../')
import numpy as np
import math
import glob
import pyspng
import PIL.Image
import torch
import dnnlib
import scipy.linalg
import sklearn.svm


_feature_detector_cache = dict()

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]


def read_image(image_path):
    with open(image_path, 'rb') as f:
        if pyspng is not None and image_path.endswith('.png'):
            image = pyspng.load(f.read())
        else:
            image = np.array(PIL.Image.open(f))
    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image.transpose(2, 0, 1) # HWC => CHW
    image = torch.from_numpy(image).unsqueeze(0).to(torch.uint8)

    return image


class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj


def calculate_metrics(folder1, folder2, kind):
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))
    assert(len(l1) == len(l2))
    print('length:', len(l1))

    # l1 = l1[:3]; l2 = l2[:3];

    # build detector
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    device = torch.device('cuda:0')
    detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=0, verbose=False)
    detector.eval()

    stat1 = FeatureStats(capture_all=True, capture_mean_cov=True, max_items=len(l1))
    stat2 = FeatureStats(capture_all=True, capture_mean_cov=True, max_items=len(l1))

    with torch.no_grad():
        for i, (fpath1, fpath2) in enumerate(zip(l1, l2)):
            # print(i)
            _, name1 = os.path.split(fpath1)
            _, name2 = os.path.split(fpath2)
            if kind=='ukiyoe':
                name1 = name1.split('.')[1][:-16]
                name2 = name2.split('.')[1]
            else:
                name1 = name1.split('.')[0][:-16]
                name2 = name2.split('.')[0]
            assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, name2)

            img1 = read_image(fpath1).to(device)
            img2 = read_image(fpath2).to(device)
            assert img1.shape == img2.shape, 'Illegal shape'
            fea1 = detector(img1, **detector_kwargs)
            stat1.append_torch(fea1, num_gpus=1, rank=0)
            fea2 = detector(img2, **detector_kwargs)
            stat2.append_torch(fea2, num_gpus=1, rank=0)

    # calculate fid
    mu1, sigma1 = stat1.get_mean_cov()
    mu2, sigma2 = stat2.get_mean_cov()
    m = np.square(mu1 - mu2).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))

    # calculate pids and uids
    fake_activations = stat1.get_all()
    real_activations = stat2.get_all()
    svm = sklearn.svm.LinearSVC(dual=False)
    svm_inputs = np.concatenate([real_activations, fake_activations])
    svm_targets = np.array([1] * real_activations.shape[0] + [0] * fake_activations.shape[0])
    print('SVM fitting ...')
    svm.fit(svm_inputs, svm_targets)
    uids = 1 - svm.score(svm_inputs, svm_targets)
    real_outputs = svm.decision_function(real_activations)
    fake_outputs = svm.decision_function(fake_activations)
    pids = np.mean(fake_outputs > real_outputs)

    return fid, pids, uids

import argparse

if __name__ == '__main__':
    folder1 = 'path to the inpainted result'
    folder2 = 'path to the gt'


    aparser = argparse.ArgumentParser()
    aparser.add_argument('--output_name', type=str, help='the dir name in outputs')
    aparser.add_argument('--kind', type=str, help='the dir name in outputs')
    aparser.add_argument('--model_name', type=str, default='all', help='')
    args = aparser.parse_args()

    root_path = os.environ.get('TORCH_HOME')
    output_name = args.output_name
    kind = args.kind
    if kind=='MetFace':
        cat_test_real_256 = f'{root_path}/datasets/MetFace/test_500_source_random_seed10002'
        cat_test_real_256_for_thick = f'{root_path}/datasets/MetFace/test_500_source_random_seed10002_for_thick'
    else:
        cat_test_real_256 = f'{root_path}/datasets/afhq/test_origin/{kind}'
        cat_test_real_256_for_thick = f'{root_path}/datasets/afhq/test_origin/{kind}_for_thick'

    if args.model_name=='all':
        three_model_name = ['model0', 'model1', 'last']
    else:
        three_model_name = [args.model_name]

    for model_name in three_model_name:
        # -------------- for thin mask -------------
        inpainted_img_path = f'{root_path}/outputs/{output_name}/{model_name}_random_thin_256'
        folder1, folder2 = inpainted_img_path, cat_test_real_256

        fid, pids, uids = calculate_metrics(folder1, folder2, kind)
        print('fid: %.4f, pids: %.4f, uids: %.4f' % (fid, pids, uids))
        with open(f'{root_path}/outputs/{output_name}/{model_name}_thin_fid_pids_uids.txt', 'w') as f:
            f.write('fid: %.4f, pids: %.4f, uids: %.4f' % (fid, pids, uids))

        # --------------- for medium mask -------------
        inpainted_img_path = f'{root_path}/outputs/{output_name}/{model_name}_random_medium_256'
        folder1, folder2 = inpainted_img_path, cat_test_real_256

        fid, pids, uids = calculate_metrics(folder1, folder2, kind)
        print('fid: %.4f, pids: %.4f, uids: %.4f' % (fid, pids, uids))
        with open(f'{root_path}/outputs/{output_name}/{model_name}_medium_fid_pids_uids.txt', 'w') as f:
            f.write('fid: %.4f, pids: %.4f, uids: %.4f' % (fid, pids, uids))

        # --------------- for thick mask ----------------
        inpainted_img_path = f'{root_path}/outputs/{output_name}/{model_name}_random_thick_256'
        folder1, folder2 = inpainted_img_path, cat_test_real_256_for_thick

        fid, pids, uids = calculate_metrics(folder1, folder2, kind)
        print('fid: %.4f, pids: %.4f, uids: %.4f' % (fid, pids, uids))
        with open(f'{root_path}/outputs/{output_name}/{model_name}_thick_fid_pids_uids.txt', 'w') as f:
            f.write('fid: %.4f, pids: %.4f, uids: %.4f' % (fid, pids, uids))

