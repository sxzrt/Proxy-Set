#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from scipy.misc import imread
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d
from scipy import misc
import random
import re
from scipy.special import softmax
from shutil import copyfile
from domain_gap.sskmean.clustering.equal_groups import EqualGroupsKMeans
import glob
import os.path as osp

import numpy as np
from sklearn.cluster import KMeans
import time

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from domain_gap.models.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='3', type=str,
                    help='GPU to use (leave blank for CPU only)')


def make_square(image, max_dim = 512):
    max_dim = max(np.shape(image)[0], np.shape(image)[1])
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image

def get_activations(opt, files, model, batch_size=50, dims=8192,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # if len(files) % batch_size != 0:
    #     print(('Warning: number of images is not a multiple of the '
    #            'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_remainder=  len(files) % batch_size

    print('\rnumber of batches is %d' % n_batches),
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs+n_remainder, dims))
    if n_remainder!=0:
        n_batches=n_batches+1
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        if n_remainder!=0 and i==n_batches-1:
          end = start + n_remainder
        else:
          end = start + batch_size

        images = np.array([misc.imresize( imread(str(f)).astype(np.float32), size=[64, 64]).astype(np.float32)
                           for f in files[start:end]])

        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()
        
        if opt.FD_model == 'inception':
            pred = model(batch)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        if opt.FD_model == 'posenet':
            pred = model(batch)
            # print (np.shape (pred))
            pred = adaptive_max_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)
        print('\rPropagating batch %d/%d' % (i + 1, n_batches))

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(opt, files, model, batch_size=50,
                                    dims=8192, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(opt, files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    #eigen_vals, eigen_vecs= np.linalg.eig(sigma)
    #sum_eigen_val=eigen_vals.sum().real
    sum_eigen_val = (sigma.diagonal()).sum()
    return mu, sigma, sum_eigen_val


def _compute_statistics_of_path(opt, path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        #random.shuffle(files)
        #files = files[:2000]
        m, s, sum_eigen_val = calculate_activation_statistics(opt, files, model, batch_size,
                                               dims, cuda) 
    return m, s, sum_eigen_val


def get_id_path_of_data (dataset_id, paths):
    img_paths = []
    dataset_ids = []
    person_ids = []
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    did = 0
    for sub_path in paths:
        sub_path = pathlib.Path(sub_path)
        files = list(sub_path.glob('*.jpg')) + list(sub_path.glob('*.png'))
        # files=glob.glob(osp.join(sub_path, '*.png'))+glob.glob(osp.join(sub_path, '*.jpg'))
        dataset_id_list = [dataset_id[did] for n in range(len(files))]
        dataset_ids.extend(dataset_id_list)
        img_paths.extend(files)
        did += 1
    dataset = []
    ii = 0
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(str(img_path)).groups())
        # if pid == -1: continue  # junk images are just ignored
        camid -= 1  # index starts from 0
        dataid = dataset_ids[ii]
        person_ids.append(pid)
        dataset.append((img_path, pid, dataid))
        ii = ii + 1

    return img_paths, person_ids, dataset_ids, dataset

def clustering_sample(tpaths, dict, dataset_id, opt, result_dir, c_num, score_name, weight, n_num):
    """clustering the ids from different datasets and sampleing"""

    # preparing dataset
    paths = [dict[i]+'bounding_box_test' for i in dataset_id]
    img_paths,  person_ids,  dataset_ids, _  = get_id_path_of_data(dataset_id, paths)

    cuda = True
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    if opt.FD_model == 'inception':
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])

    if cuda:
        model.cuda()
    batch_size=256

    # caculate the various, mu sigma of target ste
    print('=========== extracting feature of target traning set ===========')
    target_path = pathlib.Path(tpaths)
    files = list(target_path.glob('*.jpg')) + list(target_path.glob('*.png'))
    # random.shuffle(files)
    # files = files[:2000]
    target_feature = get_activations(opt, files, model, batch_size, dims, cuda, verbose=False)
    m1 = np.mean(target_feature, axis=0)
    s1 = np.cov(target_feature, rowvar=False)
    sum_eigen_val1 = (s1.diagonal()).sum()


    # extracter feature for data pool
    if not os.path.exists(result_dir + '/feature.npy'):
        print('=========== extracting feature of data pool ===========')
        feature = get_activations(opt, img_paths, model, batch_size, dims, cuda, verbose=False)
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        np.save(result_dir + '/feature.npy', feature)
    else:
        feature = np.load(result_dir + '/feature.npy')

    person_ids_array=np.array(person_ids)
    mean_feature_per_id=[]
    pid_per_id=[]
    did_per_id=[]

    # get mean fature of perid and the fid, various of per_id with the target
    if not os.path.exists(result_dir + '/mean_feature_per_id.npy'):
        for did in dataset_id:
           ind_of_set = np.argwhere(np.array(dataset_ids) == did).squeeze()
           dataset_feature = feature[ind_of_set]
           dataset_pid = person_ids_array[ind_of_set]
           pid_of_dataset=set(dataset_pid)
           for pid in pid_of_dataset:
              ind_of_pid = np.argwhere(np.array(dataset_pid) == pid).squeeze()
              feature_per_id = dataset_feature[ind_of_pid]
              id_ave_feature=feature_per_id.mean(0)
              mean_feature_per_id.append(id_ave_feature)
              pid_per_id.append(pid)
              did_per_id.append(did)
        np.save(result_dir+ '/mean_feature_per_id.npy',mean_feature_per_id)
        pid_did_fid_var = np.c_[np.array(pid_per_id), np.array(did_per_id)]
        np.save(result_dir+ '/pid_did_fid_var.npy', pid_did_fid_var)
    else:
       mean_feature_per_id=np.load(result_dir + '/mean_feature_per_id.npy')
       pid_did_fid_var = np.load(result_dir + '/pid_did_fid_var.npy')

    #remove 0 and -1
    ori_pid_per_id = pid_did_fid_var[:, 0]
    remove_ind=np.r_[np.argwhere(ori_pid_per_id == -1), np.argwhere(ori_pid_per_id == 0)].squeeze()

    new_pid_did_fid_var = np.delete(pid_did_fid_var, remove_ind, 0)
    new_mean_feature_per_id = np.delete(mean_feature_per_id, remove_ind, 0)


    print('\r=========== clustering the data pool ===========')
    pid_per_id = new_pid_did_fid_var[:,0]
    did_per_id = new_pid_did_fid_var[:,1]
    # clustering ids based on ids' mean feature
    if not os.path.exists(result_dir + '/label_cluster_'+str(c_num)+'.npy'):
        estimator = KMeans(n_clusters=c_num)
        estimator.fit(new_mean_feature_per_id)
        label_pred = estimator.labels_
        np.save(result_dir + '/label_cluster_'+str(c_num)+'.npy',label_pred)
    else:
        label_pred = np.load('sample_data/' + '/label_cluster_'+str(c_num)+'.npy')

    print('\r=========== caculating the fid and v_gap between T and C_k ===========')
    if not os.path.exists(result_dir + '/cluster_fid_var.npy'):
        cluster_feature=[]
        cluster_fid=[]
        cluster_mmd=[]
        cluster_var_gap=[]
        for k in tqdm(range(c_num)):
            # initializatn of the first seed cluster 0
            initial_pid=pid_per_id[label_pred==k]
            initial_did=did_per_id[label_pred==k]
            initial_feature = feature[(dataset_ids == initial_did[0]) & (person_ids_array == initial_pid[0])]
            for j in range(1,len(initial_pid)):
                current_feature=feature[(dataset_ids == initial_did[j]) & (person_ids_array == initial_pid[j])]
                initial_feature=np.r_[initial_feature, current_feature]
            cluster_feature.append(initial_feature)
            mu = np.mean(initial_feature, axis=0)
            sigma = np.cov(initial_feature, rowvar=False)
            # caculating various
            current_var_gap = np.abs((sigma.diagonal()).sum() - sum_eigen_val1)
            current_fid = calculate_frechet_distance(m1, s1, mu, sigma)
            # mmd_value = polynomial_mmd_averages(torch.from_numpy(initial_feature), torch.from_numpy(target_feature))
            # current_mmd=mmd_value[0].mean()
            cluster_fid.append(current_fid)
            # cluster_mmd.append(current_mmd)
            cluster_var_gap.append(current_var_gap)
        np.save(result_dir + '/cluster_fid_var.npy', np.c_[np.array(cluster_fid), np.array(cluster_var_gap)])
        #np.save(result_dir+'/cluster_fid_var.npy', np.c_[np.array(cluster_fid),np.array(cluster_var_gap)])
    else:
        cluster_fid_var=np.load(result_dir + '/cluster_fid_var.npy')
        cluster_fid=cluster_fid_var[:,0]
        cluster_var_gap=cluster_fid_var[:,1]

#    cluster_fid=cluster_mmd
#    calculatting softmax score
    cluster_fida=np.array(cluster_fid)
    cluster_var_gapa=np.array(cluster_var_gap)
    score_fid = softmax(-cluster_fida)
    score_var_gap = softmax(-cluster_var_gapa)
    if score_name == 'fid':
        sample_rate=score_fid
    elif score_name == 'var':
        sample_rate=score_var_gap
    else:
        sample_rate = score_fid* weight + score_var_gap * (1-weight)

    c_num_len = []
    id_score = []
    for kk in range(c_num):
        initial_pid = pid_per_id[label_pred == kk]
        c_num_len.append(len(initial_pid))
    for jj in range(len(label_pred)):
        id_score.append(sample_rate[label_pred[jj]] / c_num_len[label_pred[jj]])

    selected_data_ind = np.sort(np.random.choice(range(len(id_score)), n_num, p=id_score))
    sdid = did_per_id[selected_data_ind]
    spid = pid_per_id[selected_data_ind]
    data_dir = result_dir + '/proxy_set'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    print('\r=========== building proxy set ===========')
    sampled_data=np.c_[sdid,spid]
    ii = dataset_build(dict, dataset_id, sampled_data, data_dir)
    print('finished')
    return sampled_data


def dataset_build(dict, dataset_id, sampled_data,result_dir):
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    pid=sampled_data[:, 1]
    new_pid=np.arange(len(pid))+1
    did=sampled_data[:, 0]

    for ii in range(len(pid)):
        id= pid[ii]
        id_set=did[ii]
        new_id=new_pid[ii]
        # sample images
        gallery_data_path = dict[id_set]+ 'bounding_box_test'
        for root, dirs, files in os.walk(gallery_data_path, topdown=True):
            for name in files:
                current_id, _ = map(int, pattern.search(str(name)).groups())
                if not (name[-3:] == 'png' or name[-3:] == 'jpg'):
                    continue
                if int(current_id)!= id:
                    continue
                src_path = gallery_data_path + '/' + name
                dst_path = result_dir+ '/bounding_box_test'
                dstr_path = result_dir + '/bounding_box_train'
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                    os.mkdir(dstr_path)
                    # one picture in train
                    copyfile(src_path, dstr_path + '/' + '{:04}'.format(new_id) + name[4:-3] + 'jpg')
                copyfile(src_path, dst_path + '/' + '{:04}'.format(new_id) + name[4:-3]+'jpg')
        query_data_path = gallery_data_path[0:-18]+ '/query'
        for root, dirs, files in os.walk(query_data_path, topdown=True):
            for name in files:
                if not (name[-3:] == 'png' or name[-3:] == 'jpg'):
                    continue
                if int(name[0:4]) != id:
                    continue
                src_path = query_data_path + '/' + name
                dst_path = result_dir+'/query'
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + '{:04}'.format(new_id) + name[4:-3]+'jpg')



def calculate_fd_given_paths(paths, opt):
    """Calculates the FID of two paths"""

    cuda = True
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    if opt.FD_model == 'inception':
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])

    if cuda:
        model.cuda()

    m1, s1, sum_eigen_val1 = _compute_statistics_of_path(opt, paths[0], model, 256,
                                         dims, cuda)

    npz_path = None
    if not paths[0].endswith(".npz"):
        if not paths[0].endswith('/'):
            npz_path = paths[0] + ".npz"
        else:
            npz_path = paths[0][:-1] + ".npz"
        np.savez(npz_path, mu = m1, sigma = s1)
    m2, s2, sum_eigen_val2 = _compute_statistics_of_path(opt, paths[1], model, 256,
                                         dims, cuda)


    fd_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fd_value, npz_path, sum_eigen_val1, sum_eigen_val2



if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fid_value,_, sum_eigen_val1, sum_eigen_val2 = calculate_fd_given_paths(args.path,
                                          args.batch_size,
                                          args.gpu != '',
                                          8192)
    #print (fid_value)
