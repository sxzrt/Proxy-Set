import os
from domain_gap.fd_score import clustering_sample, dataset_build
from domain_gap.kd_score import calculate_kd_given_paths
import argparse
import numpy as np
import logging
import re
import pathlib
from itertools import compress
from scipy.special import softmax
import time

parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--FD_model', type=str, default='inception', choices=['inception', 'posenet'],
                    help='model to calculate FD distance')

parser.add_argument('--score_name', type=str, default='all', choices=['all', 'fid', 'var'], help='items used to caculate sampling score')
parser.add_argument('--logs_dir', type=str, metavar='PATH', default='sample_datat/log.txt')
parser.add_argument('--result_dir', type=str, metavar='PATH', default='sample_datat/test')
parser.add_argument('--use_camera', action='store_true', help='use use_camera')
parser.add_argument('--c_num', default=20, type=int, help='number of cluster')
parser.add_argument('--n_num', default=500, type=int, help='number of ids')
parser.add_argument('--weight', default=0.6, type=float, help='weight')
#parser.add_argument('--proxy', type=str, default='mix3', help='model to calculate FD distance')

opt = parser.parse_args()
logs_dir=opt.logs_dir
result_dir=opt.result_dir
weight=opt.weight
c_num=opt.c_num
n_num=opt.n_num
use_camera=opt.use_camera
score_name=opt.score_name
#######################################################################
# log
def logger_config(log_path,logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8',mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
#######################################################################

# data pool
dict = {
        0: '/home/sunxx/a_data/duke/',
        1: '/home/sunxx/a_data/market/market/', #market
        2: '/home/sunxx/a_data/MSMT/', #msmt
        3: '/home/sunxx/a_data/cuhk03-np/detected/', #cuhk
        # 3: '/home/sunxx/a_data/alice-ready3/bounding_box_test', # alice
        4: '/home/sunxx/a_data/data_base_reID/RAiD_Dataset/Raid/', # raid
        #
        5: '/home/sunxx/a_data/unreal/unreal_set1/',  #unreal
        6: '/home/sunxx/a_data/background/set13/',  # personx
        7: '/home/sunxx/a_data/data_base_reID/randperson_subset/Randperson_small/' ,# randperson
        #
        8: '/home/sunxx/a_data/data_base_reID/PKUv1a_128x48', # pku
        9: '/home/sunxx/a_data/data_base_reID/ilids/', # ilids
        }

target = dict[0]+'bounding_box_train'
training_set = dict[2]
databse_id= [1, 3, 4, 5, 6, 7]

result_dir='sample_data_2'

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

logger = logger_config(log_path=result_dir+'/log.txt',logging_name='evaluate_log')
logger.info('target is %s' % (target))
logger.info('training_set is %s' % (training_set))
logger.info(f'\r databse_id: {databse_id}')
logger.info(f'\r number of cluster: {c_num}')
logger.info(f'\r score method: {score_name}')
logger.info(f'\r weight: {weight}')


sampled_data=clustering_sample(target, dict, databse_id, opt, result_dir, c_num, score_name, weight, n_num)
np.save(result_dir + '/sampled_data.npy', sampled_data)

