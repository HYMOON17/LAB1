from __future__ import division
import argparse
import importlib
import os
import os.path as osp
import time


import torch
# from garbage_meta_train import Meta_trainer
from train_mtl_noise2true_allin import Meta_train
# from train_mtl_noise2true_revised import Meta_train
# from test_mtl_noise2true_revised import Meta_test
# from test_mtl_noise2self_revised import Meta_test
# from test_mtl_ssrl_noise2self_revised import Meta_test

def parse_args():
    parser = argparse.ArgumentParser(description='Perform Meta Training')

    parser.add_argument(
        '--pretrained', default='trained_models/noise2true_chest_abdomen.pt', help='pretrained model file')
    parser.add_argument(
        '--ssrl_knowledge', default=None, help='pretrained model file')
    parser.add_argument(
        '--meta_batch_size', default=15, help='Number of tasks per batch')
    parser.add_argument(
        '--num_of_epochs', default=1000, help='Number of epochs for meta train')
    parser.add_argument(
        '--step_size', default=10, help='Scheduler Parameter')
    parser.add_argument(
        '--gamma', default=0.95, help='Scheduler Parameter')
    parser.add_argument(
        '--lr_beta', default=1e-1, help='Learning rate for meta training')    
    parser.add_argument(
        '--lr_alpha', default=1e-2, help='Learning rate inner gradient update')



    parser.add_argument('--samples_per_task', default=5, help='total samples per task')
    parser.add_argument('--tr_sample_pr', default=None, help='training samples percentage per task')
    parser.add_argument('--num_of_tasks', default=15, help='Total Number of tasks')
    parser.add_argument('--num_of_updates', default=100, help='Number of inner gradient updates.')
    parser.add_argument('--data_path', default='data/train_imgs_mtl.mat')

    parser.add_argument('--loss_criterion',default='noise2self',
        help='Specify the loss criterion.')

    args = parser.parse_args()

    return args


def main():
    args    = parse_args()

    # trainer = Meta_trainer(args)
    trainer = Meta_train(args)
    
    # tester  = Meta_test(args)
    print("Num of updates ",args.num_of_updates)
    print("Learning rate alpha: ",args.lr_alpha)
    print("Learning rate beta: ",args.lr_beta)
    trainer.train()
    # tester.test()
    

if __name__=="__main__":
    main()