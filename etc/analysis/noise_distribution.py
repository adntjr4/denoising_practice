import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import argparse

import cv2
import matplotlib.pyplot as plt
import torch
import scipy.stats as stats

from src.datahandler import get_dataset_object
from src.util.util import imwrite_test

def main():
    # initialization
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', default=None,  type=str)

    args = args.parse_args()
    assert args.dataset is not None, 'dataset name is required'

    dataset = get_dataset_object(args.dataset)()

    graph_img_path = './etc/analysis/results/noise_distribution/'
    plt_min = -50
    plt_max = 50
    bins = 101

    # analysis one by one
    xlabel = torch.arange(plt_min, plt_max+1, (plt_max-plt_min)/(bins-1))

    total_hist = torch.zeros(bins)
    total_mean, total_std = 0., 0.

    R_hist = torch.zeros(bins)
    G_hist = torch.zeros(bins)
    B_hist = torch.zeros(bins)
    R_mean, R_std = 0., 0.
    G_mean, G_std = 0., 0.
    B_mean, B_std = 0., 0.

    data_len = dataset.__len__()
    data_len = 1
    for data_idx in range(data_len):
        data = dataset.__getitem__(1)
        if 'syn_noisy' in data:
            noise_residual = data['syn_noisy'] - data['clean']
        else:
            noise_residual = data['real_noisy'] - data['clean']

        noise_residual = noise_residual.view(3, -1)

        # total noise
        total_hist += torch.histc(noise_residual.reshape(-1), bins=bins, min=plt_min, max=plt_max)
        total_mean += noise_residual.reshape(-1).mean().item()
        total_std += noise_residual.reshape(-1).std().item()

        # RGB noise
        R_hist += torch.histc(noise_residual[0], bins=bins, min=plt_min, max=plt_max)
        G_hist += torch.histc(noise_residual[1], bins=bins, min=plt_min, max=plt_max)
        B_hist += torch.histc(noise_residual[2], bins=bins, min=plt_min, max=plt_max)
        R_mean += noise_residual[0].mean().item()
        R_std  += noise_residual[0].std().item()
        G_mean += noise_residual[1].mean().item()
        G_std  += noise_residual[1].std().item()
        B_mean += noise_residual[2].mean().item()
        B_std  += noise_residual[2].std().item()

        print('mean of RGB: ', data['clean'].mean(dim=(-1,-2)))
        imwrite_test(data['clean'])
        print(data['real_noisy'].shape)
        print('R max: ', (data['real_noisy'][0]>254.0).sum())
        print('R min: ', (data['real_noisy'][0]<1.0).sum())
        print('G max: ', (data['real_noisy'][1]>254.0).sum())
        print('G min: ', (data['real_noisy'][1]<1.0).sum())
        print('B max: ', (data['real_noisy'][2]>254.0).sum())
        print('B min: ', (data['real_noisy'][2]<1.0).sum())

        print('img%d is done'%data_idx)

    # total
    print('total mean:%.02f, std:%.02f'%(total_mean/data_len, total_std/data_len))

    plt.plot(xlabel, stats.norm.pdf(xlabel, total_mean/data_len, total_std/data_len))
    plt.scatter(xlabel, total_hist/total_hist.sum(), marker='+')
    plt.savefig(os.path.join(graph_img_path, 'total_noise_distribution.png'))
    plt.clf()

    # RGB
    print('R mean:%.02f, std:%.02f'%(R_mean/data_len, R_std/data_len))
    print('G mean:%.02f, std:%.02f'%(G_mean/data_len, G_std/data_len))
    print('B mean:%.02f, std:%.02f'%(B_mean/data_len, B_std/data_len))

    plt.plot(xlabel, stats.norm.pdf(xlabel, R_mean/data_len, R_std/data_len), c='#FF0000')
    plt.scatter(xlabel, R_hist/R_hist.sum(), c='#FF0000', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'R_noise_distribution.png'))
    plt.clf()

    plt.plot(xlabel, stats.norm.pdf(xlabel, G_mean/data_len, G_std/data_len), c='#00FF00')
    plt.scatter(xlabel, G_hist/G_hist.sum(), c='#00FF00', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'G_noise_distribution.png'))
    plt.clf()

    plt.plot(xlabel, stats.norm.pdf(xlabel, B_mean/data_len, B_std/data_len), c='#0000FF')
    plt.scatter(xlabel, B_hist/B_hist.sum(), c='#0000FF', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'B_noise_distribution.png'))
    plt.clf()

    plt.plot(xlabel, stats.norm.pdf(xlabel, R_mean/data_len, R_std/data_len), c='#FF0000')
    plt.plot(xlabel, stats.norm.pdf(xlabel, G_mean/data_len, G_std/data_len), c='#00FF00')
    plt.plot(xlabel, stats.norm.pdf(xlabel, B_mean/data_len, B_std/data_len), c='#0000FF')
    plt.scatter(xlabel, R_hist/R_hist.sum(), c='#FF0000', marker='+')
    plt.scatter(xlabel, G_hist/G_hist.sum(), c='#00FF00', marker='+')
    plt.scatter(xlabel, B_hist/B_hist.sum(), c='#0000FF', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'RGB_noise_distribution.png'))
    plt.clf()

if __name__ == '__main__':
    main()