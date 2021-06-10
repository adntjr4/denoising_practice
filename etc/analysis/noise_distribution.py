import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import argparse
import math

import cv2
import matplotlib.pyplot as plt
import torch
import scipy.stats as stats

from src.model.custom_RBSN import NLFNet, LocalMeanNet
from src.datahandler import get_dataset_object
from src.util.util import imwrite_test

def get_mean_std_from_hist(hist, xlabel):
    mean = (hist * xlabel).sum() / hist.sum()
    std = (hist * torch.square(xlabel)).sum() / hist.sum() - torch.square(mean)

    return mean, torch.sqrt(std)


def main():
    # initialization
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', default=None,  type=str)

    args = args.parse_args()
    assert args.dataset is not None, 'dataset name is required'

    dataset = get_dataset_object(args.dataset)()

    # nlf network
    nlf_net = NLFNet(real=True)
    avg_net = LocalMeanNet(3, 9)

    graph_img_path = './etc/analysis/results/noise_distribution/'
    plt_min = -120
    plt_max = 120
    bins = 241

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
        data = dataset.__getitem__(29)

        # nlf
        nlf = nlf_net(data['real_noisy'].unsqueeze(0)).mean((0,2,3))[1]

        #clipping correction
        mean = avg_net(data['real_noisy'].unsqueeze(0))
        _,w,h = data['real_noisy'].shape
        data['real_noisy'][0] -= (nlf*torch.abs_(torch.randn((w,h))) - mean[0,0]) * (data['real_noisy'][0]<1.0)
        data['real_noisy'][1] -= (nlf*torch.abs_(torch.randn((w,h))) - mean[0,1]) * (data['real_noisy'][1]<1.0)
        data['real_noisy'][2] -= (nlf*torch.abs_(torch.randn((w,h))) - mean[0,2]) * (data['real_noisy'][2]<1.0)
        print(nlf)
        print(nlf_net(data['real_noisy'].unsqueeze(0)).mean((0,2,3))[1])


        if 'syn_noisy' in data:
            noise_residual = data['syn_noisy'] - data['clean']
        else:
            noise_residual = data['real_noisy'] - data['clean']

        noise_residual = noise_residual.view(3, -1)

        # total noise
        total_hist += torch.histc(noise_residual.reshape(-1), bins=bins, min=plt_min, max=plt_max)
        total_mean += noise_residual.reshape(-1).mean().item()
        total_std += noise_residual.reshape(-1).std().item() ** 2

        # RGB noise
        mask = False
        if mask:
            R_hist += torch.histc(noise_residual[0][data['real_noisy'].view(3,-1)[0]>1.0], bins=bins, min=plt_min, max=plt_max)
            G_hist += torch.histc(noise_residual[1][data['real_noisy'].view(3,-1)[1]>1.0], bins=bins, min=plt_min, max=plt_max)
            B_hist += torch.histc(noise_residual[2][data['real_noisy'].view(3,-1)[2]>1.0], bins=bins, min=plt_min, max=plt_max)
            R_mean += noise_residual[0][data['real_noisy'].view(3,-1)[0]>1.0].mean().item()
            R_std  += noise_residual[0][data['real_noisy'].view(3,-1)[0]>1.0].std().item()
            G_mean += noise_residual[1][data['real_noisy'].view(3,-1)[1]>1.0].mean().item()
            G_std  += noise_residual[1][data['real_noisy'].view(3,-1)[1]>1.0].std().item()
            B_mean += noise_residual[2][data['real_noisy'].view(3,-1)[2]>1.0].mean().item()
            B_std  += noise_residual[2][data['real_noisy'].view(3,-1)[2]>1.0].std().item()
        else:
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
        total_pixel = data['real_noisy'].shape[1] * data['real_noisy'].shape[2]
        R_num = (data['real_noisy'][0]<1.0).sum()
        G_num = (data['real_noisy'][1]<1.0).sum()
        B_num = (data['real_noisy'][2]<1.0).sum()

        print(R_num/total_pixel)
        print(G_num/total_pixel)
        print(B_num/total_pixel)

        print('img%d is done'%data_idx)

        imwrite_test(data['real_noisy'], 'noisy')
        imwrite_test(data['clean'], 'clean')

        R_zero = torch.zeros_like(data['real_noisy'])
        R_zero[0] = 255*(data['real_noisy'][0]<1.0)
        G_zero = torch.zeros_like(data['real_noisy'])
        G_zero[1] = 255*(data['real_noisy'][1]<1.0)
        B_zero = torch.zeros_like(data['real_noisy'])
        B_zero[2] = 255*(data['real_noisy'][2]<1.0)

        imwrite_test(R_zero, 'R_zero')
        imwrite_test(G_zero, 'G_zero')
        imwrite_test(B_zero, 'B_zero')

    # total
    total_mean, total_std = get_mean_std_from_hist(total_hist, xlabel)
    print('total mean:%.02f, std:%.02f'%(total_mean, total_std))

    plt.plot(xlabel, stats.norm.pdf(xlabel, total_mean, total_std))
    plt.scatter(xlabel, total_hist/total_hist.sum(), marker='+')
    plt.savefig(os.path.join(graph_img_path, 'total_noise_distribution.png'))
    plt.clf()

    # RGB
    R_mean, R_std = get_mean_std_from_hist(R_hist, xlabel)
    G_mean, G_std = get_mean_std_from_hist(G_hist, xlabel)
    B_mean, B_std = get_mean_std_from_hist(B_hist, xlabel)
    print('R mean:%.02f, std:%.02f'%(R_mean, R_std))
    print('G mean:%.02f, std:%.02f'%(G_mean, G_std))
    print('B mean:%.02f, std:%.02f'%(B_mean, B_std))

    plt.plot(xlabel, stats.norm.pdf(xlabel, R_mean, R_std), c='#FF0000')
    plt.scatter(xlabel, R_hist/R_hist.sum(), c='#FF0000', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'R_noise_distribution.png'))
    plt.clf()

    plt.plot(xlabel, stats.norm.pdf(xlabel, G_mean, G_std), c='#00FF00')
    plt.scatter(xlabel, G_hist/G_hist.sum(), c='#00FF00', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'G_noise_distribution.png'))
    plt.clf()

    plt.plot(xlabel, stats.norm.pdf(xlabel, B_mean, B_std), c='#0000FF')
    plt.scatter(xlabel, B_hist/B_hist.sum(), c='#0000FF', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'B_noise_distribution.png'))
    plt.clf()

    plt.plot(xlabel, stats.norm.pdf(xlabel, R_mean, R_std), c='#FF0000')
    plt.plot(xlabel, stats.norm.pdf(xlabel, G_mean, G_std), c='#00FF00')
    plt.plot(xlabel, stats.norm.pdf(xlabel, B_mean, B_std), c='#0000FF')
    plt.scatter(xlabel, R_hist/R_hist.sum(), c='#FF0000', marker='+')
    plt.scatter(xlabel, G_hist/G_hist.sum(), c='#00FF00', marker='+')
    plt.scatter(xlabel, B_hist/B_hist.sum(), c='#0000FF', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'RGB_noise_distribution.png'))
    plt.clf()

if __name__ == '__main__':
    main()