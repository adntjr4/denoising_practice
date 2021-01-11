import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import argparse

import cv2
import matplotlib.pyplot as plt
import torch
import scipy.stats as stats

from src.dataset.denoise_dataset import get_dataset_object

def main():
    # initialization
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', default=None,  type=str)

    args = args.parse_args()
    assert args.dataset is not None, 'dataset name is required'

    dataset = get_dataset_object(args.dataset)(add_noise=None, crop_size=None, n_repeat=1)

    distance_list = [1,2,3,4,5,6,7,8]
    graph_img_path = './etc/analysis/results/spatial_noise/'
    plt_min = -50
    plt_max = 50
    bins = 101

    # analysis one by one
    xlabel = torch.arange(plt_min, plt_max+1, (plt_max-plt_min)/(bins-1))

    total_hist = [torch.zeros(bins) for _ in range(len(distance_list))]
    total_mean, total_std = [0.]*len(distance_list), [0.]*len(distance_list)

    R_hist = [torch.zeros(bins) for _ in range(len(distance_list))]
    G_hist = [torch.zeros(bins) for _ in range(len(distance_list))]
    B_hist = [torch.zeros(bins) for _ in range(len(distance_list))]
    R_mean, R_std = [0.]*len(distance_list), [0.]*len(distance_list)
    G_mean, G_std = [0.]*len(distance_list), [0.]*len(distance_list)
    B_mean, B_std = [0.]*len(distance_list), [0.]*len(distance_list)

    data_len = dataset.__len__()
    for data_idx in range(data_len):
        data = dataset.__getitem__(data_idx)
        if 'syn_noisy' in data:
            noise_residual = data['syn_noisy'] - data['clean']
        else:
            noise_residual = data['true_noisy'] - data['clean']

        for distance_idx, d in enumerate(distance_list):
            left_moved = torch.zeros(noise_residual.shape)
            left_moved[:, :, :-d] = noise_residual[:, :, d:]
            left_subtract = (noise_residual-left_moved)[:, :, :-d].reshape(3,-1)

            right_tensor = torch.zeros(noise_residual.shape)
            right_tensor[:, d:] = noise_residual[:, :-d]
            right_subtract = (noise_residual-right_tensor)[:, d:].reshape(3,-1)

            up_tensor = torch.zeros(noise_residual.shape)
            up_tensor[:, :-d, :] = noise_residual[:, d:, :]
            up_subtract = (noise_residual-up_tensor)[:, :-d, :].reshape(3,-1)

            down_tensor = torch.zeros(noise_residual.shape)
            down_tensor[:, d:, :] = noise_residual[:, :-d, :]
            down_subtract = (noise_residual-down_tensor)[:, d:, :].reshape(3,-1)

            res_diff_distance = torch.cat([left_subtract, right_subtract, up_subtract, down_subtract], dim=1)

            # total noise
            total_hist[distance_idx] += torch.histc(res_diff_distance.reshape(-1), bins=bins, min=plt_min, max=plt_max)
            total_mean[distance_idx] += res_diff_distance.reshape(-1).mean().item()
            total_std[distance_idx] += res_diff_distance.reshape(-1).std().item()

            # RGB noise
            R_hist[distance_idx] += torch.histc(res_diff_distance[0], bins=bins, min=plt_min, max=plt_max)
            G_hist[distance_idx] += torch.histc(res_diff_distance[1], bins=bins, min=plt_min, max=plt_max)
            B_hist[distance_idx] += torch.histc(res_diff_distance[2], bins=bins, min=plt_min, max=plt_max)
            R_mean[distance_idx] += res_diff_distance[0].mean().item()
            R_std[distance_idx]  += res_diff_distance[0].std().item()
            G_mean[distance_idx] += res_diff_distance[1].mean().item()
            G_std[distance_idx]  += res_diff_distance[1].std().item()
            B_mean[distance_idx] += res_diff_distance[2].mean().item()
            B_std[distance_idx]  += res_diff_distance[2].std().item()

            print('img%d - distance(%d) is done'%(data_idx, d))

    # total
    for idx, d in enumerate(distance_list):
        print('distance(%d) - total mean:%.02f, std:%.02f'%(d, total_mean[idx]/data_len, total_std[idx]/data_len))

        plt.plot(xlabel, stats.norm.pdf(xlabel, total_mean[idx]/data_len, total_std[idx]/data_len))
        plt.scatter(xlabel, total_hist[idx]/total_hist[idx].sum(), marker='+')
        plt.savefig(os.path.join(graph_img_path, 'distance%d_distribution.png'%d))
        plt.clf()
    for idx, d in enumerate(distance_list):
        #plt.plot(xlabel, stats.norm.pdf(xlabel, total_mean[idx]/data_len, total_std[idx]/data_len))
        plt.scatter(xlabel, total_hist[idx]/total_hist[idx].sum(), marker='+')
    plt.savefig(os.path.join(graph_img_path, 'distance_all_distribution.png'))
    plt.clf()

    # RGB
    for idx, d in enumerate(distance_list):
        print('distance(%d) R mean:%.02f, std:%.02f'%(d, R_mean[idx]/data_len, R_std[idx]/data_len))
        print('distance(%d) G mean:%.02f, std:%.02f'%(d, G_mean[idx]/data_len, G_std[idx]/data_len))
        print('distance(%d) B mean:%.02f, std:%.02f'%(d, B_mean[idx]/data_len, B_std[idx]/data_len))

        plt.plot(xlabel, stats.norm.pdf(xlabel, R_mean[idx]/data_len, R_std[idx]/data_len), c='#FF0000')
        plt.scatter(xlabel, R_hist[idx]/R_hist[idx].sum(), c='#FF0000', marker='+')
        plt.savefig(os.path.join(graph_img_path, 'distance%d_R_distribution.png'%d))
        plt.clf()

        plt.plot(xlabel, stats.norm.pdf(xlabel, G_mean[idx]/data_len, G_std[idx]/data_len), c='#00FF00')
        plt.scatter(xlabel, G_hist[idx]/G_hist[idx].sum(), c='#00FF00', marker='+')
        plt.savefig(os.path.join(graph_img_path, 'distance%d_G_distribution.png'%d))
        plt.clf()

        plt.plot(xlabel, stats.norm.pdf(xlabel, B_mean[idx]/data_len, B_std[idx]/data_len), c='#0000FF')
        plt.scatter(xlabel, B_hist[idx]/B_hist[idx].sum(), c='#0000FF', marker='+')
        plt.savefig(os.path.join(graph_img_path, 'distance%d_B_distribution.png'%d))
        plt.clf()

    for idx, d in enumerate(distance_list):
        plt.scatter(xlabel, R_hist[idx]/R_hist[idx].sum(), c='#FF0000', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'distance_all_R_distribution.png'))
    plt.clf()

    for idx, d in enumerate(distance_list):
        plt.scatter(xlabel, B_hist[idx]/B_hist[idx].sum(), c='#00FF00', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'distance_all_G_distribution.png'))
    plt.clf()

    for idx, d in enumerate(distance_list):
        plt.scatter(xlabel, B_hist[idx]/B_hist[idx].sum(), c='#0000FF', marker='+')
    plt.savefig(os.path.join(graph_img_path, 'distance_all_B_distribution.png'))
    plt.clf()

if __name__ == '__main__':
    main()