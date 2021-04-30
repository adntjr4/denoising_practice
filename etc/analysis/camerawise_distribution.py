import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import math
import argparse

import cv2
import matplotlib.pyplot as plt
import torch
import scipy.stats as stats

from src.datahandler.denoise_dataset import get_dataset_object

camera_type = {'GP':0, 'IP':1, 'S6':2, 'N6':3, 'G4':4} 
camera_name = ['GP', 'IP', 'S6', 'N6', 'G4'] 

def add_noise_histogram(hist, mean, std, noise_map, bins, plt_min, plt_max):
    noise = noise_map.view(3, -1)

    hist += torch.histc(noise.reshape(-1), bins=bins, min=plt_min, max=plt_max)
    mean += noise.reshape(-1).mean().item()
    std += noise.reshape(-1).std().item()

    return hist, mean, std

def main():
    # initialization
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', default=None,  type=str)

    args = args.parse_args()
    assert args.dataset is not None, 'dataset name is required'

    dataset = get_dataset_object(args.dataset)(add_noise=None, crop_size=None, n_repeat=1)

    graph_img_path = './etc/analysis/results/noise_distribution/'
    plt_min = -50
    plt_max = 50
    bins = 101

    # analysis one by one
    xlabel = torch.arange(plt_min, plt_max+1, (plt_max-plt_min)/(bins-1))

    hist = [torch.zeros(bins) for i in range(len(camera_type))]
    mean = [0. for i in range(len(camera_type))]
    std = [0. for i in range(len(camera_type))]
    count = [0 for i in range(len(camera_type))]

    data_len = dataset.__len__()
    for data_idx in range(data_len):
        data = dataset.__getitem__(data_idx)
        if 'syn_noisy' in data:
            noise_residual = data['syn_noisy'] - data['clean']
        else:
            noise_residual = data['real_noisy'] - data['clean']

        c_name = data['instances']['smartphone_camera_code']
        c_idx = camera_type[c_name]

        hist[c_idx], mean[c_idx], std[c_idx] = add_noise_histogram(hist[c_idx], mean[c_idx], std[c_idx], noise_residual, bins, plt_min, plt_max)
        count[c_idx] += 1

        print('img%d/%d, camera_type : %s is done'%(data_idx, data_len, c_name))

    for c_idx in range(5):
        print('mean : %f, std %f'%(mean[c_idx]/count[c_idx], std[c_idx]/count[c_idx]))

        plt.scatter(xlabel, hist[c_idx]/hist[c_idx].sum(), marker='+')
        plt.xlabel('Noise intensity')
        plt.ylabel('Probability')
        plt.title('%s'%camera_name[c_idx])
        plt.savefig(os.path.join(graph_img_path, '%s_distribution_only.png'%camera_name[c_idx]))
        plt.clf()

        plt.plot(xlabel, stats.norm.pdf(xlabel, mean[c_idx]/count[c_idx], std[c_idx]/count[c_idx]), c='#00FF00')
        plt.scatter(xlabel, hist[c_idx]/hist[c_idx].sum(), marker='+')
        plt.xlabel('Noise intensity')
        plt.ylabel('Probability')
        plt.title('%s & Gaussian'%camera_name[c_idx])
        plt.savefig(os.path.join(graph_img_path, '%s_distribution_G.png'%camera_name[c_idx]))
        plt.clf()

        plt.plot(xlabel, stats.laplace.pdf(xlabel, mean[c_idx]/count[c_idx], std[c_idx]/count[c_idx]/math.sqrt(2)), c='#00FF00')
        plt.scatter(xlabel, hist[c_idx]/hist[c_idx].sum(), marker='+')
        plt.xlabel('Noise intensity')
        plt.ylabel('Probability')
        plt.title('%s & Laplace'%camera_name[c_idx])
        plt.savefig(os.path.join(graph_img_path, '%s_distribution_L.png'%camera_name[c_idx]))
        plt.clf()

if __name__ == '__main__':
    main()