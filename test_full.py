#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights_full', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-lower', type=int, default=0, help='first test index')
    parser.add_argument('-upper', type=int, default=20, help='last test index')
    args = parser.parse_args()

    net = get_network(args.net, gpu = args.gpu)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    acc_top_1_list = []
    acc_top_5_list = []

    for i in range(args.lower, args.upper):
        print("--- STATE " + str((i+1)*10) + " ---")

        path = args.weights_full + args.net + "-" + str((i+1)*10) + "-regular.pth"

        net.load_state_dict(torch.load(path))
        # print(net)
        net.eval()

        correct_1 = 0.0
        correct_5 = 0.0
        total = 0

        with torch.no_grad():
            for n_iter, (image, label) in tqdm(enumerate(cifar100_test_loader)):
                # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

                if args.gpu:
                    image = image.cuda()
                    label = label.cuda()
                    # print('GPU INFO.....')
                    # print(torch.cuda.memory_summary(), end='')


                output = net(image)
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()

                #compute top 5
                correct_5 += correct[:, :5].sum()

                #compute top1
                correct_1 += correct[:, :1].sum()

        # if args.gpu:
            # print('GPU INFO.....')
            # print(torch.cuda.memory_summary(), end='')

        print()
        print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
        print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
        print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

        acc_top_1_list.append(float(1 - correct_1 / len(cifar100_test_loader.dataset)))
        acc_top_5_list.append(float(1 - correct_5 / len(cifar100_test_loader.dataset)))
        
    print("Top 1 full acc:", acc_top_1_list)
    print("Top 1 full acc:", acc_top_5_list)

