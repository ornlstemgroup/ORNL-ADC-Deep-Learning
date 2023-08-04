
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from skimage import io
from network.atomNet import AtomNet
from dataset import SynDataset

import imageio
SNAPSHOT_DIR = './snapshots'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 2
SAVING_ITER = 16000
NUM_ITER = 100000
POWER = 0.9

def get_arguments():
    parser = argparse.ArgumentParser(description="UnTEM")
    parser.add_argument('--gpu', metavar='GPU', default='0', type=str,
                    help='GPU id to use.')
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--saving-iter", type=float, default=SAVING_ITER,
                        help="Number of iteration for saving.")
    parser.add_argument("--num-iter", type=float, default=NUM_ITER,
                        help="Number of iteration for training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with pimgolynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    return parser.parse_args()



def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(args, optimizer, i_iter, total_steps):
    lr = lr_poly(args.learning_rate, i_iter, total_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr

def main():
    args = get_arguments()
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = AtomNet()

    model = model.cuda()
    model.train()

    loss_fn = torch.nn.MSELoss()

    data_path = '../Data/SAC_34-35 stack aligned.tif'
    reference = io.imread(data_path)
    dataset = SynDataset(image_num=args.num_iter+args.batch_size, reference=reference)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.batch_size, shuffle=False)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
    i_iter = 0
    for i, (image,gt) in enumerate(trainloader):
        i_iter+=args.batch_size
        optimizer.zero_grad()
        adjust_learning_rate(args, optimizer, i_iter, len(trainloader)*args.batch_size+args.batch_size)
        image, gt = image.float().cuda(), gt.float().cuda()
        # print (image.max(), gt.max())
        # imageio.imsave('i.jpg', image[0][0].cpu().numpy())
        # imageio.imsave('g.jpg', gt[0][0].cpu().numpy())
        # exit(0)
        # image = interp(image)
        pred = model(image)
        pred = interp(pred)
        loss = loss_fn(pred, gt)
        loss.backward()
        optimizer.step()
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}'.format(i_iter, len(trainloader)*args.batch_size, loss))
        if i_iter%args.saving_iter==0 and i_iter!=0:
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'iter_' + str(i_iter) + '.pth'))
        if i_iter>args.num_iter:
            exit(0)



if __name__ == '__main__':
    main()