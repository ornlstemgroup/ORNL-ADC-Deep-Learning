
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import argparse
from network.atomNet import AtomNet
import math
from utils import *

SAVE_PATH = './result/'
RESTORE_FROM = './snapshots/iter_96000.pth'
DATA = 'PtCN_80kV_10Mx_1.tif'
DITRI = True
DICT = {
    'Ir on MgO trimmed.mov': 71.25,
    'PtCN_80kV_10Mx_1.tif': 76.54,
    'PtCN_80kV_10Mx_2.tif': 76.54,
    'PtCN_80kV_15Mx_1.tif': 114.81,
    'PtCN_80kV_15Mx_2.tif': 114.81,
    'SAC_34-35 stack aligned.tif': 153.14,
    'SAC_34-35 stack aligned bin 2.tif': 76.57
}

def get_arguments():
    parser = argparse.ArgumentParser(description="UnTEM")
    parser.add_argument('--gpu', metavar='GPU', default='0', type=str,
                    help='GPU id to use.')
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--data", type=str, default=DATA,
                        help="Where data from.")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--ditri", type=bool, default=DITRI,
                        help="find ditri or not.")
    return parser.parse_args()


def find_smallest_dist(dist, points, bi_list, paired_list1, paired_list2, lb):
    if np.min(dist) < lb:
        result = np.where(dist == np.amin(dist))
        dist[result[0][0],  result[1][0]] = np.inf
        dist[result[1][0],  result[0][0]] = np.inf

        for i in paired_list2:
            if (i[0] == result[0][0]) or (i[0] == result[1][0]) or (i[1] == result[0][0]) or (i[1] == result[1][0]):
                dist[i[0], :] = np.inf
                dist[i[1], :] = np.inf
                dist[result[0][0], :] = np.inf
                dist[result[1][0], :] = np.inf
                dist[:, i[0]] = np.inf
                dist[:, i[1]] = np.inf
                dist[:, result[0][0]] = np.inf
                dist[:, result[1][0]] = np.inf
        if not ((result[0][0] in paired_list1) and (result[1][0] in paired_list1)):
            pairs = (points[result[0][0]], points[result[1][0]])
            paired_list2.append((result[0][0], result[1][0]))
            paired_list1.append(result[0][0])
            paired_list1.append(result[1][0])
            bi_list.append(pairs)
        dist, bi_list = find_smallest_dist(dist, points, bi_list, paired_list1, paired_list2, lb)
    return dist, bi_list


def find_point(points_list, prob, mask, r):
    prob = prob*mask
    if np.sum(prob)>0:
        result = np.where(prob == np.amax(prob))
        new_point = (result[1][0], result[0][0])
        points_list.append(new_point)
        new_mask = center_removal(mask, [new_point], r)
        points_list = find_point(points_list, prob, new_mask, r)

    return points_list

def find_bi_tri(points, lb):
    paired_point_list = []
    paired_list1 = []
    paired_list2 = []
    lenth = len(points)
    dist = np.ones((lenth, lenth))*np.inf
    for i in range(lenth):
        for j in range(i+1,lenth):
            dist[i][j] = np.sqrt(np.square(points[i][0]-points[j][0])+np.square(points[i][1]-points[j][1]))

    all_dist = (dist.reshape((lenth*lenth))).tolist()
    filtered_dist = [v for v in all_dist if not math.isinf(v)]

    _, paired_point_list = find_smallest_dist(dist, points, paired_point_list, paired_list1, paired_list2, lb)

    bi_list = []
    tri_list = []
    temp_list1 = []
    temp_list2 = []
    for i in paired_point_list:
        p1, p2 = i
        is_tri = False
        for list in [temp_list1, temp_list2]:
            if p1 in list:
                index = list.index(p1)
                tri_p1 = temp_list1.pop(index)
                tri_p2 = temp_list2.pop(index)
                tri_p3 = p2
                tri_list.append((tri_p1,tri_p2, tri_p3))
                is_tri = True
            if p2 in list:
                index = list.index(p2)
                tri_p1 = temp_list1.pop(index)
                tri_p2 = temp_list2.pop(index)
                tri_p3 = p1
                tri_list.append((tri_p1,tri_p2, tri_p3))
                is_tri = True
        if not is_tri:
            temp_list1.append(p1)
            temp_list2.append(p2)
    for i in range(len(temp_list1)):
        bi_list.append((temp_list1[i],temp_list2[i]))



    return bi_list, tri_list, filtered_dist


def plot_bi(bonds, img, color):
    for i in range(len(bonds)):
        img = cv2.circle(img, bonds[i][0], 6, color[i], 2)
        img = cv2.circle(img, bonds[i][1], 6, color[i], 2)
        img = cv2.line(img, bonds[i][0], bonds[i][1], color[i], 2)
    return img


def plot_tri(bonds, img, color):
    for i in range(len(bonds)):
        img = cv2.circle(img, bonds[i][0], 6, color[i], 2)
        img = cv2.circle(img, bonds[i][1], 6, color[i], 2)
        img = cv2.circle(img, bonds[i][2], 6, color[i], 2)
        img = cv2.line(img, bonds[i][0], bonds[i][1], color[i], 2)
        img = cv2.line(img, bonds[i][0], bonds[i][2], color[i], 2)
        img = cv2.line(img, bonds[i][2], bonds[i][1], color[i], 2)
    return img

def main(restore_from):
    args = get_arguments()
    threshold = 0.8 # 0.4 for PtCN, 0.6 for Ir, 0.55 for SAC
    aml = 15
    atomr = aml
    mto = aml*2
    lb = aml*1.5

    save_path = args.save_path+'/'+args.data.split('.')[0]+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datapath = './Data/'+args.data

    from loaders.val_loader import TEMDataset
    dataset = TEMDataset(data_path=datapath)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = AtomNet()
    model.cuda()
    model.eval()
    saved_state_dict = torch.load(restore_from)
    model.load_state_dict(saved_state_dict)

    testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    sample = enumerate(testloader)
    img_size = sample.__next__()[1]['img'].shape[1:]
    interp = nn.Upsample(size=(img_size[0], img_size[1]), mode='bilinear', align_corners=True)

    dist_avg = 0
    for i, data in enumerate(testloader):
        image = data['img'].float().cuda()
        num = int(data['num'][0])
        image = image.unsqueeze(1)/255
        with torch.no_grad():
            pred = model(image)
        pred_np = interp(pred)[0][0].cpu().numpy()

        pred_np = filters.gaussian_filter(pred_np, sigma=2)
        region = pred_np>threshold

        # cv2.imwrite('pred.png', g2rgb(pred_np * 255))
        # exit(0)

        labels = label(region)
        regions = regionprops(labels)
        points_list = []
        for j in range(len(regions)):
            props = regions[j]
            y0, x0 = props.centroid
            axis_major = props.axis_major_length

            if axis_major <= aml:
                points_list.append((int(x0), int(y0)))
            elif axis_major > mto:
                current_region = labels==(j+1)
                current_region = morphology.binary_erosion(current_region)
                current_region = morphology.binary_erosion(current_region)
                points_list = find_point(points_list, pred_np, current_region, atomr)
            else:
                orientation = props.orientation
                x1 = x0 + math.sin(orientation) * 0.5 * props.axis_major_length / 3
                y1 = y0 + math.cos(orientation) * 0.5 * props.axis_major_length / 3
                x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length / 3
                y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length / 3
                points_list.append((int(x1), int(y1)))
                points_list.append((int(x2), int(y2)))

        dist = np.abs(num - len(points_list))/num
        dist_avg +=dist

    return (1-dist_avg/200)

def plot():
    import matplotlib.pyplot as plt

    data = []
    xpoints = []
    ypoints = []
    with open('data2.txt') as my_file:
        for line in my_file:
            xpoints.append(float(line.split(',')[0]))
            ypoints.append(float(line.split(',')[1][:-2]))
    plt.plot(xpoints, ypoints)
    plt.xlabel('iterations')
    plt.ylabel('training loss')
    plt.show()

if __name__ == '__main__':
    # plot()

    xpoints = []
    ypoints = []
    for i in range(1, 101):
        name = './snapshots3/iter_'+str(i*400)+'.pth'
        acc = main(name)
        xpoints.append(400*i)
        ypoints.append(acc)
        with open('val.txt', 'a') as f:
            f.write(str(400*i)+','+str(acc)+'\n')
    plt.xlabel('iterations')
    plt.ylabel('validation accuracy')
    plt.plot(xpoints, ypoints)
    plt.show()