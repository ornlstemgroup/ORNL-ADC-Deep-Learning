from skimage.morphology import disk
from skimage.filters import rank, gaussian, threshold_otsu
from skimage import morphology
from skimage import exposure
import cv2
import numpy as np

import imageio
from scipy.ndimage import filters
from skimage.measure import label, regionprops

def equlization(im):
    ori_im = exposure.equalize_hist(im)
    median = np.median(im)
    im[im == im.min()] = median
    im[im == im.max()] = median

    im = (im - im.min()) / (im.max() - im.min())
    eq_im = rank.equalize(im, disk(20)) / 255
    seg = morphology.remove_small_objects(eq_im < 0.92, 10)
    seg = gaussian(seg, sigma=5)
    seg = (seg > 0.75)
    return ori_im * 255, eq_im * 255, seg * 255

# def equlization(im):
#     median = np.median(im)
#     im[im == im.min()] = median
#     im[im == im.max()] = median
#
#     im = (im - im.min()) / (im.max() - im.min())
#     ori_im = exposure.equalize_hist(im)
#     eq_im = rank.equalize(im, disk(20)) / 255
#     seg = morphology.remove_small_objects(eq_im < 0.9, 10)
#     seg = morphology.binary_dilation(seg)
#     seg = morphology.binary_dilation(seg)
#     seg = morphology.binary_erosion(seg)
#     seg = morphology.binary_erosion(seg)
#     seg = morphology.binary_erosion(seg)
#     seg = morphology.binary_erosion(seg)
#     return ori_im * 255, eq_im * 255, seg * 255


def color(seg, im):
    im = np.uint8(im)
    seg = np.uint8(seg)
    # print (seg.max())
    color = cv2.merge([im, im, im])
    color[seg==0]=[50,50,200]
    # color = cv2.merge([im-seg, im, im])
    # color = im*0.8+seg*0.2
    return color


def g2rgb(im):
    im = np.uint8(im)
    im = cv2.merge([im,im,im])
    return  im

def concat(im1, im2, im3, im4, im5, im6):
    full_1 = np.concatenate([im1, im2, im3], axis=1)
    full_2 = np.concatenate([im4, im5, im6], axis=1)
    full = np.concatenate([full_1, full_2], axis=0)
    return full

def concat_4(im1, im2, im3, im4):
    full_1 = np.concatenate([im1, im2], axis=1)
    full_2 = np.concatenate([im3, im4], axis=1)
    full = np.concatenate([full_1, full_2], axis=0)
    return full



def flow2rgb(flow_map_np, max_value=None):
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


def cal_flow(frame0, frame1):
    frame0 = np.uint8(frame0)
    frame1 = np.uint8(frame1)
    flow = cv2.optflow.calcOpticalFlowSF(frame0, frame1, 2, 2, 4)
    rgb_flow = flow2rgb(flow.transpose(2, 0, 1))
    rgb_flow = np.uint8(rgb_flow*255)
    return rgb_flow.transpose(1, 2, 0)


def seg_refinement(seg):
    width = 20
    seg[:width, :]=255
    seg[-width:, :]=255
    seg[:, :width]=255
    seg[:, -width:]=255
    return seg

def seg2bbox(seg, img):
    mask = (seg / 255).astype(int)
    mask = 1-mask
    lbl_0 = label(mask)
    bbox = regionprops(lbl_0)
    for prop in bbox:
        bbox = cv2.rectangle(img, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (200, 50, 50), 2)
    return bbox

def seg2point(seg, img):
    mask = (seg / 255).astype(int)
    mask = 1-mask
    lbl_0 = label(mask)
    points = regionprops(lbl_0)
    for prop in points:
        points = cv2.circle(img, (int(prop.centroid[1]), int(prop.centroid[0])), 5, (50, 200, 50), -1)
    return points

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def create_circular_att(mask, h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    dist_from_center = dist_from_center*mask

    att = (dist_from_center - dist_from_center.min()) / (dist_from_center.max() - dist_from_center.min())
    att = 1 - att
    return att*mask


def center_removal(all_mask, point, r):
    for p in point:
        mask = create_circular_mask(all_mask.shape[0], all_mask.shape[1], p, r)
        all_mask = all_mask*(1-mask)
    return all_mask

def point_birth(prob, mask, r, th, new_points_list):
    # removed_mask = create_circular_mask(mask.shape[0], mask.shape[1], point, r)
    # new_mask = mask*(1-removed_mask)
    region = (prob*mask) >= th
    # imageio.imsave('mask.png', mask*255)
    if np.sum(region)>0:
        result = np.where(region == np.amax(region))
        new_point = (result[1][0], result[0][0])
        new_points_list.append(new_point)
        new_mask = center_removal(mask, [new_point], r)
        # imageio.imsave('mask2.png', new_mask*255)
        # exit(0)
        new_points_list = point_birth(prob, new_mask, r, th, new_points_list)

    return new_points_list

def points2map(temp, point, r):
    bg = np.ones_like(temp)
    for p in point:
        mask = create_circular_mask(bg.shape[0], bg.shape[1], p, r)
        bg += create_circular_att(mask, bg.shape[0], bg.shape[1], p, r)
    bg = (bg - bg.min()) / (bg.max() - bg.min())

    return bg

def track_points(prob, init_points, noise, trad_points, d=30, r=15, lmd=1.0, th=0.8):
    print (len(init_points))
    points_list = []
    other_points_list = []
    prob = filters.gaussian_filter(prob, sigma=2)
    trad_prob = points2map(prob, trad_points, r)
    prob = np.maximum(prob, np.max(prob)*trad_prob)
    potential_region = np.ones_like(prob)
    all_mask = np.ones_like(prob)

    for p in init_points:
        mask = create_circular_mask(prob.shape[0], prob.shape[1], p, d)
        att = create_circular_att(mask, prob.shape[0], prob.shape[1], p, d)
        if np.max(prob*all_mask*mask)<th:
            region = prob*mask + 0.3*att
            result = np.where(region == np.amax(region))
            new_point = (result[1][0], result[0][0])
            points_list.append(new_point)
        else:
            region = prob*all_mask*mask + 0.3*att
            result = np.where(region == np.amax(region))
            new_point = (result[1][0], result[0][0])
            points_list.append(new_point)
        new_mask = create_circular_mask(prob.shape[0], prob.shape[1], new_point, r)
        all_mask = all_mask*(1-new_mask)
        potential_region = potential_region*(1-mask)

    potential_region = (1-potential_region)*noise
    potential_region = center_removal(potential_region, points_list, r)
    other_points_list = point_birth(prob, potential_region, r, th, other_points_list)

    # imageio.imsave('3_pred+trad.png', prob * 255)
    # imageio.imsave('6_mask.png', all_mask*255)
    # imageio.imsave('5_region.png', potential_region*255)
    # exit(0)

    return points_list+other_points_list, prob


def plot_point(points, img, color):
    for i in range(len(points)):
        img = cv2.circle(img, points[i], 5, color[i], -1)
    return img

def plot_bond(bonds, img, color):
    for i in range(len(bonds)):
        img = cv2.circle(img, bonds[i][0], 3, color[i], -1)
        img = cv2.circle(img, bonds[i][1], 3, color[i], -1)
        img = cv2.line(img, bonds[i][0], bonds[i][1], color[i], 2)
    return img

def plot_txt(bonds, img, color):
    img = cv2.putText(img, str(len(bonds)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 2, cv2.LINE_AA)
    return img