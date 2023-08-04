from torch.utils import data
import numpy as np
import random
from simulate import simulate, normalize
from skimage import exposure

import imageio
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank, gaussian

def equlization(im):
    eq_im = exposure.equalize_hist(im)
    return eq_im*255

class SynDataset(data.Dataset):
    def __init__(self, image_num=100000, dimension=(512, 512), range_of_columns=(40, 200), sigmas=(1, 8), max_intensity=100, reference=None):
        # Initialize file path or list of file names.
        self.image_num = image_num
        self.dimension = dimension
        self.range_of_columns = range_of_columns
        self.random_sigma = np.random.randint(sigmas[0], sigmas[1], image_num)
        self.max_intensity = max_intensity

        self.reference = reference


    def __getitem__(self, item):
        rand_num = random.randint(0, len(self.reference)-1)
        ref = equlization(self.reference[rand_num])
        simulated_image, gt, _ = simulate(dimension=self.dimension,
                                          range_of_columns=self.range_of_columns,
                                          sigma=self.random_sigma[item],
                                          max_intensity=self.max_intensity,
                                          reference=ref)

        # image = normalize(simulated_image)
        gt = normalize(gt)
        image = np.expand_dims(simulated_image, 0)
        gt = np.expand_dims(gt, 0)
        # image = np.array(image * 65535.0, dtype='uint16')
        # gt = np.array(gt * 65535.0, dtype='uint16')

        return image, gt

    def __len__(self):
        return self.image_num

if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt

    reference = imageio.imread('./Data/temp0.png')
    ds = SynDataset(reference=reference)


    # print(len(ds)
    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for i in range(len(ds)):
        sample, gt = ds[i]
        ax1.imshow(sample, cmap='gray')
        ax2.imshow(gt, cmap='gray')
        plt.tight_layout()
        plt.show()
        if i == 1:
            break




