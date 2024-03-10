import numpy as np

def gaussian_mask(size, sigma):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

np.set_printoptions(precision=4, suppress=True)

mask_size = 9
sigma = 1.0

gaussian_mask_9x9 = gaussian_mask(mask_size, sigma)

print("9x9 Gaussian Mask:")
print(gaussian_mask_9x9)
