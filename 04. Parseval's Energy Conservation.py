# Parseval Energy Conservation Formula

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

image = imread('/content/Sea.jpg')

if len(image.shape) == 3:
    image = np.mean(image, axis=2)

spatial_domain_energy = np.sum(image**2)

fft_image = np.fft.fft2(image)

frequency_domain_energy = np.sum(np.abs(fft_image)**2) / image.size

print(f"Spatial Domain Energy: {spatial_domain_energy}")
print(f"Frequency Domain Energy: {frequency_domain_energy}")
