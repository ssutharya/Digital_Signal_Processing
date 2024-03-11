from IPython.display import display, Image, Markdown
import cv2
import numpy as np

def calculate_psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def apply_and_display_filters(image_path, kernel_sizes, noise_sd, filter_type='blur'):
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                                    # <- greyscale the image

    display(Image(data=cv2.imencode('.png', image_gray)[1]))                                     # <- original

    noise = np.random.normal(0, noise_sd, image_gray.shape).astype(np.uint8)
    noisy_image = cv2.add(image_gray, noise)

    display(Image(data=cv2.imencode('.png', noisy_image)[1]))                                    # <- noisy image

    for size in kernel_sizes:
        if filter_type == 'blur':
            filtered_image = cv2.blur(noisy_image, (size, size))
        elif filter_type == 'gaussian':
            filtered_image = cv2.GaussianBlur(noisy_image, (size, size), 0)

        psnr_value = calculate_psnr(image_gray, filtered_image)                                  # <- psnr value
        display(Markdown(f"**{size}x{size} {filter_type.capitalize()} Filter** - PSNR: {psnr_value:.2f}"))

        display(Image(data=cv2.imencode('.png', filtered_image)[1]))                             # <- output

image_path = '/content/tree.jpg'                                                                 # <- image path
kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]                                                      # <- masks
noise_sd = 10                                                                                    # <- standard deviation of the noise

apply_and_display_filters(image_path, kernel_sizes, noise_sd, filter_type='gaussian')
