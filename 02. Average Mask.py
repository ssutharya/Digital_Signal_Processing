from IPython.display import display, Image, Markdown
import cv2
import numpy as np

def calculate_psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def apply_and_display_filters(image_path, kernel_sizes):
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                     # <- greyscale the image

    display(Image(data=cv2.imencode('.png', image_gray)[1]))                      # <- original

    for size in kernel_sizes:
        filtered_image = cv2.blur(image_gray, (size, size))

        psnr_value = calculate_psnr(image_gray, filtered_image)                   # <- psnr value
        display(Markdown(f"**{size}x{size} Filter** - PSNR: {psnr_value:.2f}"))
        
        display(Image(data=cv2.imencode('.png', filtered_image)[1]))              # <_ output

image_path = '/content/tree.jpg'                                                  # <- image path
kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]                                       # <- masks

apply_and_display_filters(image_path, kernel_sizes)
