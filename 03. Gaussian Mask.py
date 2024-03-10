from IPython.display import display, Image, Markdown
import cv2
import numpy as np

def gaussian(image_path, kernel_sizes):
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    display(Image(data=cv2.imencode('.png', image_gray)[1]))
    
    for size in kernel_sizes:
        filtered_image = cv2.GaussianBlur(image_gray, (size, size), 0)                          # <- greyscaling image

        psnr_value = calculate_psnr(image_gray, filtered_image)                                 # <- calc psnr
        display(Markdown(f"**{size}x{size} Gaussian Filter** - PSNR: {psnr_value:.2f}"))
        
        display(Image(data=cv2.imencode('.png', filtered_image)[1]))                            # <- filtered image output

image_path = '/content/tree.jpg'
kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]

gaussian(image_path, kernel_sizes)
