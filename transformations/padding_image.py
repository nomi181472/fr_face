import numpy as np
class PaddingTransform:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image):
        height, width, _ = image.shape
        if height < self.target_size or width < self.target_size:
            # Create a new black (zero) image of the target size
            padded_image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

            # Calculate the offset for centering the original image
            pad_height = (self.target_size - height) // 2
            pad_width = (self.target_size - width) // 2

            # Place the original image in the center of the padded image
            padded_image[pad_height:pad_height + height, pad_width:pad_width + width] = image
            return padded_image

        return image