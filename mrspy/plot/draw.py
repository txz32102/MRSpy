from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from typing import Tuple, Union

class DrawChar:
    def __init__(self, 
                 text: str, 
                 font_filepath: str, 
                 font_size: int = 128, 
                 image_size: Tuple[int, int] = (256, 256),
                 font_color: Union[int, Tuple[int, int, int]] = 255, 
                 bg_color: Union[int, Tuple[int, int, int]] = 0, 
                 offset_x: int = 0, 
                 offset_y: int = 0, 
                 xy: Tuple[int, int] = None):
        self.text = text
        self.font_filepath = font_filepath
        self.font_size = font_size
        self.image_size = image_size
        self.offset_x = offset_x
        self.offset_y = offset_y

        self.font_color, self.bg_color, self.color_mode = self._process_colors(font_color, bg_color)

        self.font = ImageFont.truetype(self.font_filepath, self.font_size)
        self.bbox = self.font.getbbox(self.text)

        self.pad_x = -self.bbox[0]  # Horizontal padding offset
        self.pad_y = -self.bbox[1]  # Vertical padding offset
        self.xy = xy

        self.image = self._create_base_image()

    def _process_colors(self, font_color, bg_color):
        color_mode = 'L'
        fc_is_tuple = isinstance(font_color, tuple)
        bc_is_tuple = isinstance(bg_color, tuple)

        if fc_is_tuple or bc_is_tuple:
            lengths = []
            if fc_is_tuple:
                lengths.append(len(font_color))
            if bc_is_tuple:
                lengths.append(len(bg_color))
            
            if lengths:
                unique_lengths = set(lengths)
                if len(unique_lengths) > 1:
                    raise ValueError("font_color and bg_color tuple lengths must match")
                
                length = lengths[0]
                if length == 4:
                    color_mode = 'RGBA'
                elif length == 3:
                    color_mode = 'RGB'
                else:
                    raise ValueError("Color tuple must be of length 3 (RGB) or 4 (RGBA)")

                # Convert non-tuple colors to the corresponding mode
                if not fc_is_tuple:
                    if color_mode == 'RGB':
                        font_color = (font_color,) * 3
                    else:
                        font_color = (font_color,) * 3 + (255,)
                if not bc_is_tuple:
                    if color_mode == 'RGB':
                        bg_color = (bg_color,) * 3
                    else:
                        bg_color = (bg_color,) * 3 + (255,)
            else:
                color_mode = 'L'
        else:
            # Handle grayscale case
            color_mode = 'L'
            font_color = int(font_color)
            bg_color = int(bg_color)

        return font_color, bg_color, color_mode

    def _create_base_image(self):
        """Create the base character image"""
        W, H = self.image_size
        image = Image.new(self.color_mode, (W, H), self.bg_color)
        draw = ImageDraw.Draw(image)

        # Automatically calculate the centered coordinates
        char_width = self.bbox[2] - self.bbox[0]
        char_height = self.bbox[3] - self.bbox[1]
        if(self.xy is not None):
            x = self.xy[0] - self.bbox[0] + self.offset_x 
            y = self.xy[1] - self.bbox[1] + self.offset_y
        else:
            x = (W - char_width) // 2 - self.bbox[0] + self.offset_x
            y = (H - char_height) // 2 - self.bbox[1] + self.offset_y
        draw.text((x, y), self.text, font=self.font, fill=self.font_color)
        return image

    def rotated(self, angle):
        """Generate a rotated image"""
        return self.image.rotate(angle, expand=0)
    
    def to_numpy(self):
        """Convert the image to a NumPy array"""
        return np.array(self.image)

    def to_tensor(self, device='cpu'):
        """Convert the image to a PyTorch tensor"""
        img_np = np.array(self.image)
        img_tensor = torch.from_numpy(img_np).float()
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
        else:
            img_tensor = img_tensor.permute(2, 0, 1)
        
        return img_tensor.to(device)

    def save(self, filepath):
        """Save the image to a file"""
        self.image.save(filepath)
    def show(self):
        """Display the image (requires matplotlib)"""
        try:
            from matplotlib import pyplot as plt
            if self.color_mode == 'L':
                plt.figure()
                plt.imshow(np.array(self.image), cmap='gray')
            else:
                plt.figure()
                plt.imshow(np.array(self.image))
            plt.show()
        except ImportError:
            raise RuntimeError("matplotlib is required to display the image")
        
    def get_char_corners(self):
        """
        Get the coordinates of the four corners of the character.
        Returns:
            list: A list of tuples representing the coordinates of the four corners.
                Order: (left_top, right_top, left_bottom, right_bottom)
        """
        # Get the bounding box coordinates
        x0, y0, x1, y1 = self.bbox

        # Calculate the actual drawing coordinates based on the offset and image size
        W, H = self.image_size
        char_width = x1 - x0
        char_height = y1 - y0

        if self.xy is not None:
            x = self.xy[0] + self.offset_x
            y = self.xy[1] + self.offset_y
        else:
            x = (W - char_width) // 2 + self.offset_x
            y = (H - char_height) // 2 + self.offset_y

        # Ensure the character is within the image boundaries
        x = max(0, min(x, W - char_width))
        y = max(0, min(y, H - char_height))

        # Calculate the four corners
        left_top = (x, y)
        right_top = (x + char_width, y)
        left_bottom = (x, y + char_height)
        right_bottom = (x + char_width, y + char_height)

        return [left_top, right_top, left_bottom, right_bottom]
    
    def add_gaussian_noise_to_region(self, corners: list, noise_level: float = 0.1):
        """
        Add Gaussian noise to a specific rectangular region in the image.
        
        Args:
            corners (list): List of four corner coordinates [left_top, right_top, left_bottom, right_bottom].
                            Each corner is a tuple (x, y).
            noise_level (float): Noise intensity level (0 to 1). Default is 0.1.
        """
        # Extract coordinates
        left_top, right_top, left_bottom, right_bottom = corners
        x0, y0 = left_top
        x1, y1 = right_bottom

        # Ensure coordinates are within image boundaries
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.image.width, x1), min(self.image.height, y1)

        # Convert image to numpy array
        image_array = np.array(self.image)

        # Extract the region of interest (ROI)
        region = image_array[y0:y1, x0:x1]

        # Calculate the average pixel value in the region
        if self.color_mode == 'L':  # Grayscale
            avg_pixel_value = np.mean(region)
        else:  # RGB or RGBA
            avg_pixel_value = np.mean(region, axis=(0, 1))  # Average across spatial dimensions

        # Calculate noise parameters
        mean = 0
        std_dev = noise_level * avg_pixel_value  # Scale standard deviation based on average pixel value

        # Generate Gaussian noise for the specified region
        noise = np.random.normal(mean, std_dev, region.shape)

        # Add noise to the region
        noisy_region = np.clip(region + noise, 0, 255).astype(np.uint8)
        image_array[y0:y1, x0:x1] = noisy_region

        # Convert back to PIL image
        self.image = Image.fromarray(image_array)

    def increase_pixel_value_in_region(self, corners: list, increase_value: int = 10):
        """
        Increase pixel values in a specific rectangular region of the image by a given amount.

        Args:
            corners (list): List of four corner coordinates [left_top, right_top, left_bottom, right_bottom].
                            Each corner is a tuple (x, y).
            increase_value (int): The value to be added to each pixel in the region. Default is 10.
        """
        # Extract coordinates
        left_top, right_top, left_bottom, right_bottom = corners
        x0, y0 = left_top
        x1, y1 = right_bottom

        # Ensure coordinates are within image boundaries
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.image.width, x1), min(self.image.height, y1)

        # Convert image to numpy array
        image_array = np.array(self.image)

        # Extract the region of interest (ROI)
        region = image_array[y0:y1, x0:x1]

        # Increase pixel values in the region
        increased_region = np.clip(region + increase_value, 0, 255).astype(np.uint8)

        # Update the region in the image array
        image_array[y0:y1, x0:x1] = increased_region

        # Convert back to PIL image
        self.image = Image.fromarray(image_array)