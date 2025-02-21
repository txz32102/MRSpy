from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

class DrawChar:
    def __init__(self, text, font_filepath, font_size=128, image_size=(256, 256),
                 font_color=255, bg_color=0, offset_x=0, offset_y=0):
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