from mrspy.plot import DrawChar

# Initialize parameters
text = '(0)'
# you can download the font file from https://github.com/justrajdeep/fonts/blob/master/Times%20New%20Roman.ttf
font_path = "/home/data1/musong/workspace/2025/1/1-30/ttf/Times New Roman.ttf"

# Create instance
drawer = DrawChar(
    text=text,
    font_filepath=font_path,
    font_size=128,
    bg_color=189,
    font_color=155,
    image_size=(256, 256),
    offset_x=10,
    offset_y=5
)

# Display the original image
drawer.show()

# Generate and display rotated image
rotated_img = drawer.rotated(90)

# Save the image
drawer.save("original.png")
rotated_img.save("rotated.png")