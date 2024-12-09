from PIL import Image

def combine_images_vertically(images, padding=50, background_color=(255, 255, 255)):
    """
    Combines a list of images into a single vertically stacked image with padding between each image.
    Each image is aligned to the left, with whitespace added to match the width of the widest image.

    Parameters:
        images (list): List of PIL.Image objects.
        padding (int): Vertical and horizontal whitespace between images and edges.
        background_color (tuple): RGB color for the background.

    Returns:
        PIL.Image: Combined image.
    """
    # Calculate the maximum width and total height
    max_width = max(image.width for image in images)
    total_height = sum(image.height for image in images) + (len(images) - 1) * padding
    
    # Create a new image with the calculated dimensions
    combined_image = Image.new(
        "RGB", (max_width + padding, total_height + padding), color=background_color
    )
    
    # Paste each image into the combined image with padding
    y_offset = padding
    for image in images:
        combined_image.paste(image, (padding, y_offset))
        y_offset += image.height + padding
    
    return combined_image

# Example usage
if __name__ == "__main__":
    # Load images
    image_paths = ["llava_v1_5_radar.jpg", "downloaded_image.jpg"]
    images = [Image.open(image_path) for image_path in image_paths]

    # Combine images
    combined_image = combine_images_vertically(images)

    # Save or display the result
    combined_image.save("combined_image.jpg")
    combined_image.show()