import numpy as np


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    """
    This function handles both 2D grayscale images and 3D RGB images.
    For RGB images, it first converts the RGB values to linear space, applies the grayscale
    conversion formula, and then transforms the result back to sRGB space.

    Returns:
    np.ndarray: A grayscale image represented as a NumPy ndarray. The output is a 2D array for grayscale images.
    """
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")

    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]

    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255

    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]


def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    This function first validates the dimensions and sizes of the specified image area to be pixelated.
    It then applies pixelation to the specified area and returns the pixelated image, an array indicating the
    areas of the image that are known (not pixelated), and a copy of the target array (the original image area).

    Parameters:
    image : The input image as a NumPy ndarray with shape (..., 1, H, W).
    x : The x-coordinate of the top-left corner of the area to be pixelated.
    y : The y-coordinate of the top-left corner of the area to be pixelated.
    width : The width of the area to be pixelated.
    height : The height of the area to be pixelated.
    size : The size of the blocks used for pixelation.

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the pixelated image, an array indicating
                                               the known unpixelated areas, and a copy of the target array.
    """

    if image.ndim < 3 or image.shape[-3] != 1:
        # This is actually more general than the assignment specification
        raise ValueError("image must have shape (..., 1, H, W)")
    if width < 2 or height < 2 or size < 2:
        raise ValueError("width/height/size must be >= 2")
    if x < 0 or (x + width) > image.shape[-1]:
        raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
    if y < 0 or (y + height) > image.shape[-2]:
        raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")

    # The (height, width) slices to extract the area that should be pixelated. Since we
    # need this multiple times, specify the slices explicitly instead of using [:] notation
    area = (..., slice(y, y + height), slice(x, x + width))

    # This returns already a copy, so we are independent of "image"
    pixelated_image = pixelate(image, x, y, width, height, size)

    known_array = np.ones_like(image, dtype=bool)
    known_array[area] = False

    # Create a copy to avoid that "target_array" and "image" point to the same array
    target_array = image[area].copy()

    return pixelated_image, known_array, target_array


def pixelate(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> np.ndarray:
    """
    Applies a pixelation effect to a specified area of the image.

    This function modifies the specified area of the image by averaging the color values in blocks of a given size,
    creating a pixelated effect. The pixelation is applied in-place, and a copy of the modified image is returned.

    """
    # Need a copy since we overwrite data directly
    image = image.copy()
    curr_x = x

    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (..., slice(curr_y, min(curr_y + size, y + height)), slice(curr_x, min(curr_x + size, x + width)))
            image[block] = image[block].mean()
            curr_y += size
        curr_x += size

    return image
