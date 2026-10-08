
from io import TextIOWrapper
import numpy as np

def read_int(file : TextIOWrapper, byte_order : str) -> int:
    """
    Reads a 4-byte integer from the given file object using specific byte order.
    """
    b = file.read(4)
    return int.from_bytes(b, byte_order)

def trim_digits(digits : np.ndarray, border : int) -> np.ndarray:
    """
    Trims images given in the stack 'digits' by cropping border of width specified by 'border'.
    """
    return digits[:, border : digits.shape[1] - border, border : digits.shape[2] - border]

def normalize_pixel_values(digits : np.ndarray) -> np.ndarray:
    """
    Normalises the pixels values of 'digits' into the range 0-1.
    """
    return digits / 255.0

def load_mnist(
        image_file : str,
        label_file : str,
        read_digits : int = -1,
        offset : int = 0,
        trim : int = 0,
        normalize : bool = True
    ) -> tuple:
    """
    Summary:
    Loads images and labels of the mnist digits dataset from files given as the first
    two arguments. Optionally reads only a number of images given by readDigits
    argument. It can also skip part of the dataset by specifying the offset argument.
    Lastly, the function can optionally trim the images using a given border width
    (argument trim) and normalise pixel values into the range 0-1.

    Arguments:
    - 'image_file' - Path to the file with image data.
    - 'label_file' - Path to the file with label data.
    - 'read_digits' - Number of images read from the file (-1 means all of them).
    - 'offset' - Number of images skipped at the beginning.
    - 'trim' - Number of pixels cut from each side of the image.
    - 'normalize' - Whether the values in images should be scaled to interval from 0 to 1.

    Returns:
    - An (N,K,K) array of images where K is 28-trim*2 for mnist images with side length 28
      and N is the number of requested images.
    - An (N) array of integer labels for the requested images.
    """

    # Mnist is encoded using big endian.
    byte_order = "big"

    with open(image_file, "rb") as f:

        # Check image header identifier.
        header = read_int(f, byte_order)
        if header != 2051:
            raise ValueError("Invalid image file header!")

        # Check the number of images in the file and their compatibility with
        # offset and ReadDigits values.
        count = read_int(f, byte_order)
        if count < offset:
            raise ValueError("Offset exceeds the data in the file!")
        if read_digits != -1 and count < read_digits + offset:
            raise ValueError("Trying to read too many digits!")

        height = read_int(f, byte_order)
        width = read_int(f, byte_order)

        # Skip part of the dataset.
        if offset > 0:
            f.seek(height * width * offset, 1)

        # Read the data.
        if read_digits != -1:
            data = np.fromfile(f, np.uint8, height * width * read_digits)
        else:
            data = np.fromfile(f, np.uint8)

    # Reshape the int array into a series of images.
    # The first axis enumerates the individual images.
    images = np.reshape(data, (-1, height, width))

    with open(label_file, "rb") as f:

        # Check label header identifier.
        header = read_int(f, byte_order)
        if header != 2049:
            raise ValueError("Invalid image file header!")

        # Check the number of images in the file and their compatibility with
        # offset and ReadDigits values.
        count = read_int(f, byte_order)
        if count < offset:
            raise ValueError("Offset exceeds the data in the file!")
        if read_digits != -1 and count < read_digits + offset:
            raise ValueError("Trying to read too many digits!")

        # Skip part of the dataset.
        if offset > 0:
            f.seek(offset, 1)

        # Read the data.
        labels = np.fromfile(f, np.uint8, read_digits)

    # Optionally trim and normalise the images.
    if trim > 0:
        images = trim_digits(images, trim)
    if normalize:
        images = normalize_pixel_values(images)

    return images, labels
