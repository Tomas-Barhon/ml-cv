import argparse
from contextlib import AsyncExitStack

import numpy as np
import skimage.io
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument(
    "--task",
    default="arrays",
    type=str,
    choices=["arrays", "functions", "image"],
    help="Selected task. One of 'arrays', 'functions' or 'image'.",
)


def arrays(args: argparse.Namespace) -> None:
    """
    Simple exercises for verifying basic knowledge of creating and reshaping numpy arrays.
    Study and try out the related functions if you don't know them.
    """
    # TODO: Create a list with numbers from 0 to 9.
    numbers = [range(10)]
    print(f"Python list: {numbers}")
    # TODO: Use 'np.asarray' to convert the list into numpy array. Use the parameter
    # 'dtype=int' to create an integer array.
    ascend_int = np.asarray(numbers, dtype=int)
    print(f"Int array:   {ascend_int}")
    # TODO: Use 'np.asarray' to convert the list into numpy array. Use the parameter
    # 'dtype=float' to create a float array.
    ascend_float = np.asarray(numbers, dtype=float)
    print(f"Float array: {ascend_float}")
    # TODO: Create a float array of zeros with shape (10,) using the function 'np.zeros'.
    zeros = np.zeros(10, dtype=float)
    print(f"Zeros:       {zeros}")
    # TODO: Create a 2D int array of ones with shape (5, 10) using the function 'np.ones'.
    ones = np.ones((5, 10))
    print(f"Ones:\n{ones}")

    # TODO: Use 'np.arange' to create an array of numbers from 5 to 25 with step 2.
    # It works the same as Python 'range'.
    arange_arr = np.arange(5, 26, 2)
    print(f"Arange array: {arange_arr}")
    # TODO: Use 'np.linspace' to compute 9 equidistantly spaced values in
    # the itnerval [0, 1] (including the boundary values ... 'endpoint=True').
    linspace_arr = np.linspace(0, 1, 9)
    print(f"Linspace array: {linspace_arr}")

    # TODO: Use 'np.mgrid[a_min:a_max, b_min:b_max]' to compute 2D array element
    # coordinates. It returns two arrays - row and column coordinates.
    rr, cc = np.mgrid[0:10, 0:10]
    print(f"Row coordinates:\n{rr}")
    print(f"Column coordinates:\n{cc}")

    ascend_1x30 = np.asarray([i for i in range(30)], dtype=int)
    ascend_shape = ascend_1x30.shape
    print(f"Ascending array shape: {ascend_shape}")
    print(f"Ascending 1x30:\n{ascend_1x30}")
    # TODO: Use 'np.reshape' to transform the 1x30 array into 3x10 array.
    ascend_3x10 = np.reshape(ascend_1x30, (3, 10))
    print(f"Ascending 3x10:\n{ascend_3x10}")
    # TODO: Use 'np.reshape' to transform the 3x10 array into 2x3x5 array.
    ascend_2x3x5 = np.reshape(ascend_3x10, (2, 3, 5))
    print(f"Ascending 2x3x5:\n{ascend_2x3x5}")

    # TODO: Use either 'np.transpose' or 'array.T' to transpose the 3x10 array.
    transposed = ascend_3x10.T
    print(f"Transposition of the 3x10 array:\n{transposed}")


def functions(args: argparse.Namespace) -> None:
    """
    Simple exercises for using functions and operations on numpy arrays.
    Study these exercises if you are not experienced with using numpy arrays.
    """
    # Creating a matrix of data with some randomness. (seed=42 makes it replicable)
    rng = np.random.RandomState(seed=42)
    data = np.reshape(rng.permutation(np.arange(30, dtype=float)), [5, 6])
    data = data**2
    # TODO: USe "np.ravel" to compute flattened data - reduces multi-dimensional arrays
    # to one 1D array.
    flat_data = np.ravel(data)
    # TODO: Use "np.argmax" and "np.argmin" to find the indices of the maximum and minimum
    # of the flattened array.
    all_max_idx = np.argmax(flat_data)
    all_min_idx = np.argmin(flat_data)
    # TODO: Use the above indices to get the maximum and minimum in the array.
    all_max = flat_data[all_max_idx]
    all_min = flat_data[all_min_idx]
    # TODO: Compute column-wise maximum, minimum, mean, median and sum of "data" array.
    # - Use functions "np.max", "np.min", "np.mean", "np.median", "np.sum" with parameter
    #   "axis=0" (0 - column-wise, 1 - row-wise, etc.).
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    col_mean = np.mean(data, axis=0)
    col_median = np.median(data, axis=0)
    col_sum = np.sum(data, axis=0)
    print(f"Source data:\n{data}")
    print(
        (
            f"Maximum (Idx/Value): {all_max_idx}/{all_max} "
            f"and minimum (Idx/Value): {all_min_idx}/{all_min}"
        )
    )
    print(f"Column-wise maximum: {col_max}")
    print(f"Column-wise minimum: {col_min}")
    print(f"Column-wise mean:    {col_mean}")
    print(f"Column-wise median:  {col_median}")
    print(f"Column-wise sum:     {col_sum}")

    x_data = np.asarray([0, 1, 2, 6, 7, 13], dtype=float)
    y_data = np.asarray([0, 5, 2, 1, 3, 6], dtype=float)
    # TODO: Compute the result of a function "1/2 * x^3 - 5 * x * y^2 + 3 * x + 3" from
    # 'x_data' and 'y_data'.
    # - Do it for all data points at once (the numpy way - mathematical operations on
    #   arrays are computed element-wise).
    result = 0.5 * x_data**3 - 5 * x_data * y_data**2 + 3 * x_data + 3
    print(f"\nX Data: {x_data}")
    print(f"Y Data: {y_data}")
    print(f"Result: {result}")

    # TODO: Compute the rotation of a vector around an origin point using matrix
    # multiplication.
    h_vec = np.asarray([4, 6, 1], float)
    angle = 45 / 180 * np.pi
    origin = np.asarray([2, 4], float)
    target = np.asarray([2, 4 + np.sqrt(8)], float)
    # TODO: Define the 3x3 rotation matrix as a numpy array (use "np.cos", "np.sin" for
    # cosine and sine):
    #   [cos(angle), -sin(angle), 0]
    #   [sin(angle), cos(angle),  0]
    #   [0, 0, 1]
    rot_mat = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    # TODO: Define translation matrices as numpy arrays for translation to and from
    # the local frame of the origin:
    #   [1, 0, translate_x]
    #   [0, 1, translate_y]
    #   [0, 0, 1]
    to_orig = np.array([[1, 0, -origin[0]], [0, 1, -origin[1]], [0, 0, 1]])
    from_orig = np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]])
    # TODO: Convert the row vector 'h_vec' into column vector 'v_vec' by either
    # 'np.c_[h_vec]' or reshaping.
    v_vec = h_vec.T
    # TODO: Use matrix multiplication ('np.dot', 'np.matmul' or '@') to compute
    # the transformation.
    # - The order is: translation to the local frame, rotation and translation from
    #   the local frame.
    res_vec = from_orig @ rot_mat @ to_orig

    fig, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={"aspect": "equal"})
    ax.set_title("MatMul transformation")
    ax.scatter(h_vec[0], h_vec[1], c="blue", label="Point")
    ax.scatter(origin[0], origin[1], c="red", label="Origin")
    ax.scatter(res_vec[0], res_vec[1], c="green", label="Transformed point")
    ax.scatter(
        target[0], target[1], s=150, facecolors="none", edgecolors="red", label="Target"
    )
    ax.legend()
    ax.set_xlim((0, 6))
    ax.set_ylim((0, 8))
    fig.tight_layout()
    plt.show()


def image_like_arrays(args: argparse.Namespace) -> None:
    """
    Simple manipulation of images in numpy - a loaded image becomes a 2D [H, W] or
    3D [H, W, C] ndarray.
    - We load images through 'skimage.io.imread' but using PIL or OpenCV is also allowed,
      however, skimage is concise and tends to work as expected.
    """
    img: np.ndarray = skimage.io.imread("discord_avatar_default.png")
    img_shape = img.shape
    print(f"Image shape: {img_shape}")
    pad_width = 20
    white_threshold = 250

    # TODO: Use 'np.pad' to extend the image in spatial dimensions by 'pad_width'. Do not
    # extend the colour channel dimension.
    # - 3D Padding is written in the form [(r_t, r_b), (c_l, c_r), (ch_f, ch_b)]
    padded = np.pad(
        img,
        ((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    # TODO: Create a mask with 'True' values where all 3 colour channels have value
    # greater than 'white_threshold' and 'False' otherwise.
    # - Use comparison operators on arrays and function 'np.logical_and'. (There is also
    #   'np.logical_or', 'np.logical_xor', 'np.logical_not')
    mask = np.logical_and.reduce(padded > white_threshold, axis=2)
    # TODO: Use 'np.mgrid' to get pixel coordinate arrays and compute minimum and maximum
    # coordinates of the 'True' pixels in the mask.
    # - You can use 'mask' as an index into an array - mask has to be bool or integer and
    #   it has to have matching shape with the array it is used in.
    rr, cc = None, None
    min_r, max_r = None, None
    min_c, max_c = None, None
    # TODO: Crop the padded image to the minima and maxima computed in the previous step.
    # Keep all colour channels.
    cropped = None

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_title("Original image")
    ax[0].imshow(img)
    ax[1].set_title("Padded image")
    ax[1].imshow(padded if padded is not None else np.zeros_like(img))
    ax[2].set_title("Cropped image")
    ax[2].imshow(cropped if cropped is not None else np.zeros_like(img))
    fig.tight_layout()
    plt.show()


def main(args: argparse.Namespace) -> None:
    tasks = {
        "arrays": arrays,
        "functions": functions,
        "image": image_like_arrays,
    }
    if args.task not in tasks.keys():
        raise ValueError("Unrecognised task: '{}'.".format(args.task))
    tasks[args.task](args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
