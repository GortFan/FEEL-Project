import numpy as np

def rectangle_mask(matrix, centroid_row, centroid_col, rect_height, rect_width, angle=0.0, taper_x=1.0, taper_y=1.0):
    """
    Create a 2D boolean mask of the same shape as `matrix` with a rectangle of True values,
    where the rectangle is centered at (centroid_row, centroid_col), rotated by `angle`,
    and optionally tapered along the x and y axes.

    Parameters:
        matrix (np.ndarray): 2D array whose shape determines the mask size.
        centroid_row (float or int): Row index of the rectangle centroid.
        centroid_col (float or int): Column index of the rectangle centroid.
        rect_height (int): Height of the rectangle.
        rect_width (int): Width of the rectangle.
        angle (float): Rotation angle in degrees (counterclockwise, around the center).
        taper_x (float): Taper factor along x (width) axis, 0 to 1. 1=no taper, 0=full taper to a point.
        taper_y (float): Taper factor along y (height) axis, 0 to 1. 1=no taper, 0=full taper to a point.

    Returns:
        np.ndarray: 2D boolean mask.
    """
    height, width = matrix.shape

    # Create coordinate grid
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]

    # Shift grid so centroid is at (0, 0)
    y = rows - centroid_row
    x = cols - centroid_col

    # Convert angle to radians
    theta = np.deg2rad(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Apply rotation (inverse transform)
    x_rot = cos_theta * x + sin_theta * y
    y_rot = -sin_theta * x + cos_theta * y

    # Rectangle bounds (centered at origin after shift)
    half_h = rect_height / 2.0
    half_w = rect_width / 2.0

    # Tapering logic
    taper_x = np.clip(taper_x, 0.0, 1.0)
    taper_y = np.clip(taper_y, 0.0, 1.0)

    if taper_x < 1.0:
        width_at_y = half_w * (1 - (1 - taper_x) * (np.abs(y_rot) / half_h))
        width_at_y = np.clip(width_at_y, 0, half_w)
    else:
        width_at_y = half_w

    if taper_y < 1.0:
        height_at_x = half_h * (1 - (1 - taper_y) * (np.abs(x_rot) / half_w))
        height_at_x = np.clip(height_at_x, 0, half_h)
    else:
        height_at_x = half_h

    mask = (
        (np.abs(x_rot) <= width_at_y) &
        (np.abs(y_rot) <= height_at_x)
    )
    return mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example matrix
    matrix = np.ones((25, 100))
    centroid_row, centroid_col = 0, 50
    rect_height, rect_width = 4, 69
    angle = 4
    taper_x = 1
    taper_y = 0.6

    mask = rectangle_mask(matrix, centroid_row, centroid_col, rect_height, rect_width, angle, taper_x, taper_y)

    plt.imshow(mask, cmap='gray')
    plt.title("Rectangle Mask Visualization")
    plt.show()