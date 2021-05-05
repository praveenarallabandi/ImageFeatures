def erode(image: np.array, window: int = 1) -> np.array:

    imageZerosLike = np.zeros_like(image)
    [yy, xx] = np.where(image > 0)

    off = np.tile(range(-window, window + 1), (2 * window + 1, 1))
    x_off = off.flatten()
    y_off = off.T.flatten()

    n = len(xx.flatten())
    x_off = np.tile(x_off, (n, 1)).flatten()
    y_off = np.tile(y_off, (n, 1)).flatten()

    ind = np.sqrt(x_off ** 2 + y_off ** 2) > window
    x_off[ind] = 0
    y_off[ind] = 0

    xx = np.tile(xx, ((2 * window + 1) ** 2))
    yy = np.tile(yy, ((2 * window + 1) ** 2))

    nx = xx + x_off
    ny = yy + y_off

    ny[ny < 0] = 0
    ny[ny > image.shape[0] - 1] = image.shape[0] - 1
    nx[nx < 0] = 0
    nx[nx > image.shape[1] - 1] = image.shape[1] - 1

    imageZerosLike[ny, nx] = 255

    return imageZerosLike.astype(np.uint8)