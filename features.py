import numpy as np
import cv2


def apply_filter(img: np.ndarray, filter_name: str = "gaussian") -> np.ndarray:
    """
    Apply optional smoothing filter to a grayscale image in [0, 1].

    Parameters
    ----------
    img : np.ndarray
        Grayscale image in [0, 1].
    filter_name : {'none', 'gaussian', 'median', 'bilateral'}

    Returns
    -------
    np.ndarray
        Smoothed image in [0, 1].
    """
    if filter_name == "none":
        return img

    img_uint8 = (img * 255).astype(np.uint8)

    if filter_name == "gaussian":
        filtered = cv2.GaussianBlur(img_uint8, (3, 3), 0)
    elif filter_name == "median":
        filtered = cv2.medianBlur(img_uint8, 3)
    elif filter_name == "bilateral":
        filtered = cv2.bilateralFilter(img_uint8, 5, 75, 75)
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    return filtered.astype(np.float32) / 255.0


def apply_edge(img: np.ndarray, edge_name: str = "sobel") -> np.ndarray:
    """
    Apply edge detection to a grayscale image in [0, 1].

    Parameters
    ----------
    img : np.ndarray
        Grayscale image in [0, 1] (after smoothing).
    edge_name : {'none', 'sobel', 'prewitt', 'log', 'canny'}

    Returns
    -------
    np.ndarray
        Edge map in [0, 1].
    """
    img_uint8 = (img * 255).astype(np.uint8)

    if edge_name == "none":
        return img

    if edge_name == "sobel":
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    if edge_name == "prewitt":
        kernelx = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float32)
        kernely = np.array([[-1, -1, -1],
                            [ 0,  0,  0],
                            [ 1,  1,  1]], dtype=np.float32)
        gx = cv2.filter2D(img, cv2.CV_32F, kernelx)
        gy = cv2.filter2D(img, cv2.CV_32F, kernely)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    if edge_name == "log":
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        log = cv2.Laplacian(blur, cv2.CV_32F)
        return cv2.normalize(log, None, 0.0, 1.0, cv2.NORM_MINMAX)

    if edge_name == "canny":
        edges = cv2.Canny(img_uint8, 100, 200)
        return edges.astype(np.float32) / 255.0

    raise ValueError(f"Unknown edge operator: {edge_name}")
