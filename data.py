import os
import numpy as np
import pandas as pd
import cv2

from .features import apply_filter, apply_edge


def load_gray_normalized(img_id: str, img_dir: str) -> np.ndarray:
    """
    Load a single image as grayscale in [0, 1].

    Parameters
    ----------
    img_id : str
        Image identifier without extension (e.g. '000001').
    img_dir : str
        Directory where JPEG images are stored.

    Returns
    -------
    gray : np.ndarray
        2D float32 array with values in [0, 1].
    """
    img_path = os.path.join(img_dir, f"{img_id}.jpg")
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 0–255
    gray = gray.astype(np.float32) / 255.0        # 0–1
    return gray


def preprocess_single(
    img_id: str,
    img_dir: str,
    filter_name: str = "gaussian",
    edge_name: str = "sobel",
    add_gray: bool = False,
) -> np.ndarray:
    """
    Apply filtering + edge detection to a single image and return a 1D feature vector.
    """
    gray = load_gray_normalized(img_id, img_dir)
    filtered = apply_filter(gray, filter_name=filter_name)
    edged = apply_edge(filtered, edge_name=edge_name)

    feat_edge = edged.flatten()

    if add_gray:
        feat_gray = gray.flatten()
        return np.concatenate([feat_gray, feat_edge])

    return feat_edge


def build_feature_matrix(
    df: pd.DataFrame,
    img_dir: str,
    filter_name: str = "gaussian",
    edge_name: str = "sobel",
    add_gray: bool = False,
    max_samples: int | None = None,
):
    """
    Construct X, y from a BeeSpotter-style dataframe with columns ['id', 'genus'].

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'id' and 'genus' (0 = Apis, 1 = Bombus).
    img_dir : str
        Directory with images.
    filter_name : {'gaussian', 'median', 'bilateral', 'none'}
    edge_name   : {'sobel', 'prewitt', 'log', 'canny', 'none'}
    add_gray    : bool
        If True, concatenates raw grayscale with edge map.
    max_samples : int or None
        Optional limit for debugging.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    """
    X_list, y_list = [], []

    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        img_id = row["id"]
        label = int(row["genus"])
        feats = preprocess_single(
            img_id=img_id,
            img_dir=img_dir,
            filter_name=filter_name,
            edge_name=edge_name,
            add_gray=add_gray,
        )
        X_list.append(feats)
        y_list.append(label)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y
