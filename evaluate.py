from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from .models import get_model


@dataclass
class Metrics:
    model: str
    imbalance: str
    filter: str
    edge: str
    accuracy: float
    f1_macro: float
    f1_apis: float
    f1_bombus: float


def evaluate_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "lr",
    imbalance: str = "none",
    test_size: float = 0.3,
    random_state: int = 1,
):
    """
    Fit a model under a given imbalance strategy and return metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    class_weight = None

    if imbalance == "smote":
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    if imbalance == "class_weight":
        class_weight = "balanced"

    model = get_model(model_name, class_weight=class_weight, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_apis = f1_score(y_test, y_pred, pos_label=0)
    f1_bombus = f1_score(y_test, y_pred, pos_label=1)

    return model, acc, f1_macro, f1_apis, f1_bombus
