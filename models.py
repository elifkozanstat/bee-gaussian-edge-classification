from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_model(model_name: str, class_weight=None, random_state: int = 1):
    """
    Construct a scikit-learn classifier given its short name.

    Parameters
    ----------
    model_name : {'lr', 'svm', 'rf', 'gb'}
    class_weight : None or 'balanced'
        Used for LR, SVM, RF. Ignored for GB.
    """
    if model_name == "lr":
        return LogisticRegression(
            max_iter=500,
            class_weight=class_weight,
            n_jobs=-1,
        )
    if model_name == "svm":
        return SVC(
            kernel="rbf",
            class_weight=class_weight,
            probability=False,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
    if model_name == "gb":
        # GradientBoostingClassifier does not support class_weight
        return GradientBoostingClassifier(random_state=random_state)

    raise ValueError(f"Unknown model: {model_name}")
