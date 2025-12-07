import argparse
import os
import pandas as pd

from bee_gaussian_pipeline.data import build_feature_matrix
from bee_gaussian_pipeline.evaluate import evaluate_pipeline, Metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to train.csv with columns ['id', 'genus'].")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory with JPEG images.")
    parser.add_argument("--out_csv", type=str, default="results_all.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    filters = ["gaussian"]            # in the paper we only use Gaussian
    edges = ["sobel", "prewitt", "log", "canny"]
    models = ["lr", "svm", "rf", "gb"]
    imbalances = ["none", "class_weight", "smote"]

    records = []

    for f_name in filters:
        for e_name in edges:
            print(f"\n=== Filter: {f_name}, Edge: {e_name} ===")
            X, y = build_feature_matrix(
                df,
                img_dir=args.img_dir,
                filter_name=f_name,
                edge_name=e_name,
                add_gray=False,
            )
            for m_name in models:
                for imb in imbalances:
                    print(f"--- Model: {m_name}, Imbalance: {imb} ---")
                    _, acc, f1_macro, f1_apis, f1_bombus = evaluate_pipeline(
                        X,
                        y,
                        model_name=m_name,
                        imbalance=imb,
                    )
                    records.append(
                        {
                            "filter": f_name,
                            "edge": e_name,
                            "model": m_name,
                            "imbalance": imb,
                            "accuracy": acc,
                            "f1_macro": f1_macro,
                            "f1_apis": f1_apis,
                            "f1_bombus": f1_bombus,
                        }
                    )

    results_df = pd.DataFrame(records)
    results_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved results to {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
