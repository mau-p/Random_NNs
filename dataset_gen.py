from sklearn.datasets import make_classification
import numpy as np
import pandas as pd


def _generate_data():
    num_features = 10
    num_classes = 5
    data = make_classification(
        n_samples=5000,
        n_features=num_features,
        n_informative=7,
        n_redundant=2,
        n_classes=num_classes,
        random_state=69)
    X_data = np.array(data[0])
    y_data = np.array(data[1])
    data = np.append(X_data, y_data.reshape(-1, 1), axis=1)
    col_names = [f"feature_{i}" for i in range(num_features+1)]
    col_names[-1] = "label"
    return pd.DataFrame(data, columns=col_names)


def main():
    _generate_data().to_csv("dataset.csv", index=False)
    print(f"Dataset saved to dataset.csv")


if __name__ == '__main__':
    main()