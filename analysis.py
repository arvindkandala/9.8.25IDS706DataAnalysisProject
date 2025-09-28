import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path

REQUIRED_FOR_FILTER = {"age", "sex"}
REQUIRED_FOR_GROUP = {"sex", "chol"}
REQUIRED_FOR_PREP = {"num"}


def _require_columns(df: pd.DataFrame, required: set):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def filter_old_men(df: pd.DataFrame) -> pd.DataFrame:
    """return rows for men with age>=60"""
    _require_columns(df, REQUIRED_FOR_FILTER)
    return df[(df["age"] >= 60) & (df["sex"] == "Male")]


def group_sex_chol(df: pd.DataFrame) -> pd.DataFrame:
    """group by sex and get stats for cholesterol"""
    _require_columns(df, REQUIRED_FOR_GROUP)
    return df.groupby("sex")["chol"].agg(
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        mean="mean",
        med="median",
        min="min",
        max="max",
        cnt="count",
    )


def preprocess(df: pd.DataFrame, target_col: str = "num"):
    """split features/target and one-hot encode categoricals"""
    _require_columns(df, {target_col})
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train_rf(X, y, n_estimators: int = 50, random_state: int = 24):
    """train random forest and return model and accuracy"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                                random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return rf, acc


def plot_top_features(model: RandomForestClassifier,
                       X, top_k: int = 5, 
                       outpath: str = "IDS706HeartRateVisualization.png") -> str:
    """save horizontal bar chart of most important heart disease factors"""
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top = importances.nlargest(top_k).sort_values()
    top.plot(kind="barh")
    plt.title("most important predictors for heart disease")
    plt.tight_layout()
    outpath = str(outpath)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    return outpath


def main():
    df = load_data("cleanned.csv")

    # quick EDA on console
    print(df.head())
    print(df.info())
    print(df.describe())

    oldmen = filter_old_men(df)
    print(oldmen)
    print(oldmen.shape)

    sexChol = group_sex_chol(df)
    print(sexChol)

    X, y = preprocess(df, target_col="num")
    model, acc = train_rf(X, y, n_estimators=50, random_state=24)
    print(f"accuracy is {acc:.3f}")

    out = plot_top_features(model, X, top_k=5, outpath="IDS706HeartRateVisualization.png")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()