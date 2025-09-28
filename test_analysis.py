import pandas as pd
import os
import analysis as a


def tiny_df():
    return pd.DataFrame(
        {
            "age": [59, 60, 72, 45, 80],
            "sex": ["Male", "Male", "Female", "Female", "Male"],
            "chol": [200, 180, 220, 210, 199],
            "num": [0, 1, 1, 0, 1],  # binary target
        }
    )


def test_filter_old_men_selects_age_60_plus_and_male():
    df = tiny_df()
    out = a.filter_old_men(df)
    assert (out["age"] >= 60).all()
    assert (out["sex"] == "Male").all()
    assert len(out) == 2


def test_filter_old_men_missing_columns_raises():
    df = tiny_df().drop(columns=["sex"])
    try:
        a.filter_old_men(df)
        assert False, "Expected ValueError for missing column"
    except ValueError as e:
        assert "Missing required column" in str(e)


def test_group_sex_chol_returns_expected_aggregations():
    df = tiny_df()
    g = a.group_sex_chol(df)
    for col in ["q1", "q3", "mean", "med", "min", "max", "cnt"]:
        assert col in g.columns
    assert set(g.index) == {"Male", "Female"}
    # sanity: q1 <= q3
    assert (g["q1"] <= g["q3"]).all()


def test_preprocess_creates_dummies_and_separates_target():
    df = tiny_df()
    X, y = a.preprocess(df, target_col="num")
    assert "num" not in X.columns
    assert any(c.startswith("sex_") for c in X.columns)
    assert len(y) == len(df)


def test_preprocess_missing_target_raises():
    df = tiny_df().drop(columns=["num"])
    try:
        a.preprocess(df, target_col="num")
        assert False, "Expected ValueError for missing target"
    except ValueError as e:
        assert "Missing required column" in str(e)


def test_train_rf_returns_model_and_reasonable_accuracy():
    df = tiny_df()
    X, y = a.preprocess(df, target_col="num")
    model, acc = a.train_rf(X, y, n_estimators=10, random_state=0)
    assert hasattr(model, "predict")
    assert 0.0 <= acc <= 1.0


def test_plot_top_features_writes_file(tmp_path):
    df = tiny_df()
    X, y = a.preprocess(df, target_col="num")
    model, _ = a.train_rf(X, y, n_estimators=10, random_state=0)
    out = tmp_path / "feat.png"
    saved = a.plot_top_features(model, X, top_k=3, outpath=str(out))
    assert os.path.exists(saved) and os.path.getsize(saved) > 0
