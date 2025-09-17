import pandas as pd
import numpy as np
import os
import analysis as a


def tiny_df():
    return pd.DataFrame({
        "age": [59, 60, 72, 45, 80],
        "sex": ["Male", "Male", "Female", "Female", "Male"],
        "chol": [200, 180, 220, 210, 199],
        "num": [0, 1, 1, 0, 1],  # binary target
    })

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