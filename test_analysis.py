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
