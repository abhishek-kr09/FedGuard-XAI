import pandas as pd

from src.preprocessing import prepare_frame


def test_prepare_frame_encodes_numeric_and_categorical_columns():
    frame = pd.DataFrame({"bytes": [1, 2], "protocol": ["tcp", "udp"]})
    prepared = prepare_frame(frame, [0, 1])
    assert prepared.features.shape == (2, 3)
    assert len(prepared.feature_names) == 3
