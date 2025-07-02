# File: tests/test_loader.py
import pytest
import pandas as pd
from io import StringIO
from src.data.loader import load_dataset, list_sample_datasets

def test_list_sample_datasets(tmp_path, monkeypatch):
    # Create fake sample data directory
    sample_dir = tmp_path / "sample_data"
    sample_dir.mkdir()
    file = sample_dir / "iris.csv"
    file.write_text("col1,col2
1,2")
    # Monkeypatch Path in loader
    monkeypatch.setattr(
        'src.data.loader.Path', lambda *args, **kwargs: sample_dir.parent,
    )
    samples = list_sample_datasets()
    assert "iris" in samples


def test_load_dataset_csv(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b
1,2
3,4")
    df = load_dataset(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)