import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from velsera.utils.files import load_df, save


@pytest.fixture
def sample_df():
    """Fixture for creating a sample DataFrame."""
    return pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})


def test_load_df_csv(sample_df, tmp_path):
    """Test loading a DataFrame from a CSV file."""
    file_path = tmp_path / "test.csv"
    sample_df.to_csv(file_path, index=False)
    loaded_df = load_df(file_path)
    assert_frame_equal(loaded_df, sample_df)


def test_load_df_parquet(sample_df, tmp_path):
    """Test loading a DataFrame from a Parquet file."""
    file_path = tmp_path / "test.parquet"
    sample_df.to_parquet(file_path, index=False)
    loaded_df = load_df(file_path)
    assert_frame_equal(loaded_df, sample_df)


def test_load_df_json(sample_df, tmp_path):
    """Test loading a DataFrame from a JSON file."""
    file_path = tmp_path / "test.json"
    sample_df.to_json(file_path, orient="records")
    loaded_df = load_df(file_path)
    assert_frame_equal(loaded_df, sample_df)


def test_load_df_invalid_path():
    """Test loading from a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_df("non_existent_file.csv")


def test_load_df_unsupported_extension(tmp_path):
    """Test loading from an unsupported file extension."""
    file_path = tmp_path / "test.txt"
    file_path.touch()
    with pytest.raises(
        KeyError
    ):  # Expecting KeyError from the internal _func_to_load mapper
        load_df(file_path)


# --- Tests for the save decorator ---


def test_save_decorator_csv(sample_df, tmp_path):
    """Test the save decorator with a CSV file."""
    file_path = tmp_path / "decorated_save.csv"

    @save(file_path)
    def func_to_save(*args, **kwargs):
        return sample_df

    result_df = func_to_save(1, b=2)

    assert file_path.exists()
    loaded_df = pd.read_csv(file_path)
    assert_frame_equal(loaded_df, sample_df)
    assert_frame_equal(result_df, sample_df)


def test_save_decorator_parquet(sample_df, tmp_path):
    """Test the save decorator with a Parquet file."""
    file_path = tmp_path / "decorated_save.parquet"

    @save(file_path)
    def func_to_save():
        return sample_df

    result_df = func_to_save()

    assert file_path.exists()
    loaded_df = pd.read_parquet(file_path)
    assert_frame_equal(loaded_df, sample_df)
    assert_frame_equal(result_df, sample_df)


def test_save_decorator_json(sample_df, tmp_path):
    """Test the save decorator with a JSON file."""
    file_path = tmp_path / "decorated_save.json"

    @save(file_path)
    def func_to_save():
        return sample_df

    result_df = func_to_save()

    assert file_path.exists()
    loaded_df = pd.read_json(file_path, orient="columns")
    assert_frame_equal(loaded_df, sample_df)
    assert_frame_equal(result_df, sample_df)


def test_save_decorator_unsupported_extension(sample_df, tmp_path):
    """Test the save decorator with an unsupported file extension."""
    file_path = tmp_path / "decorated_save.txt"

    @save(file_path)
    def func_to_save():
        return sample_df

    with pytest.raises(KeyError):
        func_to_save()
    assert not file_path.exists()
