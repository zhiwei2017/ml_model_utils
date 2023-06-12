import os
import pytest
from unittest import mock
from ml_model_utils.files import is_s3, get_local_files, get_s3_files


@pytest.mark.parametrize("path, expected_result",
                         [("s3://dummy_bucket/dummy/path", True), ("/dummy/path", False)])
def test_is_s3(path, expected_result):
    result = is_s3(path)
    assert result == expected_result


@mock.patch("ml_model_utils.files.os")
def test_get_local_files(mocked_os):
    mocked_files = ["/app/tests/resources/dummy.csv",
                    "/app/tests/resources/dummy",
                    "/app/tests/resources/dummy1.parquet",
                    "/app/tests/resources/dummy2.parquet",
                    "/app/tests/resources/dummy3.parquet"]
    mocked_os.listdir = mock.MagicMock(return_value=mocked_files)
    mocked_os.path.isfile = mock.MagicMock(side_effect=lambda x: "." in x)
    mocked_os.path.join = os.path.join
    path = "/app/tests/resources"
    files = get_local_files(path)
    assert list(files) == ["/app/tests/resources/dummy1.parquet",
                           "/app/tests/resources/dummy2.parquet",
                           "/app/tests/resources/dummy3.parquet"]


@mock.patch("ml_model_utils.files.S3FileSystem")
def test_get_s3_files(mocked_s3filesystem):
    mocked_files = [dict(name="dummy_bucket/dummy/path/dummy.csv", type="file", size=10),
                    dict(name="dummy_bucket/dummy/path/dummy.parquet", type="directory", size=100),
                    dict(name="dummy_bucket/dummy/path/dummy1.parquet", type="file", size=0),
                    dict(name="dummy_bucket/dummy/path/dummy2.parquet", type="file", size=5),
                    dict(name="dummy_bucket/dummy/path/dummy3.parquet", type="file", size=15)]
    mocked_listdir = mock.MagicMock(return_value=mocked_files)
    mocked_s3filesystem.return_value = mock.MagicMock(listdir=mocked_listdir)
    path = "s3://dummy_bucket/dummy/path"
    files = get_s3_files(path)
    assert list(files) == ['s3://dummy_bucket/dummy/path/dummy2.parquet',
                           's3://dummy_bucket/dummy/path/dummy3.parquet']
