import os
import re
from s3fs import S3FileSystem  # type: ignore
from typing import Generator


S3_HEADER_REGEX = re.compile(r'^s3[a-z]?://')


def is_s3(path: str) -> bool:
    """Check whether the given path is a valid s3 path.

    Args:
        path (str): path to check.

    Returns:
        bool: True, if it's a s3 path.

    Examples:
        >>> path1 = "s3://dummy_bucket/dummy/path"
        >>> is_s3(path1)
        True
        >>> path2 = "/dummy/path"
        >>> is_s3(path2)
        False
    """
    return bool(path and S3_HEADER_REGEX.match(path.lower()))  # type: ignore


def get_local_files(path: str, file_format: str = "parquet") -> Generator:
    """Get files for a specific format from a folder path in a file system.

    Args:
        path (str): folder path in a file system.
        file_format (str): file format of the files to get. By default, is parquet.

    Returns:
        :obj:`typing.Generator`: files in the given folder.

    Examples:
        >>> path = "/app/tests/resources"
        >>> get_local_files(path)
        <generator object get_files_from_file_system at ...>
    """
    desired_extension = ".{}".format(file_format)
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue
        elif not file_name.endswith(desired_extension):
            continue
        yield file_path


def get_s3_files(path: str, file_format: str = "parquet") -> Generator:
    """Get csv files from a folder path in s3.

    Args:
        path (str): folder path in s3.
        file_format (str): file format of the files to get. By default, is parquet.

    Returns:
        :obj:`typing.Generator`: files in the given folder.

    Examples:
        >>> path = "s3://dummy_bucket/dummy/path"
        >>> get_s3_files(path)
        <generator object get_files_from_s3 at ...>
    """
    s3_fs = S3FileSystem()
    s3_header = S3_HEADER_REGEX.match(path).group(0)  # type: ignore
    desired_extension = ".{}".format(file_format)
    for object_summary in s3_fs.listdir(path):
        if not object_summary["name"].endswith(desired_extension):
            continue
        elif object_summary["type"] == 'directory':
            continue
        elif not object_summary["size"]:
            continue
        file_path = s3_header + object_summary["name"]
        yield file_path
