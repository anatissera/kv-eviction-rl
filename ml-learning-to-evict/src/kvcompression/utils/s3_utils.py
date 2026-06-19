#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import fsspec
import torch.distributed as dist
from s3fs import S3FileSystem as FileSystemClass

pylogger = logging.getLogger(__name__)


def get_s3_filesystem() -> Dict[str, Any]:
    """Get S3 filesystem with generic configuration via environment variables.

    Environment Variables:
        AWS_PROFILE: AWS profile name (optional)
        AWS_ENDPOINT_URL: Custom S3 endpoint (optional, for S3-compatible storage)
        AWS_VERIFY_SSL: Whether to verify SSL certificates (default: True)

    Returns:
        An fsspec S3 filesystem instance configured from environment variables.
    """
    options = {}

    profile = os.environ.get("AWS_PROFILE")
    if profile:
        options["profile"] = profile

    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    verify_ssl = os.environ.get("AWS_VERIFY_SSL", "true").lower() != "false"

    if endpoint_url:
        options["client_kwargs"] = {
            "endpoint_url": endpoint_url,
            "verify": verify_ssl,
        }

    return fsspec.filesystem("s3", **options)


def upload_directory(
    local_directory: str,
    remote_destination_path: str,
    fs: Optional[FileSystemClass] = None,
) -> bool:
    """Uploads the contents of a local directory to a destination path using fsspec.

    This function recursively copies files from the local_directory to the
    destination_path using the provided fsspec filesystem instance (`fs`).

    - Preserves the directory structure relative to local_directory.
    - Uses `fs.put` with `recursive=True`, which typically overwrites files
      in the destination if they already exist.
    - Does NOT delete files in the destination that are not present locally.
    - Uses the provided `fs` instance or gets a default one via
      `get_s3_filesystem()`.

    Args:
        local_directory (str): The path to the local directory whose contents
                               to upload. Must be an existing directory.
        remote_destination_path (str): The target destination path (prefix/folder)
                                   where the contents should be placed (e.g.,
                                   'your-bucket-name/path/to/destination' for S3).
                                   The exact format depends on the `fs` implementation
                                   (e.g., s3fs expects this *without* 's3://').
        fs (Optional[FileSystemClass]): An optional, pre-configured fsspec-compatible
                                        filesystem instance. If None,
                                        `get_s3_filesystem()` will be called
                                        to get a default instance (assumed to be S3-like).

    Returns:
        bool: True if the upload operation completed without raising an
              exception during the `fs.put` call. False if the local
              directory doesn't exist or an error occurred during upload.
    """
    local_directory = str(local_directory)
    remote_destination_path = str(remote_destination_path)

    if not os.path.isdir(local_directory):
        pylogger.warning(
            f"Local directory not found or is not a directory: '{local_directory}'. Skipping upload."
        )
        return False

    # Ensure local path ends with '/' for recursive put to copy contents *into* destination
    local_source = local_directory.rstrip("/") + "/"
    log_destination_path = remote_destination_path.rstrip("/") + "/"

    fs = fs if fs is not None else get_s3_filesystem()

    pylogger.info(
        f"Starting recursive put from '{local_source}' to '{log_destination_path}'..."
    )
    start_time = time.time()

    try:
        fs.put(local_source, remote_destination_path, recursive=True)

        end_time = time.time()
        duration = end_time - start_time
        pylogger.info(f"Recursive put finished in {duration:.2f} seconds.")
        return True

    except Exception as e:
        pylogger.error(
            f"Failed recursive put from '{local_source}' to '{log_destination_path}': {e}",
            exc_info=True,
        )
        return False


def download_file(
    remote_source_path: str,
    local_destination: str,
    fs: Optional[FileSystemClass] = None,
) -> bool:
    """Downloads a file from a source path to a local filepath using fsspec.

    Args:
        remote_source_path (str): The source path file to download
                              from (e.g., 'your-bucket-name/path/to/source.txt' for S3).
        local_destination (str): The path to the local file
        fs (Optional[FileSystemClass]): An optional, pre-configured fsspec-compatible
                                        filesystem instance. If None,
                                        `get_s3_filesystem()` will be called
                                        to get a default instance (assumed to be S3-like).

    Returns:
        bool: True if the download operation completed without raising an
              exception during the `fs.get` call. False if an error occurred
              during download (e.g., source path not found, permissions issue).
    """
    remote_source_path = str(remote_source_path)
    local_destination = str(local_destination)

    fs = fs if fs is not None else get_s3_filesystem()

    source_path_for_get = remote_source_path.rstrip("/")
    local_target = local_destination.rstrip("/")

    pylogger.info(
        f"Starting file download from '{source_path_for_get}' to '{local_target}'..."
    )

    start_time = time.time()

    try:
        fs.get(source_path_for_get, local_target, recursive=False)
        end_time = time.time()
        duration = end_time - start_time
        pylogger.info(f"File download finished in {duration:.2f} seconds.")
        return True

    except Exception as e:
        pylogger.error(
            f"Failed file download from '{source_path_for_get}' to '{local_target}': {e}",
            exc_info=True,
        )
        return False


def get_corresponding_remote_path(
    path: Union[Path, str], local_root: Union[Path, str], remote_root: str
) -> Path:
    """
    Given a local path and the corresponding local and remote root directories,
    this function returns the corresponding remote path."""
    relative_run_directory = Path(path).relative_to(Path(local_root))
    remote_run_directory = (
        str(remote_root).rstrip("/") + "/" + str(relative_run_directory)
    )
    return remote_run_directory


def load_remote_json_secrets(
    remote_secret_file: str,
    local_secret_file: str = ".secrets.json",
    rank: int = 0,
) -> None:
    """Downloads a remote JSON secret file and loads environment variables.

    This function first downloads a JSON file from a specified remote location
    to a local path. It then parses this local JSON file. If the JSON contains
    a top-level key named "environment" whose value is a dictionary, the
    key-value pairs from this dictionary are loaded into the current process's
    environment variables (`os.environ`).

    Important: Environment variables are only added if they do not already
    exist in `os.environ`. Existing variables will not be overwritten.

    Expected JSON structure in the remote file:
    ```json
    {
        "environment": {
            "SECRET_KEY": "your_secret_value",
            "DATABASE_URL": "your_db_connection_string",
            "ANOTHER_VAR": "some_other_value"
        },
        "other_config": {
            "key": "value"
        }
        // Other top-level keys are allowed but ignored by this function.
    }
    ```

    Args:
        remote_secret_file: The path, URL, or identifier for the remote JSON
            secret file to be downloaded. Passed directly to the download
            function.
        local_secret_file: The local file path where the remote secret file
            will be saved. Defaults to ".secrets.json" in the current working
            directory.
        rank: the rank of the process calling this function, in a distributed setting.

    Raises:
        FileNotFoundError: If the `local_secret_file` cannot be opened after
            the download attempt (e.g., due to permissions issues or if the
            download failed silently).
        json.JSONDecodeError: If the content of the downloaded file is not
            valid JSON.
        TypeError: If the value associated with the "environment" key in the
            JSON is not a dictionary.
    """

    if rank == 0:
        pylogger.info(
            f"Attempting to download secrets from '{remote_secret_file}' to '{local_secret_file}'..."
        )
        try:
            download_file(
                remote_source_path=remote_secret_file,
                local_destination=local_secret_file,
            )
            pylogger.info("Download successful.")
        except Exception as e:
            pylogger.error(
                f"Failed to download secrets from '{remote_secret_file}': {e}",
                exc_info=True,
            )
            raise

    if dist.is_initialized():
        dist.barrier()

    pylogger.info(f"Loading secrets from local file: '{local_secret_file}'")
    try:
        with Path(local_secret_file).open() as fp:
            secrets = json.load(fp)
    except FileNotFoundError:
        pylogger.error(
            f"Error: Local secret file not found at {local_secret_file} after download attempt."
        )
        raise
    except json.JSONDecodeError:
        pylogger.error(f"Error: Failed to decode JSON from {local_secret_file}.")
        raise

    environment_secrets = secrets.get("environment", {})

    if not isinstance(environment_secrets, dict):
        raise TypeError(
            f"Expected 'environment' key in {local_secret_file} to contain a dictionary (object), "
            f"but found type {type(environment_secrets).__name__}."
        )

    variables_set = []
    variables_skipped = []
    for key, value in environment_secrets.items():
        if key not in os.environ:
            os.environ[key] = str(value)
            variables_set.append(key)
        else:
            variables_skipped.append(key)

    if variables_set:
        variables_set.sort()
        pylogger.info(f"Set environment secrets from remote: {variables_set}")
    else:
        pylogger.info("No new environment variables were set.")

    if variables_skipped:
        variables_skipped.sort()
        pylogger.info(
            f"Skipped environment secrets from remote (already present): {variables_skipped}"
        )
    else:
        pylogger.info("No environment variables were skipped.")


def ls_remote_folder(
    remote_source_path: str,
    fs: Optional[FileSystemClass] = None,
) -> Sequence[str]:
    remote_source_path_str = str(remote_source_path)

    current_fs = fs if fs is not None else get_s3_filesystem()
    fs_sep = current_fs.sep

    source_path_for_get = remote_source_path_str.rstrip(fs_sep) + fs_sep
    all_remote_items = current_fs.find(source_path_for_get, detail=True, withdirs=False)
    return sorted(all_remote_items)
