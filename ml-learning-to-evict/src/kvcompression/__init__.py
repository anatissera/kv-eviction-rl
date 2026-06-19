#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging
import os
from pathlib import Path
from typing import Optional

import dotenv
import git
from omegaconf import OmegaConf
from rich.logging import RichHandler
from rich.traceback import install

install(show_locals=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


def load_envs(env_file: Optional[str] = None) -> None:
    """Load all the environment variables defined in the `env_file`.

    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    if env_file is None:
        env_file = dotenv.find_dotenv(usecwd=True)
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


load_envs()

try:
    PROJECT_ROOT = Path(
        git.Repo(Path.cwd(), search_parent_directories=True).working_dir
    )
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()

logger.debug(f"Inferred project root: {PROJECT_ROOT}")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

__all__ = ["PROJECT_ROOT"]


def format_resolver(value: any, format_spec: str) -> str:
    """
    Applies f-string-style formatting to a value.
    Example usage in YAML: ${fmt:12, 03d} -> "012"
    """
    return f"{value:{format_spec}}"


OmegaConf.register_new_resolver("fmt", format_resolver, replace=True)
