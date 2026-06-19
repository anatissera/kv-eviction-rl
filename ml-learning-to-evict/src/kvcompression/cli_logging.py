#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Simple CLI logging configuration that overrides Rich logging for cleaner output."""

import logging

from rich.console import Console
from rich.logging import RichHandler


class CLIRichHandler(RichHandler):
    """Rich handler optimized for CLI scripts - clean INFO logs, styled warnings/errors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = Console(stderr=False)  # Use stdout for all logs

    def emit(self, record):
        if record.levelno == logging.INFO:
            # For INFO logs, just print the message with rich formatting
            self.console.print(record.getMessage())
        else:
            # For other levels, use standard rich handler formatting
            super().emit(record)


def setup_cli_logging():
    """Configure clean CLI logging that overrides kvcompression's default Rich setup."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = CLIRichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_level=False,
        show_path=False,
    )

    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
