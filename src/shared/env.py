"""Environment utilities for resolving secret files."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_secret_file_variables() -> None:
    """
    Resolve environment variables that follow Docker secret conventions.

    For every KEY_FILE entry, read the referenced file and expose its
    contents via KEY. Errors are logged but do not raise exceptions.
    """

    for key, file_path in list(os.environ.items()):
        if not key.endswith("_FILE"):
            continue
        target_key = key[:-5]
        if os.environ.get(target_key):
            continue
        if not file_path:
            continue
        try:
            value = Path(file_path).read_text(encoding="utf-8").strip()
            os.environ[target_key] = value
        except FileNotFoundError as exc:
            logger.warning(
                "env.secret_file.missing",
                extra={"key": key, "path": file_path, "error": str(exc)},
            )
        except UnicodeDecodeError as exc:
            logger.warning(
                "env.secret_file.decode_failed",
                extra={"key": key, "path": file_path, "error": str(exc)},
            )
        except OSError as exc:
            logger.warning(
                "env.secret_file.load_failed",
                extra={"key": key, "path": file_path, "error": str(exc)},
            )


# Ensure the util can be imported without manual invocation.
load_secret_file_variables()
