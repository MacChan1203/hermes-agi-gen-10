from __future__ import annotations

from enum import Enum

from .config import STUCK_FAILURE_THRESHOLD, REPEATED_ERROR_THRESHOLD


class ErrorType(str, Enum):
    """エラー分類の列挙型。"""
    PERMISSION = "permission_error"
    TIMEOUT = "timeout_error"
    MISSING_MODULE = "missing_python_module"
    SYNTAX = "syntax_error"
    RUNTIME = "runtime_error"
    NETWORK = "connection_error"
    NOT_FOUND = "missing_file"
    MISSING_COMMAND = "missing_command"
    UNKNOWN = "unknown_error"


def classify_error(stderr: str) -> str:
    """標準エラー出力からエラータイプを分類する。

    Returns:
        ErrorType の値（文字列）
    """
    s = (stderr or "").lower()
    if "command not found" in s:
        return ErrorType.MISSING_COMMAND.value
    if "permission denied" in s:
        return ErrorType.PERMISSION.value
    if "no module named" in s or "modulenotfounderror" in s:
        return ErrorType.MISSING_MODULE.value
    if "connection refused" in s or "failed to establish a new connection" in s:
        return ErrorType.NETWORK.value
    if "no such file or directory" in s or "file not found" in s:
        return ErrorType.NOT_FOUND.value
    if "syntaxerror" in s:
        return ErrorType.SYNTAX.value
    if "timed out" in s or "timeout" in s:
        return ErrorType.TIMEOUT.value
    if "runtimeerror" in s:
        return ErrorType.RUNTIME.value
    return ErrorType.UNKNOWN.value


def should_retry_step(step: str, failed_steps: list[str], max_retries_for_same_step: int = REPEATED_ERROR_THRESHOLD) -> bool:
    return failed_steps.count(step) < max_retries_for_same_step


def should_retry_error_type(error_type: str, error_history: list[str], max_retries_for_same_error: int = REPEATED_ERROR_THRESHOLD) -> bool:
    return error_history.count(error_type) < max_retries_for_same_error
