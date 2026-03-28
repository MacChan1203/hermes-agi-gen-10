from __future__ import annotations


def classify_error(stderr: str) -> str:
    s = (stderr or "").lower()
    if "command not found" in s:
        return "missing_command"
    if "permission denied" in s:
        return "permission_error"
    if "no module named" in s or "modulenotfounderror" in s:
        return "missing_python_module"
    if "connection refused" in s or "failed to establish a new connection" in s:
        return "connection_error"
    if "no such file or directory" in s or "file not found" in s:
        return "missing_file"
    if "syntaxerror" in s:
        return "syntax_error"
    return "unknown_error"


def should_retry_step(step: str, failed_steps: list[str], max_retries_for_same_step: int = 2) -> bool:
    return failed_steps.count(step) < max_retries_for_same_step


def should_retry_error_type(error_type: str, error_history: list[str], max_retries_for_same_error: int = 2) -> bool:
    return error_history.count(error_type) < max_retries_for_same_error
