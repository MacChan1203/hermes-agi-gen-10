from hermes_agi_gen.errors import classify_error


def test_classify_missing_command():
    assert classify_error("bash: uv: command not found") == "missing_command"


def test_classify_permission_error():
    assert classify_error("Permission denied") == "permission_error"


def test_classify_missing_python_module():
    assert classify_error("ModuleNotFoundError: No module named 'requests'") == "missing_python_module"


def test_classify_connection_error():
    assert classify_error("Connection refused") == "connection_error"
