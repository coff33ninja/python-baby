import pytest
import time # pytest is now used for a marker
import logging
import sys
import os

from evaluate import SecureExecutor  # Now it should be found

# Get a logger for this test module
test_logger = logging.getLogger(__name__)

DEFAULT_TEST_TIMEOUT = 20  # Increased default timeout

@pytest.mark.smoke
def test_execute_simple_pass():
    test_logger.info("Starting test_execute_simple_pass")
    code = "x = 10\ny = 20\nresult = x + y"
    tests = "assert result == 30, f'Expected 30, got {result}'"
    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert (
        passed is True
    ), f"Test should pass. Log: {log}, Stdout: {stdout}, Stderr: {stderr}"
    assert "execution completed" in log.lower()  # Case-insensitive
    test_logger.info("Finished test_execute_simple_pass")


def test_execute_assertion_fail():
    code = "x = 10"
    tests = "assert x == 20, 'x should be 20'"  # Add assert message for clarity
    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert passed is False, f"Test should fail. Log: {log}"
    assert (
        "assertionerror" in log.lower() or "assertionerror" in stderr.lower()
    ), "Log or stderr should indicate AssertionError"
    assert (
        "x should be 20" in log or "x should be 20" in stderr
    ), "Assertion message not found"


def test_execute_restricted_import_os(caplog):
    code = "import os\nprint(os.getcwd())"  # Attempting to use os
    tests = "assert True"  # Test itself is trivial
    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)

    # caplog.set_level(logging.DEBUG) # If SecureExecutor or RestrictedPython logs verbosely

    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert passed is False, "Execution should fail due to restricted import."
    # RestrictedPython typically raises a NameError for disallowed imports at compile/exec time,
    # if 'os' is not handled by safe_globals or a custom importer.
    # The log in evaluate.py will capture this as "Execution error: SyntaxError: ..."
    # If the code were `print(os.getcwd())` without `import os`, it would be a NameError.
    assert (
        "syntaxerror" in log.lower()
        or "nameerror" in log.lower()
        or "syntaxerror" in stderr.lower()
        or "nameerror" in stderr.lower()
    ), f"Expected SyntaxError or NameError due to restricted import. Log: {log}, Stderr: {stderr}"


def test_execute_infinite_loop_timeout():
    code = "count = 0\nwhile True:\n  count += 1 # Keep it minimally busy"
    tests = "assert True"  # This test won't be reached
    executor = SecureExecutor(timeout_seconds=3)  # Increased short timeout

    start_time = time.time()
    passed, log, stdout, stderr = executor.execute(code, tests)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Execution duration: {duration:.4f}s")
    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert passed is False, "Execution should have timed out and thus failed."
    # Simpler assertion as the log is consistently "Execution timed out."
    assert "execution timed out" in log.lower(), "Log message should indicate timeout."
    # Check if duration is close to timeout, allowing for some overhead but not excessively long.
    # Allowing a range around the 1-second timeout to account for process start/stop overhead
    # For a 3s timeout, expect it to be slightly above 3s due to overhead.
    # Allow an upper bound, e.g., timeout + 1.0s for overhead.
    # Allow a lower bound, e.g., timeout - 0.5s for overhead.
    assert (
        2.8 <= duration < (3.0 + 1.0)
    ), f"Execution duration {duration:.4f}s was not within the expected range for a 3s timeout."

def test_execute_sandboxed_code_cannot_access_host_variables():
    """
    Tests that code executed by SecureExecutor cannot access variables
    from the host Python environment (e.g., the test function's scope).
    """
    host_variable = "sensitive_data_from_host_environment"

    # Attempt to print the host_variable from within the sandboxed code.
    # This should fail with a NameError.
    code = f"print({host_variable})" # Intentionally try to access undefined variable
    tests = "assert True"  # Test string is trivial for this scenario

    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert passed is False, "Execution should fail due to trying to access an undefined (host) variable."
    assert "nameerror" in log.lower() or "nameerror" in stderr.lower(), \
        "Log or stderr should indicate a NameError."
    assert host_variable.split('_')[0] in log.lower() or host_variable.split('_')[0] in stderr.lower(), \
        f"The error message should reference the variable '{host_variable.split('_')[0]}'."
    assert host_variable not in stdout, "The content of the host variable should not be in stdout."

def test_execute_print_capture():
    code = "print('Hello from sandbox')\nprint('Line 2')\nvar = 'done'"
    tests = "assert var == 'done'"  # Ensure main code runs
    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert passed is True, f"Test should pass. Log: {log}, Stderr: {stderr}"
    assert (
        "Hello from sandbox" in stdout
    ), f"Stdout did not contain expected print. Stdout: {stdout}"
    assert (
        "Line 2" in stdout
    ), f"Stdout did not contain expected print. Stdout: {stdout}"


def test_execute_syntax_error_in_generated_code():
    code = "x = 10\ny = 20 +\nresult = x + y"  # Syntax error
    tests = "assert True"
    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert passed is False
    assert "syntaxerror" in log.lower() or "syntaxerror" in stderr.lower()


def test_execute_runtime_error_in_generated_code():
    code = "x = 10\nresult = x / 0"  # Runtime error (ZeroDivisionError)
    tests = "assert True"
    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert passed is False
    assert "zerodivisionerror" in log.lower() or "zerodivisionerror" in stderr.lower()


def test_execute_empty_code_and_tests():
    code = ""
    tests = ""
    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert passed is True  # Empty code and tests should "pass" (no errors)
    assert "execution completed" in log.lower()  # Case-insensitive


def test_execute_code_modifying_restricted_globals_fails_safely():
    # This test's behavior depends heavily on the strictness of _write_ guard.
    # The current basic _write_ = lambda x: x guard might allow this.
    # A stricter guard like full_write_guard would make this fail.
    # For now, we test based on the current (more permissive) setup.
    code = "len = lambda x: 'modified_len'"  # Attempt to modify a builtin in restricted_globals
    tests = (
        "assert len([]) == 0"  # This would fail if 'len' was successfully overwritten
    )

    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    # The assignment `len = ...` in the `code` string creates a local variable `len`
    # within that execution's scope due to RestrictedPython's handling and the current
    # permissive `_write_` guard in evaluate.py (`_write_ = lambda x: x`).
    # This local `len` does not override the builtin `len` available to the
    # `tests_string` execution context, which still resolves to the `len` from
    # `safe_globals['__builtins__']`. Thus, the assertion uses the original `len`.
    assert (
        passed is True
    ), "Test should pass as the overwrite of 'len' is contained in its scope."
    assert "Execution completed" in log, "Log should indicate normal completion."
    assert (
        "modified_len" not in stdout
    ), "The modified 'len' should not have affected print output if it was printed."


def test_execute_malformed_test_string_syntax_error():
    code = "x = 10"
    tests = "assert x =="  # Syntax error in test string
    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    print(f"Log: {log}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")

    assert passed is False, "Execution should fail due to syntax error in test string."
    assert (
        "syntaxerror" in log.lower() or "syntaxerror" in stderr.lower()
    ), "Log or stderr should indicate SyntaxError."


def test_execute_very_large_code_string_basic_pass():
    # Simple test to ensure basic handling of large code string, not for performance.
    # RestrictedPython or the system might have its own limits.
    # Create a long series of simple assignments.
    num_assignments = (
        500  # Reduced from a very large number to keep test reasonably fast
    )
    code_lines = [f"var_{i} = {i}" for i in range(num_assignments)]
    code = "\n".join(code_lines) + f"\nfinal_result = var_{num_assignments-1}"
    tests = f"assert final_result == {num_assignments-1}"

    executor = SecureExecutor(
        timeout_seconds=DEFAULT_TEST_TIMEOUT + 5
    )  # Slightly longer timeout for larger code
    passed, log, stdout, stderr = executor.execute(code, tests)

    assert (
        passed is True
    ), f"Execution with large code string should pass. Log: {log}, Stderr: {stderr}"
    assert "execution completed" in log.lower()


# It might be useful to also add tests for _get_full_ MartyrReport if that was part of SecureExecutor,
# but currently, it's not directly exposed. The current tests focus on the execute method's behavior.

def test_long_output_capture(caplog):
    # Tests if large stdout/stderr are handled without crashing (though they might be truncated by other means)
    long_string = "a" * 2000
    code = f"print('{long_string}')"
    tests = "assert True"
    executor = SecureExecutor(timeout_seconds=DEFAULT_TEST_TIMEOUT)
    passed, log, stdout, stderr = executor.execute(code, tests)

    assert passed is True
    assert long_string in stdout
    assert len(stdout) >= 2000


# It might be useful to also add tests for _get_full_ MartyrReport if that was part of SecureExecutor,
# but currently, it's not directly exposed. The current tests focus on the execute method's behavior.
