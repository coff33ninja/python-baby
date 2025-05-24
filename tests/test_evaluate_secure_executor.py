import pytest
import time # pytest is now used for a marker
import logging
import sys
import os

# Adjust sys.path to allow evaluate module to be found if tests are run from root
# This assumes tests/ is a subdirectory of the project root where evaluate.py is.
# If evaluate.py is part of an installable package, this might not be needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    assert "Execution completed" in log
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
        "AssertionError" in log or "AssertionError" in stderr
    )  # Error might be in log or stderr
    assert "x should be 20" in log or "x should be 20" in stderr


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
    # or if a guard prevents access. The exact message can vary.
    # Looking for common indicators of such failures.
    assert any(
        err_indicator in log.lower() or err_indicator in stderr.lower()
        for err_indicator in [
            "importerror",
            "nameerror",
            "syntaxerror",
            "restricted",
            "permission denied",
            "is not defined",
        ]
    ), f"Expected an import/security related error. Log: {log}, Stderr: {stderr}"


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
    assert (
        2.5 <= duration < (3.0 + 2.0)  # Adjusted for 3s timeout, allow up to 2s overhead
    ), f"Execution duration {duration} was not close to the timeout of 3s."


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
    assert "Execution completed" in log


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

    # Depending on RestrictedPython's behavior with the current guards for reassigning globals:
    # 1. If it fails (e.g. a stricter _write_ guard or if RestrictedPython itself prevents it):
    #    assert passed is False
    #    assert "error" in log.lower() or "error" in stderr.lower()
    # 2. If it passes (meaning the re-assignment was contained or didn't affect the test's 'len'):
    #    assert passed is True
    #    assert "Execution completed" in log
    # Given the current basic _write_ guard and how RestrictedPython handles scope,
    # the 'len' in `assert len([]) == 0` within the test string will still refer to the
    # original builtin 'len' provided in restricted_globals['__builtins__'], not the one
    # modified in the 'code' string's scope.
    assert (
        passed is True
    ), "Test should pass as the overwrite of 'len' is contained in its scope."
    assert "Execution completed" in log, "Log should indicate normal completion."
    assert (
        "modified_len" not in stdout
    ), "The modified 'len' should not have affected print output if it was printed."


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
