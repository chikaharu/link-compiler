# AssertionError with a diagnostic payload
def check_format(offset, expected):
    assert offset == expected, f"format mismatch: offset {offset} != expected {expected}"
check_format(40, 32)
