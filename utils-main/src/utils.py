def valid_positive_int(value: object, name: str,
                       maximum: int | None = None) -> bool:
    """Check that a CLI value is a positive int (optionally bounded).

    Args:
        value: The value received from the command line.
        name: Argument name, used in the error message.
        maximum: Optional upper bound (inclusive).

    Returns:
        True if valid, False otherwise (an error is printed).
    """

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        print(f"Error: --{name} must be a positive integer, got {value!r}")
        return False
    if maximum is not None and value > maximum:
        print(f"Error: --{name} must be <= {maximum}, got {value}")
        return False
    return True


def valid_query(query: object) -> bool:
    """Check that a CLI query is a non-empty string.

    Args:
        query: The query received from the command line.

    Returns:
        True if valid, False otherwise (an error is printed).
    """

    if not isinstance(query, str) or not query.strip():
        print("Error: --query must be a non-empty string")
        return False
    return True
