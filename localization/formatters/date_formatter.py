from datetime import datetime

def format_date(date: datetime, format_string: str) -> str:
    """
    Formats a given date into a specified string format.

    Args:
        date (datetime): The date to format.
        format_string (str): The format string to use for formatting.

    Returns:
        str: The formatted date as a string.
    """
    return date.strftime(format_string)