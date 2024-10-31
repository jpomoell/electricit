# This file is part of ELECTRICIT.
#
# Copyright 2024 ELECTRICIT developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Utilities
"""

import re
import datetime

def find_and_parse_datetime(input_str, datetime_format):
    """Finds and parses a datetime string from input_str based on the given datetime_format.

    Args:
        input_str: The input string that may contain a datetime.
        datetime_format: The format of the datetime string (e.g., '%Y-%m-%d %H:%M:%S').

    Returns:
        A datetime object if a valid datetime string is found and parsed, otherwise None.
    """
    
    format_regex_mapping = {
        '%Y': r'\d{4}',  # Year (4 digit)
        '%m': r'\d{2}',  # Month (01-12)
        '%d': r'\d{2}',  # Day (01-31)
        '%H': r'\d{2}',  # Hour (00-23)
        '%M': r'\d{2}',  # Minute (00-59)
        '%S': r'\d{2}',  # Second (00-59)
    }

    # Escape other regex characters in the datetime format to avoid misinterpretation
    datetime_pattern = re.escape(datetime_format)
    
    # Replace datetime format codes with corresponding regex patterns
    for format_code, regex_pattern in format_regex_mapping.items():
        datetime_pattern = datetime_pattern.replace(re.escape(format_code), regex_pattern)

    # Search for the first occurrence of the pattern in the input string
    match = re.search(datetime_pattern, input_str)

    if match:
        
        # Extract the matched datetime string
        datetime_str = match.group(0)
        
        return datetime.datetime.strptime(datetime_str, datetime_format)

    else:
        raise ValueError("No datetime string found in the input")