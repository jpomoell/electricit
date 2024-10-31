# This file is part of ELECTRICIT.
#
# Copyright 2024 ELECTRICIT developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Dataset class
"""

import os
import glob

import electricit.utils


class Dataset:

    def __init__(self):
        pass

    def create(self, path, title, datetime_format, filename_format, segments):

        # Get files from the given path    
        files = glob.glob(os.path.join(path, title + "*"))

        if len(files) == 0:
            raise ValueError("No files found with the given path and file pattern")

        # Set path to the data
        self.path = os.path.join(path, '')

        # Set title
        self.title = title
        
        # Extract dates from the found files
        dates = set()
        for file in files:
            dates.add(electricit.utils.find_and_parse_datetime(file, datetime_format))

        # Store datetimes
        self.times = sorted(dates)

        # Store requested segments
        self.segments = segments

        # Store the used date-time format
        self.datetime_fmt = datetime_format
        
        self.filename_fmt = filename_format
        
    def create_filestring(self, time, segment):
        """Creates filename string for a given time and segment in the dataset
        
        Filename string is the absolute path to the file, which is named using 
        convention specified by class variable "filename_fmt" and input 
        variables "time" (format: 'self.time_fmt') and "segment". 
        """

        # Date in the correct format        
        datetime_str = time.strftime(self.datetime_fmt)

                
        # Construct the file name
        file_name \
            = self.filename_fmt.format(title=self.title,
                                       date=datetime_str,
                                       time=datetime_str,
                                       datetime=datetime_str,
                                       segment=segment)

        return os.path.join(self.path, file_name)
