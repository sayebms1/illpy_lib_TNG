"""
"""

import os
import sys
from datetime import datetime
import numpy as np
import traceback as tb


class Log(object):

    def __init__(self, filename='log.txt', verbose=True, num=0, clean=False, binary=False):
        self.filename = filename
        self.verbose = verbose
        self.num = num

        self.__log = None                                                                           # Default to no log-file

        # If non-zero filename is provided, open log file
        if (type(filename) == str):
            if (len(filename) > 0):
                self.__initFile(filename, clean, binary)



    def __initFile(self, fname, cln, bin):
        """ Initialize a log file """

        # Determine File Mode
        if (cln): mode  = 'w'                                                                      # Clear file when opening ('w'rite)
        else:      mode  = 'a'                                                                      # Append to file ('a'ppend)

        if (bin): mode += 'b'                                                                      # Open as binary

        # Open File
        self.__log = open(fname, mode)

        # Setup File
        if (not cln): self.__log.write("\n\n\n\n\n\n")                                             # Add some whitespace

        # Add the date and time
        border = "="*100
        timeStr = str(datetime.now())
        timeStr = timeStr.center(100, ' ')
        timeStr = border + ("\n%s\n" % (timeStr)) + border + "\n\n\n"
        self.__log.write(timeStr)
        self.log("Opened log file '%s'" % (fname))

        return


    def log(self, arg, add=0):
        """ Write a log entry to stdout, and to log file if it exists. """

        if (self.verbose == False and self.__log == None): return

        #usenum = self.num + add
        tbLen = len(tb.extract_stack()) - 1                                                         # Subtract one to discount this function
        #usenum = tbLen-self.num                                                                     # Get length of stack traceback
        usenum = tbLen-self.num+add                                                                 # Get length of stack traceback
        prep = " -"*usenum
        if (usenum > 0): prep += " "

        if (self.verbose): print(prep + arg)

        if (type(self.__log) == file):
            if (not self.__log.closed):
                self.__log.write(prep + arg + "\n")
                self.__log.flush()


    def mark(self, num=0):
        """ Add a decorated timestamp to the logfile """
        timeStr = str(datetime.now())
        if (num > 0): timeStr = "  " + timeStr + "  "
        timeStr = timeStr.center(2*num + len(timeStr), '=')
        self.__log.write(timeStr + "\n")


    def __del__(self):
        """ Make sure log file is closed """

        if (type(self.__log) == file):
            if (not self.__log.closed): self.__log.flush()
            self.__log.close()
