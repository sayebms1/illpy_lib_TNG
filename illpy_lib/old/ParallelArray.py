# ==================================================================================================
# ParallelArray.py
# ----------------
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================

import numpy as np


class ParallelArray(object):
    """

    """

    def __init__(self, length, names, types, keys=None, zero=True):
        self.num = len(names)                                                                       # Number of arrays

        # Make sure number of names matches number of types
        if (self.num != len(types)):
            raise RuntimeError("Num names doesn't match num types!")

        # Make sure keys (if provided) match number of names
        if (keys != None):
            if (len(keys) != self.num):
                raise RuntimeError("Num keys doesn't match num names or types!")

        # Choose how to initialize arrays
        if (zero): initFunc = np.zeros                                                             # Initialize arrays to zero  (clean)
        else:       initFunc = np.empty                                                             # Initialize arrays to empty (unclean)

        # Create dictionary to store keys for each array (for later access)
        self.__keys = {}
        #self.__names = {}

        # Initialize arrays
        for ii in range(self.num):
            # Add array as attribute
            setattr(self, names[ii], initFunc(length, dtype=types[ii]))
            # Establish an ordering for different arrays
            self.__keys[names[ii]] = ii

        self.__len = length                                                                            # Set length to initial value


    @classmethod
    def initFromArrays(cls, names, arrs):

        # Get number of arrays
        if (len(arrs) != len(names)): raise RuntimeError("Names must match arrays in number!")

        # Get length of each array
        length = len(arrs[0])

        print(("num = ",  length))

        # Get types from given arrays
        types = [type(ar[0]) for ar in arrs]

        print(("TYPES = ",  types))

        # Initialize object using default constructor
        pararr = cls(length, names, types, zero=False)

        print(pararr)
        print(pararr.num)
        print(getattr(pararr, names[0]))
        print(pararr[names[0]])


        # Set each array in object to given array
        for name, ar in zip(names, arrs):
            pararr[name] = ar

        return pararr



    def keys(self):
        ''' List of array 'keys' (names) for each array, and their index for each row '''
        return sorted(list(self.__keys.items()), key=lambda x:x[1])

    def __len__(self): return self.__len

    def __getitem__(self, key):
        """
        Return target arrays.

        For a particular merger, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. MERGER_TIME = 'time' for the time array
        """

        if (type(key) == int): key = np.int(key)                                                   # Make sure ints are numpy ints

        # If int, return slice across all arrays
        if (np.issubdtype(type(key), int)):
            return [getattr(self, name)[key] for name, ind in list(self.keys())]
        # If str, try to return that attribute
        elif (type(key) == str):
            if (hasattr(self, key)): return getattr(self, key)
            else: raise KeyError("Unrecognized key '%s' !" % (key))
        # Otherwise, error
        else:
            raise KeyError("Key must be a string or integer, not a %s!" % (str(type(key))))

    '''
    def __getattr__(self, key):
        if (hasattr(self, key)): return getattr(self, key)
        else: raise KeyError("Not attribute '%s'!" % (key))
    '''

    def __setitem__(self, key, vals):
        """
        Set target array.

        For a particular merger, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. MERGER_TIME = 'time' for the time array
        """

        # If int, set same element of each array
        if (isInt(key)):
            # name is the array name, ind is the corresponding index of 'vals'
            for (name, ind), aval in zip(list(self.keys()), vals):
                getattr(self, name)[key] = aval                                                      # set element 'key' of array 'name' to aval (at ind)

        # If str, set full array
        elif (type(key) == str):
            getattr(self, key)[:] = vals

        # Otherwise, error
        else:
            raise KeyError("Key must be a string or integer, not a %s!" % (str(type(key))))

        return


    def __delitem__(self, key):
        """ Delete target index of each array """
        return self.delete(key)


    def delete(self, keys):
        """ Delete target indices of each array """

        # In each array, delete target element
        for name, ind in list(self.keys()):
            setattr(self, name, np.delete(getattr(self, name), key))
            if (ind == 0): self.__len = len(getattr(self, name))

        return self.__len


    def append(self, vals):
        """ Append the given merger information as a new last element """

        # In each array, delete target element
        for (name, ind), aval in zip(list(self.keys()), vals):
            setattr(self, name, np.append(getattr(self, name), aval))
            if (ind == 0): self.__len = len(getattr(self, name))

        return self.__len



def isInt(val):
    # python int   ==> True
    if (type(val) == int): return True
    # numpy int    ==> True
    elif (np.issubdtype(type(val), int)): return True
    # anyting else ==> False
    else: return False
