# ==================================================================================================
# ObjDetails.py
# -------------
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================

import numpy as np
from glob import glob
from datetime import datetime

from constants import *


class Details(object):
    ''' Class to store contents of details files with components as simple arrays.

    Usage:
      details = Details(NUM_DETAILS)
      details[i] = [ID, TIME, MASS, MDOT, RHO, CS]

    This class is just a wrapper for 6 numpy arrays storing BH properties.
    Each constituent array can be accessed as if this object was a dict, e.g
      times = details[DETAIL_TIME]
    or the values for a single 'detail' can be accessed, e.g.
      detail100 = detailss[100]

    Details supports deletion by element, or series of elements,
    e.g.
      del details[100]
      details.delete([100, 200, 221])

    Individual details can be added using either the Details.add()
    method, or by accessing the last+1 memeber, e.g. if len(details) = N
      details[N+1] = [ID, TIME, ...]
    will also work.

    '''

    DETAIL_ID       = 0
    DETAIL_TIME     = 1
    DETAIL_MASS     = 2
    DETAIL_MDOT     = 3
    DETAIL_RHO      = 4
    DETAIL_CS       = 5

    def __init__(self, nums):
        ''' Initialize object with empty arrays for 'nums' entries '''
        self.id    = np.zeros(nums, dtype=LONG)
        self.time  = np.zeros(nums, dtype=DBL)
        self.mass  = np.zeros(nums, dtype=DBL)
        self.mdot  = np.zeros(nums, dtype=DBL)
        self.rho   = np.zeros(nums, dtype=DBL)
        self.cs    = np.zeros(nums, dtype=DBL)
        self.__len = nums


    @classmethod
    def keys(cls):
        return str([cls.DETAIL_ID, cls.DETAIL_TIME, cls.DETAIL_MASS,
                    cls.DETAIL_MDOT, cls.DETAIL_RHO, cls.DETAIL_CS])


    def __len__(self): return self.__len


    def __getitem__(self, key):
        '''
        Return target arrays.

        For a particular detail, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. DETAIL_TIME = 'time' for the time array
        '''

        # If normal int, Convert to numpy int
        if (type(key) == int): key = np.int(key)


        if (np.issubdtype(type(key), int)):
            return [self.id[key], self.time[key], self.mass[key],
                     self.mdot[key], self.rho[key], self.cs[key]]
        elif (key == Details.DETAIL_ID  ): return self.id
        elif (key == Details.DETAIL_TIME): return self.time
        elif (key == Details.DETAIL_MASS): return self.mass
        elif (key == Details.DETAIL_MDOT): return self.mdot
        elif (key == Details.DETAIL_RHO ): return self.rho
        elif (key == Details.DETAIL_CS  ): return self.cs
        else: raise KeyError("Unrecozgnized key '%s' type %s!" % (str(key), type(key)))


    def __setitem__(self, key, vals):
        '''
        Set target array.

        For a particular detail, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. DETAIL_TIME = 'time' for the time array
        '''
        if (  type(key) == int):
            if (key == self.__len): self.add(vals)
            else:
                self.id[key]   = vals[Details.DETAIL_ID]
                self.time[key] = vals[Details.DETAIL_TIME]
                self.mass[key] = vals[Details.DETAIL_MASS]
                self.mdot[key] = vals[Details.DETAIL_MDOT]
                self.rho[key]  = vals[Details.DETAIL_RHO]
                self.cs[key]   = vals[Details.DETAIL_CS]
        elif (key == DETAIL_ID  ):
            self.id = vals
        elif (key == DETAIL_TIME):
            self.time = vals
        elif (key == DETAIL_MASS):
            self.mass = vals
        elif (key == DETAIL_MDOT):
            self.mdot = vals
        elif (key == DETAIL_RHO ):
            self.rho = vals
        elif (key == DETAIL_CS  ):
            self.cs = vals
        else: raise KeyError("Unrecozgnized key '%s'!" % (str(key)))


    def __delitem__(self, key):
        ''' Delete the detail array at the target index '''

        self.id    = np.delete(self.id,   key)
        self.time  = np.delete(self.time, key)
        self.mass  = np.delete(self.mass, key)
        self.mdot  = np.delete(self.mdot, key)
        self.rho   = np.delete(self.rho,  key)
        self.cs    = np.delete(self.cs,   key)
        self.__len = len(self.time)


    def delete(self, keys):
        ''' Delete the detail(s) at 'keys' - an integer (list) '''

        self.id    = np.delete(self.id,   keys)
        self.time  = np.delete(self.time, keys)
        self.mass  = np.delete(self.mass, keys)
        self.mdot  = np.delete(self.mdot, keys)
        self.rho   = np.delete(self.rho,  keys)
        self.cs    = np.delete(self.cs,   keys)
        self.__len = len(self.id)


    def add(self, vals):
        ''' Append the given detail information as a new last element '''

        self.id    = np.append(self.id,   vals[Details.DETAIL_ID])
        self.time  = np.append(self.time, vals[Details.DETAIL_TIME])
        self.mass  = np.append(self.mass, vals[Details.DETAIL_MASS])
        self.mdot  = np.append(self.mdot, vals[Details.DETAIL_MDOT])
        self.rho   = np.append(self.rho,  vals[Details.DETAIL_RHO])
        self.cs    = np.append(self.cs,   vals[Details.DETAIL_CS])
        self.__len = len(self.id)
