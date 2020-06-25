# ==================================================================================================
# ObjMergers.py
# ------------
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
import pickle as pickle
from constants import *

import AuxFuncs as aux


class Mergers(object):
    ''' Class to store many mergers with components as simple arrays.

    Usage:
      mergers = Mergers(NUM_MERGERS)
      mergers[i] = [TIME, OUT_ID, OUT_MASS, IN_ID, IN_MASS]

    This class is just a wrapper for 5 numpy arrays storing the time,
    IDs and masses of both blackholes ('in' the accreted, and 'out'
    the accretor).  Each constituent array can be accessed as if this
    object was a dict, e.g.
      times = mergers[MERGER_TIME]
    or the values for a single merger can be accessed, e.g.
      merger100 = mergers[100]

    Mergers supports deletion by element, or series of elements,
    e.g.
      del mergers[100]
      mergers.delete([100, 200, 221])

    Individual mergers can be added using either the Mergers.add()
    method, or by accessing the last+1 memeber, e.g. if len(mergers) = N
      mergers[N+1] = [TIME, IN_ID, ...]
    will also work.


    *** NOTE: MASS OF THE ACCRETOR IS *WRONG*, this is dynamical mass ***
    ***       not the BH mass... must cross-check with details files. ***


    '''

    # Keys for each merger entry
    MERGER_TIME     = 0
    MERGER_OUT_ID   = 1
    MERGER_OUT_MASS = 2
    MERGER_IN_ID    = 3
    MERGER_IN_MASS  = 4


    def __init__(self, nums):
        ''' Initialize object with empty arrays for 'num' entries '''
        self.time     = np.zeros(nums, dtype=DBL)
        self.out_id   = np.zeros(nums, dtype=LONG)
        self.out_mass = np.zeros(nums, dtype=DBL)
        self.in_id    = np.zeros(nums, dtype=LONG)
        self.in_mass  = np.zeros(nums, dtype=DBL)
        self.__len    = nums


    def __len__(self): return self.__len


    def __getitem__(self, key):
        '''
        Return target arrays.

        For a particular merger, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. MERGER_TIME = 'time' for the time array
        '''

        if (  type(key) == int):
            return [self.time[key], self.in_id[key], self.in_mass[key],
                     self.out_id[key], self.out_mass[key]]
        elif (key == Mergers.MERGER_TIME    ): return self.time
        elif (key == Mergers.MERGER_OUT_ID  ): return self.out_id
        elif (key == Mergers.MERGER_OUT_MASS): return self.out_mass
        elif (key == Mergers.MERGER_IN_ID   ): return self.in_id
        elif (key == Mergers.MERGER_IN_MASS ): return self.in_mass
        else: raise KeyError("Unrecozgnized key '%s'!" % (str(key)))


    def __setitem__(self, key, vals):
        '''
        Set target array.

        For a particular merger, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. MERGER_TIME = 'time' for the time array
        '''
        if (  type(key) == int):
            if (key == self.__len): self.add(vals)
            else:
                self.time[key]     = vals[Mergers.MERGER_TIME]
                self.out_id[key]   = vals[Mergers.MERGER_OUT_ID]
                self.out_mass[key] = vals[Mergers.MERGER_OUT_MASS]
                self.in_id[key]    = vals[Mergers.MERGER_IN_ID]
                self.in_mass[key]  = vals[Mergers.MERGER_IN_MASS]
        elif (key == Mergers.MERGER_TIME    ):
            self.time = vals
        elif (key == Mergers.MERGER_OUT_ID  ):
            self.out_id = vals
        elif (key == Mergers.MERGER_OUT_MASS):
            self.out_mass = vals
        elif (key == Mergers.MERGER_IN_ID   ):
            self.in_id = vals
        elif (key == Mergers.MERGER_IN_MASS ):
            self.in_mass = vals
        else: raise KeyError("Unrecozgnized key '%s'!" % (str(key)))


    def __delitem__(self, key):
        ''' Delete the merger array at the target index '''

        self.time     = np.delete(self.time,     key)
        self.out_id   = np.delete(self.out_id,   key)
        self.out_mass = np.delete(self.out_mass, key)
        self.in_id    = np.delete(self.in_id,    key)
        self.in_mass  = np.delete(self.in_mass,  key)
        self.__len    = len(self.time)


    def delete(self, keys):
        ''' Delete the merger(s) at 'keys' - an integer (list) '''

        self.time     = np.delete(self.time,     keys)
        self.out_id   = np.delete(self.out_id,   keys)
        self.out_mass = np.delete(self.out_mass, keys)
        self.in_id    = np.delete(self.in_id,    keys)
        self.in_mass  = np.delete(self.in_mass,  keys)
        self.__len    = len(self.time)


    def add(self, vals):
        ''' Append the given merger information as a new last element '''

        self.time     = np.append(self.time,     vals[0])
        self.out_id   = np.append(self.out_id,   vals[3])
        self.out_mass = np.append(self.out_mass, vals[4])
        self.in_id    = np.append(self.in_id,    vals[1])
        self.in_mass  = np.append(self.in_mass,  vals[2])
        self.__len    = len(self.time)





#  =============================================  #
#  ==========  STATIC MERGER METHODS  ==========  #
#  =============================================  #




def getIllustrisMergerFilenames(runNum, runsDir, log=None):
    '''Get a list of 'blackhole_mergers' files for a target Illustris simulation'''

    if (log): log.log("getIllustrisMergerFilenames()", 1)

    mergerNames      = np.copy(runsDir).tostring()
    if (not mergerNames.endswith('/')): mergerNames += '/'
    mergerNames += RUN_DIRS[runNum]
    if (not mergerNames.endswith('/')): mergerNames += '/'
    mergerNames += BH_MERGERS_FILENAMES

    if (log): log.log("Searching '%s'" % mergerNames, 2)
    files = sorted(glob(mergerNames))                                                               # Find and sort files
    if (log): log.log("Found %d files" % (len(files)), 3)

    return files


def loadAllIllustrisMergers(runNum, runsDir, log=None):

    if (log):
        log += 1
        log.log("loadAllIllustrisMergers()")

    # Load list of merger filenames for this simulation run (runNum)
    if (log): log += 1
    mergerFiles = getIllustrisMergerFilenames(runNum, runsDir, log=log)
    if (log):
        log -= 1
        log.log("Found %d illustris merger files" % (len(mergerFiles)), 1)

    # Load Merger Data from Illutris Files
    if (log): log.log("Parsing merger lines", 1)
    tmpList = [parseIllustrisMergerLine(mline) for mfile in mergerFiles for mline in open(mfile)]
    mnum = len(tmpList)

    # Fill merger object with Merger Data
    if (log): log.log("Creating mergers object", 1)
    mergers = Mergers(mnum)
    for ii, tmp in enumerate(tmpList):
        mergers[ii] = tmp

    return mergers



def loadMergers(runNum, runsDir, loadFile=None, saveFile=None, log=None):

    if (log):
        log += 1
        log.log("loadMergers()")
        log.log("Loading Mergers from run %d" % (runNum), 1)

    load = False
    save = False
    if (loadFile): load = True
    if (saveFile): save = True

    # Load an existing save file (NPZ)
    if (load):
        if (log): log.log("Trying to load mergers from '%s'" % (loadFile), 2)
        # Try to load save-file
        try: mergers = loadMergersFromSave(loadFile)
        # Fall back to loading mergers from merger-files
        except Exception as err:
            if (log): log.log("FAILED '%s'" % err.message, 3)
            load = False

    # Load Mergers from Illustris merger files
    if (not load or len(mergers) == 0):
        if (log):
            log.log("Loading mergers directly from Illustris Merger files", 2)
            log += 1

        mergers = loadAllIllustrisMergers(runNum, runsDir, log=log)

        if (log): log -= 1

    if (log):
        log.log("Loaded %d mergers." % (len(mergers)), 2)
        log += 1

    # Save Mergers to save-file if desired
    if (save and len(mergers) > 0): saveMergers(mergers, saveFile, log=log)
    if (log): log -= 2

    return mergers


def parseIllustrisMergerLine(instr):
    '''
    Parse a line from an Illustris blachole_mergers_#.txt file

    The line is formatted (in C) as:
        '%d %g %llu %g %llu %g\n',
        ThisTask, All.Time, (long long) id,  mass, (long long) P[no].ID, BPP(no).BH_Mass

    return time, accretor_id, accretor_mass, accreted_id, accreted_mass
    '''
    args = instr.split()
    return DBL(args[1]), LONG(args[2]), DBL(args[3]), LONG(args[4]), DBL(args[5])


def saveMergers(mergers, saveFilename, log=None):
    '''
    Save mergers object using pickle.

    Overwrites any existing file.  If directories along the path don't exist,
    they are created.
    '''

    if (log): log.log("saveMergers()", 1)

    # Make sure output directory exists
    saveDir, saveName = os.path.split(saveFilename)
    checkDir(saveDir)

    # Save binary pickle file
    if (log): log.log("Saving mergers to '%s'" % (saveFilename), 2)
    saveFile = open(saveFilename, 'wb')
    pickle.dump(mergers, saveFile)
    saveFile.close()
    if (log): log.log("Saved, size %s" % getFileSizeString(saveFilename), 2)
    return


def loadMergersFromSave(loadFilename):
    '''
    Load mergers object from file.
    '''
    loadFile = open(loadFilename, 'rb')
    mergers = pickle.load(loadFile)
    return mergers
