"""
"""

import os
from .. import particle_hosts
from nose.tools import assert_equal, assert_raises, assert_true, assert_false

from zcode import inout as zio


def test_filenames():
    print("particle_hosts_test.filenames_test()")
    for run in range(1, 4):
        print("run = '{}'".format(run))
        processed_dir = particle_hosts._get_path_processed(run)
        print("processed_dir = '{}'".format(processed_dir))
        zio.check_path(processed_dir)
        assert_true(os.path.exists(processed_dir))
        # make sure directory is writeable
        test_fname = os.path.join(processed_dir, 'test_123_test.txt')
        print("Trying to write to file '{}'".format(test_fname))
        with open(test_fname, 'w') as test:
            test.write('hello')
        print("File '{}' Exists: {}".format(test_fname, str(os.path.exists(test_fname))))
        assert_true(os.path.exists(test_fname))
        print("Deleting file '{}'".format(test_fname))
        os.remove(test_fname)
        print("File '{}' Exists: {}".format(test_fname, str(os.path.exists(test_fname))))
        assert_false(os.path.exists(test_fname))

        # Offset table
        print("loading path for: offset table")
        path = particle_hosts._get_filename_offset_table(run, 135, version='1.0')
        print(path)
        # bh-hosts-snap table
        print("loading path for: bh-hosts-snap table")
        path = particle_hosts._get_filename_bh_hosts_snap_table(run, 135, version='1.0')
        print(path)
        # bh-hosts table
        print("loading path for: bh-hosts table")
        path = particle_hosts._get_filename_bh_hosts_table(run, 135, version='1.0')
        print(path)

    print("using 'run' 0 should fail")
    assert_raises(KeyError, particle_hosts._get_path_processed, 0)

    return
