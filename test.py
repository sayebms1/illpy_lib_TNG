"""
"""

import os

import numpy as np
from numpy.testing import run_module_suite
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

import illpy_lib
from illpy_lib import illbh


def test_core_paths():
    core = illbh.Core()
    paths = core.paths

    mergers_files = paths.mergers_input
    assert_true(len(mergers_files) > 0)
    assert_true(os.path.exists(mergers_files[0]))

    details_files = paths.details_input
    assert_true(len(details_files) > 0)
    assert_true(os.path.exists(details_files[0]))

    return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
