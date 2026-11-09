import numpy as np

import src


def test_package_imports():
    assert src.__version__


def test_numpy_works():
    assert np.zeros(3).sum() == 0


def test_environment_is_py310():
    import sys

    assert sys.version_info[:2] == (3, 10), "W1 约定 Python 3.10"