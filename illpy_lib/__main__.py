"""
"""
import sys

from . illbh import details


def main():
    details.main()

    return


if __name__ == "__main__":
    print("illpy_lib.__main__")
    print("sys.argv: '{}'".format(sys.argv))
    main()
