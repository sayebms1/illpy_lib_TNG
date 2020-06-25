"""
"""
from datetime import datetime

import illpy_lib
import illpy_lib.illbh


def main():
    import illpy_lib
    import illpy_lib.illbh
    core = illpy_lib.illbh.Core(sets=dict(NAME='illbh', LOG_FILENAME='log_illbh.log'))
    log = core.log
    log.warning(__file__ + " " + str(datetime.now()))

    import illpy_lib.illbh.mergers
    log.warning("Loading temporary mergers")
    illpy_lib.illbh.mergers.load_temp_mergers(core=core)

    import illpy_lib.illbh.details
    log.warning("Reorganizing details")
    illpy_lib.illbh.details.reorganize(core=core)

    log.warning("Reformatting details")
    illpy_lib.illbh.details.reformat(core=core)


    import illpy_lib.illbh.matcher
    log.warning("Loading merger-details")
    mdets = illpy_lib.illbh.matcher.load_merger_details(core)

    log.warning("Loading fixed mergers")
    mrgs = illpy_lib.illbh.mergers.load_fixed_mergers(core=core)

    log.warning("Loading merger tree")
    tree = illpy_lib.illbh.mergers.load_tree(core=core, mrgs=mrgs)

    log.warning("Loading remmant-details")
    rdets = illpy_lib.illbh.matcher.load_remnant_details(core)

    return


if __name__ == "__main__":
    main()
