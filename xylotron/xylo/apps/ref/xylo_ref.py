#!/usr/bin/xylopython
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
add_paths = [str(Path(__file__).resolve().parent.parent.parent.parent/'source'),
             str(Path(__file__).resolve().parent.parent.parent/'source')]
for p in add_paths:
    if p not in sys.path:
        sys.path.append(p)
from xylo_welcome_screen import *
from xylo_ref_ui import *

if __name__ == "__main__":
    # Show welcome screen
    proceed = run_welcome_screen("XyloRef")
    # proceed = True
    if not proceed:
        sys.exit()

    # run the app
    app = QApplication(sys.argv)
    scrn_geom = app.desktop().screenGeometry()
    scrn_size = (scrn_geom.height(), scrn_geom.width())
    ui = XyloRefUI()
    sys.exit(app.exec_())
