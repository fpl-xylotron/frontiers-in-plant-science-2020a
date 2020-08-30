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
from xylo_inf_ui import *

if __name__ == "__main__":
    # Show welcome screen
    proceed = run_welcome_screen("XyloInf")
    if not proceed:
        sys.exit()

    # read model spec
    p = Path(__file__).resolve().parent
    with open(p/"model_specification.txt", 'r') as fh:
        mp = fh.readlines()[0].strip()
        if mp == '*':
            mp = ''
        else:
            mp = str(p/"models"/mp)

    # run the app
    app = QApplication(sys.argv)
    scrn_geom = app.desktop().screenGeometry()
    scrn_size = (scrn_geom.height(), scrn_geom.width())
    ui = XyloInfUI(k=3, model_path=mp)
    sys.exit(app.exec_())
