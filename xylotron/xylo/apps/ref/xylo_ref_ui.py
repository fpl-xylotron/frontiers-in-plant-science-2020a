import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
from datetime import datetime
from PyQt5.QtCore import (Qt, QObject, QThread, QRunnable, QThreadPool,
                          pyqtSlot, pyqtSignal, QCoreApplication, QSize)
from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, QCheckBox,
                             QMessageBox, QApplication, QDialog, QFileDialog,
                             QDialogButtonBox, QLabel, QFrame, QAction,
                             QVBoxLayout, QShortcut, QTextEdit, QDesktopWidget,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QHBoxLayout, QVBoxLayout, QFormLayout,
                             QGridLayout, QInputDialog, QLineEdit)
from PyQt5.QtGui import (QImage, QPixmap, QPalette, QBrush,
                         QColor, QFont, QKeySequence)
import json
# from PIL import Image, ImageDraw, ImageQt
import cv2
add_paths = [str(Path(__file__).resolve().parent.parent.parent.parent/'source'),
             str(Path(__file__).resolve().parent.parent.parent/'source')]
for p in add_paths:
    if p not in sys.path:
        sys.path.append(p)
from opencv_events_handler import *
from xylo_ref_screen import *
from xylo_scope import *
from xylo_logger import *

class SpecimenDialog(QDialog):
    def __init__(self, data, df):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.data = data
        self.df = df
        self.maxid = 9

        self.idlabel = QLabel('Specimen &ID: ')
        self.idtext = QLineEdit('')
        self.idlabel.setBuddy(self.idlabel)

        self.genlabel = QLabel('&Genus: ')
        self.gentext = QLineEdit('')
        self.genlabel.setBuddy(self.genlabel)

        self.spclabel = QLabel('&Species: ')
        self.spctext = QLineEdit('')
        self.spclabel.setBuddy(self.spclabel)

        self.query = QPushButton("&Query")
        self.query.clicked.connect(self._query_cb)
        self.add = QPushButton("&Add")
        self.add.clicked.connect(self._add_cb)

        grid = QGridLayout()
        grid.addWidget(self.idlabel, 0, 0)
        grid.addWidget(self.idtext, 0, 1)
        grid.addWidget(self.genlabel, 1, 0)
        grid.addWidget(self.gentext, 1, 1)
        grid.addWidget(self.spclabel, 2, 0)
        grid.addWidget(self.spctext, 2, 1)
        grid.addWidget(self.query, 3, 0)
        grid.addWidget(self.add, 3, 1)
        self.setLayout(grid)

        self.exec_()

    def _add_cb(self):
        spid = self.idtext.text().strip()
        # print(f'add - spid input: {spid}')
        if len(spid) == 0:
            return
#        spid = self._sanitize_spid(spid)
        # print(f'add - spid sanit: {spid}')

        zf = self.df["SpecimenID"].map(str).map(lambda s: s.zfill(self.maxid))
        row = self.df.loc[zf == spid]
        if row.shape[0] > 0:
            self.gentext.setText("")
            self.spctext.setText("")
            return

        taxapcs = [self.gentext.text().strip(), self.spctext.text().strip()]
        if not all([len(pc) for pc in taxapcs]):
            return

#        self.data["spid"] = self._sanitize_spid(spid)
        self.data["spid"] = spid
        self.data["genus"] = self._sanitize_genus(self.gentext.text())
        self.data["species"] = self._sanitize_species(self.spctext.text())
        self.data["update"] = True
        self.close()

    def _query_cb(self):
        spid = self.idtext.text().strip()
        # print(f'query - spid input: {spid}')

        if len(spid) == 0:
            return
        #spid = self._sanitize_spid(spid)
        # print(f'query - spid sanit: {spid}')

        # print(f'df.shape: {self.df.shape}')

        # zf = self.df["SpecimenID"].map(str).map(lambda s: s.zfill(self.maxid))
        # zf = self.df["SpecimenID"].map(str)
        zf = self.df["SpecimenID"]
        # print(zf)
        row = self.df.loc[zf == spid]
        if row.shape[0] == 0:
            return

#        self.data["spid"] = self._sanitize_spid(row.iloc[0]['SpecimenID'])
        self.data["spid"] = row.iloc[0]['SpecimenID']
        self.data["genus"] = self._sanitize_genus(row.iloc[0]['Genus'])
        self.data["species"] = self._sanitize_species(row.iloc[0]['Species'])
        self.data["update"] = False
        self.close()

    def _sanitize_spid(self, spid: str) -> str:
        sspid = ""
        for c in str(spid).strip():
            if c.isalnum():
                sspid += c
        return sspid.zfill(self.maxid)

    def _sanitize_genus(self, genus: str) -> str:
        sgenus = ""
        for c in genus.strip():
            if c.isalnum() and c != "_":
                sgenus += c
            else:
                sgenus += "-"
        tkns = [t.lower() for t in sgenus.split("-") if t.strip()]
        tkns[0] = tkns[0].title()
        sgenus = "-".join(tkns)
        return sgenus

    def _sanitize_species(self, species: str) -> str:
        sspecies = ""
        for c in species.strip():
            if c.isalnum() and c != "_":
                sspecies += c
            else:
                sspecies += "-"
        tkns = [t.lower() for t in sspecies.split("-") if t.strip()]
        sspecies = "-".join(tkns)
        return sspecies

class XyloRefUI(QMainWindow):
    def __init__(self, screen_size=(1080, 1920)) -> None:
        super(XyloRefUI, self).__init__()
        self.scrnsize = screen_size
        self.xylodir = Path(__file__).resolve().parent.parent.parent
        self.appsdir = Path(__file__).resolve().parent.parent
        self.refdir = Path(__file__).resolve().parent
        self.dbsdir = self.refdir/"dbs"
        self.imgsdir = None
        self.dbfile = None
        self.df = None

        cfg_dir = self.xylodir/"camcfgs"
        self.xs = XyloScope(config_dir=cfg_dir)
        xstatus = self.xs.start_camera()
        if not xstatus:
            sys.exit()
        self.xs.configure_camera()

        self._init_ui()

    def _imagesbutton_cb(self) -> None:
        self.imgsdir = None
        self.imagespath.setText('')
        self._flatten()

        data_dir = Path.home()/"xylo_ref_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.imgsdir = QFileDialog.getExistingDirectory(self,
                                                        'Choose images directory',
                                                        str(data_dir.parent))
        self.imgsdir = self.imgsdir.strip()
        if len(self.imgsdir) == 0:
            self.imgsdir = None
            return

        self.imgsdir = Path(self.imgsdir)
        self.imagespath.setText(self.imgsdir.name)
        self._flatten()

    def _dbbutton_cb(self) -> None:
        self.dbfile = None
        self.df = None
        self.dbpath.setText('')
        self.colltext.setText('')
        self._flatten()

        self.dbfile = QFileDialog.getOpenFileName(self,
                                                  'Choose database file',
                                                  str(self.dbsdir),
                                                  '*.csv')

        self.dbfile = self.dbfile[0].strip()
        if len(self.dbfile) == 0:
            self.dbfile = None
            return

        self.dbfile = Path(self.dbfile)
        self.df = pd.read_csv(self.dbfile, dtype={'SpecimenID': 'str'})
        reqcols = ["SpecimenID", "Genus", "Species"]
        status = all([c in self.df.columns for c in reqcols])
        if not status:
            return
        self.df = self.df[reqcols]

        self.dbpath.setText(self.dbfile.name)
        coll = self.dbfile.name.split('.')[0]
        self.colltext.setText(self._sanitize_collection(coll))
        self._flatten()

    def _spcmn_cb(self):
        if self.spcmn.isFlat():
            return
        d = {}
        self._set_specimen_text(d)
        dlg = SpecimenDialog(data=d, df=self.df)
        self._set_specimen_text(d)
        self._flatten()

        if "update" in d and d["update"]:
            # print("we will update df")
            dd = {"SpecimenID": [d['spid']],
                  "Genus": [d['genus']],
                  "Species": [d['species']]}
            ddf = pd.DataFrame(dd)
            self.df = self.df.append(ddf, ignore_index=True)
            print(self.df)
            print(self.dbfile)
            self.df.to_csv(self.dbfile, index=False)

    def _set_specimen_text(self, d: dict) -> None:
        if d:
            self.idtext.setText(d["spid"])
            self.genustext.setText(d["genus"])
            self.speciestext.setText(d["species"])
        else:
            self.idtext.setText("")
            self.genustext.setText("")
            self.speciestext.setText("")

    def _beginbutton_cb(self):
        if self.beginbutton.isFlat():
            return
        self._run_screen()
        self._set_specimen_text({})
        self._flatten()

    def _get_refwin_size(self):
        hgt = self.scrnsize[0] - 100
        wdt = self.scrnsize[1] - 100
        while (wdt%10) > 0:
            wdt -= 1
        quanta = wdt//10
        hgt = quanta*4
        return (wdt, hgt)

    def _get_user(self):
        with open(self.refdir/"user_specification.txt", "r") as fh:
            line = fh.readlines()[0].strip()
        if line[0] == '*':
            return None
        return line

    def _run_screen(self):
        self.setHidden(True)
        logger = XyloLogger(top_dir=self.imgsdir,
                            camera_serial_number=self.xs.serial_number(),
                            user_name=self._get_user())
        cfgstrmaker = ConfigStringMaker(xs=self.xs)
        handler = OpencvEventsHandler()
        screen = XyloRefScreen()
        screen.live_image()
        screen.prev_image()
        winname = "XyloRef"
        (winwdt, winhgt) = self._get_refwin_size()
        cv.namedWindow(winname, cv.WINDOW_NORMAL)
        cv.resizeWindow(winname, winwdt, winhgt)
        cv.moveWindow(winname, 50, 50)
        cv.setMouseCallback(winname, mouse_cb, handler)
        cv.imshow(winname, screen.image())
        handler.reset_keystroke()
        handler.reset_mouseevent()
        prev = None
        if not self.xs.is_capturing():
            self.xs.start_capturing()
        while True:
            handler.log_keystroke(cv.waitKey(30))
            screen.process(keystroke=handler.get_keystroke(),
                           mousexy=handler.get_mousexy())
            screen.live_image(self.xs.get_image()[:, :, ::-1])
            if screen.is_snapsave():
                prev = self.xs.get_image()[:, :, ::-1]
                logger.save(img=prev,
                            log=cfgstrmaker.string(),
                            genus=self.genustext.text(),
                            species=self.speciestext.text(),
                            collection=self.colltext.text(),
                            specimen=self.idtext.text())
                screen.prev_image(prev)
            if screen.is_change():
                break
            cv.imshow(winname, screen.image())
            screen.reset_state()
            handler.reset_keystroke()
            handler.reset_mouseevent()
        if self.xs.is_capturing():
            self.xs.stop_capturing()
        cv.destroyAllWindows()
        self.setHidden(False)

    def _exitbutton_cb(self):
        if self.xs.is_capturing():
            self.xs.stop_capturing()
        self.xs.cleanup()
        del self.xs
        cv.destroyAllWindows()
        sys.exit()

    def _flatten(self):
        if self.imgsdir and self.dbfile:
            self.spcmn.setFlat(False)
        if not self.imgsdir or not self.dbfile:
            self.spcmn.setFlat(True)

        if len(self.genustext.text()) > 0:
            self.beginbutton.setFlat(False)
        else:
            self.beginbutton.setFlat(True)

    def _sanitize_collection(self, coll: str) -> str:
        scoll = ""
        for c in coll:
            if c.isalnum():
                scoll += c
        return scoll

    def _init_ui(self):
        fixedwdt = 150
        fixedhgt = 25
        buff1 = 10

        self.widlayout = QFormLayout()
        nrows = 0

        self.imagesbutton = QPushButton("&Images Directory")
        self.imagesbutton.setShortcut(QKeySequence('I'))
        self.imagesbutton.setFixedWidth(fixedwdt)
        self.imagesbutton.setFixedHeight(fixedhgt)
        self.imagesbutton.clicked.connect(self._imagesbutton_cb)
        self.imagespath = QLabel()
        self.imagespath.setFrameStyle(QFrame.Box)
        self.imagespath.setFixedWidth(fixedwdt)
        self.imagespath.setFixedHeight(fixedhgt)
        self.widlayout.addRow(self.imagesbutton, self.imagespath)
        nrows += 1

        self.dbbutton = QPushButton("&Database (*.csv)")
        self.dbbutton.setShortcut(QKeySequence('D'))
        self.dbbutton.setFixedWidth(fixedwdt)
        self.dbbutton.setFixedHeight(fixedhgt)
        self.dbbutton.clicked.connect(self._dbbutton_cb)
        self.dbpath = QLabel()
        self.dbpath.setFrameStyle(QFrame.Box)
        self.dbpath.setFixedWidth(fixedwdt)
        self.dbpath.setFixedHeight(fixedhgt)
        self.widlayout.addRow(self.dbbutton, self.dbpath)
        nrows += 1

        self.coll = QLabel("Collection: ")
        self.coll.setAlignment(Qt.AlignRight)
        self.coll.setFixedWidth(fixedwdt)
        self.coll.setFixedHeight(fixedhgt)
        self.colltext = QLabel()
        self.colltext.setFrameStyle(QFrame.Box)
        self.colltext.setFixedWidth(fixedwdt)
        self.colltext.setFixedHeight(fixedhgt)
        self.widlayout.addRow(self.coll, self.colltext)
        nrows += 1

        self.spcmn = QPushButton("&SpecimenData")
        self.spcmn.setShortcut(QKeySequence('S'))
        self.spcmn.setFixedWidth(2*fixedwdt)
        self.spcmn.setFixedHeight(fixedhgt)
        self.spcmn.clicked.connect(self._spcmn_cb)
        self.widlayout.addRow(self.spcmn)
        nrows += 1

        self.id = QLabel("Specimen ID: ")
        self.id.setAlignment(Qt.AlignRight)
        self.id.setFixedWidth(fixedwdt)
        self.id.setFixedHeight(fixedhgt)
        self.idtext = QLabel()
        self.idtext.setFrameStyle(QFrame.Box)
        self.idtext.setFixedWidth(fixedwdt)
        self.idtext.setFixedHeight(fixedhgt)
        self.widlayout.addRow(self.id, self.idtext)
        nrows += 1

        self.genus = QLabel("Specimen Genus: ")
        self.genus.setAlignment(Qt.AlignRight)
        self.genus.setFixedWidth(fixedwdt)
        self.genus.setFixedHeight(fixedhgt)
        self.genustext = QLabel()
        self.genustext.setFrameStyle(QFrame.Box)
        self.genustext.setFixedWidth(fixedwdt)
        self.genustext.setFixedHeight(fixedhgt)
        self.widlayout.addRow(self.genus, self.genustext)
        nrows += 1

        self.species = QLabel("Specimen Species: ")
        self.species.setAlignment(Qt.AlignRight)
        self.species.setFixedWidth(fixedwdt)
        self.species.setFixedHeight(fixedhgt)
        self.speciestext = QLabel()
        self.speciestext.setFrameStyle(QFrame.Box)
        self.speciestext.setFixedWidth(fixedwdt)
        self.speciestext.setFixedHeight(fixedhgt)
        self.widlayout.addRow(self.species, self.speciestext)
        nrows += 1

        self.beginbutton = QPushButton("&Begin Imaging")
        self.beginbutton.setShortcut(QKeySequence('B'))
        self.beginbutton.setFixedWidth(2*fixedwdt)
        self.beginbutton.setFixedHeight(fixedhgt)
        self.beginbutton.clicked.connect(self._beginbutton_cb)
        self.widlayout.addRow(self.beginbutton)
        nrows += 1

        self.exitbutton = QPushButton('E&xit')
        self.exitbutton.setShortcut(QKeySequence('X'))
        self.exitbutton.setFixedWidth(2*fixedwdt)
        self.exitbutton.setFixedHeight(fixedhgt)
        self.exitbutton.clicked.connect(self._exitbutton_cb)
        self.widlayout.addRow(self.exitbutton)
        nrows += 1

        # Main window geometry
        uiwdt = 2*fixedwdt + 3*buff1
        uihgt = nrows*fixedhgt + (nrows + 1)*fixedhgt
        self.setGeometry(50, 50, uiwdt, uihgt)
        self.setFixedSize(uiwdt, uihgt)
        self.setWindowTitle('XyloRef')

        widget = QWidget(self)
        self.setCentralWidget(widget)
        widget.setLayout(self.widlayout)

        self._flatten()

        # Show.
        self.show()
