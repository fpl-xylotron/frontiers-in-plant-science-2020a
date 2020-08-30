import sys
from pathlib import Path
import numpy as np
import importlib
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QFileDialog, QLineEdit)
from PyQt5.QtGui import QKeySequence
import pprint
import torch
import torch.nn as nn
import cv2 as cv
add_paths = [str(Path(__file__).resolve().parent.parent.parent.parent/'source'),
             str(Path(__file__).resolve().parent.parent.parent/'source')]
for p in add_paths:
    if p not in sys.path:
        sys.path.append(p)
from opencv_shapes import *
from opencv_events_handler import *
from xylo_scope import *
from xylo_inf_screen import *
from xylo_logger import *

pp = pprint.PrettyPrinter().pprint

class XyloInfUI(QMainWindow):
    def __init__(self, k=3, model_path="", screen_size=(1080, 1920)):
        super(XyloInfUI, self).__init__(None, Qt.WindowStaysOnTopHint)
        self.k = k
        self.xylodir = Path(__file__).resolve().parent.parent.parent
        self.appsdir = Path(__file__).resolve().parent.parent
        self.infdir = Path(__file__).resolve().parent
        self.clsfrname = ''
        self.classifier = None
        self.classnames = []
        self.handler = None
        self.screen = None
        self.winname = None
        self.modelpath = model_path
        self.scrnsize = screen_size

        cfg_dir = self.xylodir/"camcfgs"
        self.xs = XyloScope(config_dir=cfg_dir)
        xstatus = self.xs.start_camera()
        if not xstatus:
            sys.exit()
        self.xs.configure_camera()

        self._init_ui()
        if len(self.modelpath) > 0:
            # one fixed model
            self._loadmodel_cb()
            self._proceed_cb()
            self._close_cb()

    def _init_ui(self):
        self.uiwdt = 300
        self.uihgt = 200
        self.spacing = 10
        self.btnwdt = self.uiwdt - 2*self.spacing
        self.btnhgt = 35

        self.setWindowTitle('XyloInf')
        self.setGeometry(50, 50, self.uiwdt, self.uihgt)
        self.setFixedSize(self.uiwdt, self.uihgt)

        ox, oy= self.spacing, self.spacing
        self.loadmodel = QPushButton("LoadModel", self)
        self.loadmodel.setShortcut(QKeySequence('M'))
        self.loadmodel.setFixedWidth(self.btnwdt)
        self.loadmodel.setFixedHeight(self.btnhgt)
        self.loadmodel.move(ox, oy)
        self.loadmodel.clicked.connect(self._loadmodel_cb)

        ox = self.spacing
        oy = self.loadmodel.pos().y() + self.loadmodel.size().height() + self.spacing
        self.modelname = QLineEdit(self)
        self.modelname.setFixedWidth(self.btnwdt)
        self.modelname.setFixedHeight(self.btnhgt)
        self.modelname.move(ox, oy)

        ox = self.spacing
        oy = self.modelname.pos().y() + self.modelname.size().height() + self.spacing
        self.proceed = QPushButton("Proceed", self)
        self.proceed.setShortcut(QKeySequence('P'))
        self.proceed.setFixedWidth(self.btnwdt)
        self.proceed.setFixedHeight(self.btnhgt)
        self.proceed.move(ox, oy)
        self.proceed.clicked.connect(self._proceed_cb)

        ox = self.spacing
        oy = self.proceed.pos().y() + self.proceed.size().height() + self.spacing
        self.close = QPushButton("Close", self)
        self.close.setShortcut(QKeySequence('C'))
        self.close.setFixedWidth(self.btnwdt)
        self.close.setFixedHeight(self.btnhgt)
        self.close.move(ox, oy)
        self.close.clicked.connect(self._close_cb)

        self.show()

    def _loadmodel_cb(self):
        if len(self.modelpath) == 0:
            self.modelpath = QFileDialog.getExistingDirectory(self,
                                                              'Choose model directory',
                                                              str(self.infdir/"models"))

        self.clsfrname = ""
        if len(self.modelpath) > 0:
            self.clsfrname = Path(self.modelpath).name
            self.modeldir = Path(self.modelpath)/"model"
            self.imgdbdir = Path(self.modelpath)/"imgdb"

        if self.modeldir:
            sys.path.append(str(self.modeldir))
            labels_file = self.modeldir/'class_labels.txt'
            weights_file = self.modeldir/'wts.pth'
            import model
            self.classifier, self.classnames = model.get(labels_file,
                                                         weights_file)
            self.classifier.eval()
        else:
            self.classifier = None
            self.classnames = []

        self.modelname.setText(self.clsfrname)

    def _proceed_cb(self):
        if self.classifier:
            self.setHidden(True)
            self._run_screen()
            cv.resizeWindow(self.winname, 50, 50)
            cv.moveWindow(self.winname, 55, 55)
            self.setHidden(False)
            self.setFocus(True)
            self.classifier = None
            self.classnames = []
            self.clsfrname = ''
            self.modelname.setText(self.clsfrname)

    def _close_cb(self):
        if self.xs.is_capturing():
            self.xs.stop_capturing()
        self.xs.cleanup()
        del self.xs
        cv.destroyAllWindows()
        sys.exit()

    def _draw_splash(self, imgs: List[np.ndarray],
                           splash: np.ndarray) -> np.ndarray:
        rowstep = splash.shape[0]//len(imgs)
        midcol = splash.shape[1]//2
        for (idx, img) in enumerate(imgs):
            rszimg = cv.resize(img, (img.shape[1]*2, img.shape[0]*2))
            midrow = (2*idx + 1)*rowstep//2
            trow = midrow - rszimg.shape[0]//2
            brow = trow + rszimg.shape[0]
            lcol = midcol - rszimg.shape[1]//2
            rcol = lcol + rszimg.shape[1]
            splash[trow:brow, lcol:rcol, :] = rszimg
        return splash

    def _get_splash(self) -> Tuple[np.ndarray, np.ndarray]:
        if not (self.infdir/"splash_specification.txt").exists():
            font = OpencvFont(scale=4.5, thick=3)
            ltxt = OpencvText(text="XyloInf",
                              rgb=(255, 255, 255),
                              font=font,
                              tl=(724, 724),
                              br=(1324, 1324),
                              justify=('center', 0))
            lsplash = ltxt(np.zeros((2048, 2048, 3), dtype=np.uint8))
            return (lsplash, lsplash)

        with open(self.infdir/"splash_specification.txt", 'r') as fh:
            lines = fh.readlines()
        lines = ["fs", "uw"] + [line for line in lines if line.strip()]
        impths = [self.xylodir/"resrcs"/f"{line.strip()}.png" for line in lines]
        imgs = [cv.imread(str(p)) for p in impths]
        limgs, rimgs = imgs[:], imgs[:]
        if len(imgs) > 3:
            limgs, rimgs = imgs[:2], imgs[2:]
        lsplash = self._draw_splash(limgs, np.zeros((2048, 2048, 3), dtype=np.uint8))
        rsplash = self._draw_splash(rimgs, np.zeros((2048, 2048, 3), dtype=np.uint8))
        return (lsplash, rsplash)

    def _get_infwin_size(self):
        hgt = self.scrnsize[0] - 100
        wdt = self.scrnsize[1] - 100
        while (wdt%10) > 0:
            wdt -= 1
        quanta = wdt//10
        hgt = quanta*4
        return (wdt, hgt)

    def _run_screen(self):
        logger = XyloLogger(top_dir=Path.home()/"xylo_inf_data",
                            camera_serial_number=self.xs.serial_number())
        cfgstrmaker = ConfigStringMaker(xs=self.xs)
        self.handler = OpencvEventsHandler()
        self.screen = XyloInfScreen(k=3)
        (lsplash, rsplash) = self._get_splash()
        self.screen.left_image(lsplash)
        self.screen.right_image(rsplash)
        self.winname = "XyloInf"
        (winwdt, winhgt) = self._get_infwin_size()
        cv.namedWindow(self.winname, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.winname, winwdt, winhgt)
        cv.moveWindow(self.winname, 50, 50)
        cv.setMouseCallback(self.winname, mouse_cb, self.handler)
        cv.imshow(self.winname, self.screen.image())
        self.screen.unset_classchange()
        self.handler.reset_keystroke()
        self.handler.reset_mouseevent()
        img4pred = None
        topimgs = []
        identify_clear = True
        while True:
            self.handler.log_keystroke(cv.waitKey(30))
            self.screen.process(keystroke=self.handler.get_keystroke(),
                                mousexy=self.handler.get_mousexy())
            if self.screen.is_livefeed():
                if not self.xs.is_capturing():
                    self.xs.start_capturing()
                    self.screen.right_image()
                self.screen.left_image(self.xs.get_image()[:, :, ::-1])
                topimgs = []

            if self.screen.is_identify():
                identify_clear = False
                if not self.screen.is_barset():
                    # capture and display as left image
                    img4pred = self.xs.get_image()[:, :, ::-1]
                    if self.xs.is_capturing():
                        self.xs.stop_capturing()
                    logger.save(img=img4pred,
                                log=cfgstrmaker.string())
                    self.screen.left_image(img4pred)

                    # do inference
                    import data
                    with torch.no_grad():
                        out = self.classifier(data.tensorify(img4pred))
                        out = nn.Softmax(dim=1)(out)
                    kk = min(len(self.classnames), self.k)
                    # topprbs, topidxs = torch.topk(out, self.k)
                    topprbs, topidxs = torch.topk(out, kk)
                    topprbs = topprbs.view(topprbs.numel()).numpy()
                    topidxs = topidxs.view(topidxs.numel()).numpy()
                    topclss = [self.classnames[i] for i in topidxs]
                    texts = []
                    for prb, cls in zip(topprbs, topclss):
                        texts.append(f"{cls}({int(round(prb*100))}%)")
                    topimgs = []
                    for topidx in topidxs:
                        fn = f"{str(topidx).zfill(3)}.png"
                        topimgs.append(cv.imread(str(self.imgdbdir/fn)))
                    self.screen.classes_data(kfills=topprbs, ktexts=texts)
                    if self.screen.get_classidx() is not None:
                        self.screen.right_image(topimgs[self.screen.get_classidx()])

            if not identify_clear and not self.screen.is_identify():
                self.screen.left_image()
                self.screen.right_image()
                identify_clear = True

            if self.screen.is_classchange():
                if self.screen.get_classidx() is not None:
                    self.screen.right_image(topimgs[self.screen.get_classidx()])

            if self.screen.is_exit():
                if self.xs.is_capturing():
                    self.xs.stop_capturing()
                break

            cv.imshow(self.winname, self.screen.image())
            self.screen.unset_classchange()
            self.handler.reset_keystroke()
            self.handler.reset_mouseevent()
        self.modelpath = ""
