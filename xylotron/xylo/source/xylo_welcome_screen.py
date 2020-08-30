from pathlib import Path
import sys
add_paths = [Path("../../source").resolve()]
for p in add_paths:
    if str(p) not in sys.path:
        sys.path.append(str(p))
from typing import List, Union, Tuple, Dict, Any, Sequence
from collections import defaultdict
import numpy as np
import cv2 as cv
from opencv_shapes import *
from opencv_widgets import *
from opencv_events_handler import *


class XyloWelcomeScreen(object):
    def __init__(self, app_name: str,
                       text: str = None,
                       goff: tuple = (0, 0)) -> None:
        self.img = None
        self._reset_img()
        font = OpencvFont(scale=1, thick=2)
        self.btnhgt = 50
        self.btnwdt = 300
        self.spacing = 10
        self.widgets = {}
        tl = (self.img.shape[0] - self.spacing - 3*self.btnhgt,
              self.img.shape[1] - 3*(self.btnwdt + self.spacing))
        br = (tl[0] + self.btnhgt, tl[1] + self.btnwdt)
        self.widgets["yaccept"] = OpencvFillableHButton(tl=tl,
                                                        br=br,
                                                        text="Accept",
                                                        keystroke='a',
                                                        box_rgb=(50, 50, 50),
                                                        box_thick=1,
                                                        dot_rgb=(0, 229, 0),
                                                        dot_thick=2,
                                                        text_rgb=(200, 200, 200),
                                                        text_font=font,
                                                        text_justify=("left", 0))
        tl = (tl[0], tl[1] + self.btnwdt + self.spacing)
        br = (tl[0] + self.btnhgt, tl[1] + self.btnwdt)
        self.widgets["naccept"] = OpencvFillableHButton(tl=tl,
                                                        br=br,
                                                        text="Do not accept",
                                                        keystroke='d',
                                                        box_rgb=(50, 50, 50),
                                                        box_thick=1,
                                                        dot_rgb=(229, 0, 0),
                                                        dot_thick=2,
                                                        text_rgb=(200, 200, 200),
                                                        text_font=font,
                                                        text_justify=("left", 0))
        tl = (tl[0], tl[1] + self.btnwdt + self.spacing)
        br = (tl[0] + self.btnhgt, tl[1] + self.btnwdt)
        self.widgets["submit"] = OpencvFillableHButton(tl=tl,
                                                       br=br,
                                                       text="Submit",
                                                       keystroke='s',
                                                       box_rgb=(50, 50, 50),
                                                       box_thick=1,
                                                       fill=0.0,
                                                       fill_rgb=(50, 50, 50),
                                                       text_rgb=(200, 200, 200),
                                                       text_font=font,
                                                       text_justify=("center", 0))
        tl = (10, self.img.shape[1]//2 - 200)
        br = (250, self.img.shape[1]//2 + 200)
        font = OpencvFont(scale=2.5, thick=3)
        self.appname = OpencvText(text=str(app_name),
                                  rgb=(68, 188, 238),
                                  font=font,
                                  tl=tl,
                                  br=br,
                                  justify=('center', 0))
        self.finished = False

        p = Path(__file__).resolve().parent.parent/'resrcs/weltxt.png'
        self.txtimg = cv.imread(str(p))
        p = Path(__file__).resolve().parent.parent/'resrcs/uw.png'
        self.uwimg = cv.imread(str(p))
        p = Path(__file__).resolve().parent.parent/'resrcs/fs.png'
        self.fsimg = cv.imread(str(p))

    def process(self, keystroke: str,
                      mousexy: tuple) -> None:
        selection = None
        for n, w in self.widgets.items():
            if w.keymatches(keystroke):
                selection = n
            if mousexy and w.contains((mousexy[1], mousexy[0])):
                selection = n

        self._process_yaccept(selection)
        self._process_naccept(selection)
        self._process_submit(selection)

    def _process_yaccept(self, selection: str) -> None:
        if selection and selection == "yaccept":
            yaccept = not self.widgets['yaccept'].is_active()
            naccept = False
            self.widgets['yaccept'].set_active(yaccept)
            self.widgets['naccept'].set_active(naccept)
            self.widgets['submit'].set_fill(0.0)
            if yaccept or naccept:
                self.widgets['submit'].set_fill(1.0)


    def _process_naccept(self, selection: str) -> None:
        if selection and selection == "naccept":
            naccept = not self.widgets['naccept'].is_active()
            yaccept = False
            self.widgets['yaccept'].set_active(yaccept)
            self.widgets['naccept'].set_active(naccept)
            self.widgets['submit'].set_fill(0.0)
            if yaccept or naccept:
                self.widgets['submit'].set_fill(1.0)

    def _process_submit(self, selection) -> None:
        yaccept = self.widgets['yaccept'].is_active()
        naccept = self.widgets['naccept'].is_active()
        if selection and selection == "submit" and (yaccept or naccept):
            self.finished = True

    def image(self):
        self._draw_widgets()
        return self.img

    def _draw_widgets(self):
        self._reset_img()
        self.img = self.appname(self.img)

        startrow =  self.img.shape[0] - self.spacing - 4*self.btnhgt - self.txtimg.shape[0]
        stoprow = startrow + self.txtimg.shape[0]
        excess = self.img.shape[1] - self.txtimg.shape[1]
        startcol = excess//2
        stopcol = startcol + self.txtimg.shape[1]
        self.img[startrow:stoprow, startcol:stopcol, :] = self.txtimg

        startrow = self.img.shape[0]//2 - self.uwimg.shape[0]//2
        stoprow = startrow + self.uwimg.shape[0]
        startcol = self.img.shape[1]//2 + self.img.shape[1]//4 - self.uwimg.shape[1]//2
        stopcol = startcol + self.uwimg.shape[1]
        self.img[startrow:stoprow, startcol:stopcol, :] = self.uwimg

        startrow = self.img.shape[0]//2 - self.fsimg.shape[0]//2
        stoprow = startrow + self.fsimg.shape[0]
        startcol = self.img.shape[1]//2 - self.img.shape[1]//4 - self.fsimg.shape[1]//2
        stopcol = startcol + self.fsimg.shape[1]
        self.img[startrow:stoprow, startcol:stopcol, :] = self.fsimg

        for n, w in self.widgets.items():
            self.img = w(self.img)

    def _reset_img(self):
        self.img = np.zeros((900, 1000, 3), dtype = np.uint8)

    def is_finished(self):
        return (self.finished, self.widgets['yaccept'].is_active())


def run_welcome_screen(app_name: str) -> bool:
    handler = OpencvEventsHandler()
    screen = XyloWelcomeScreen(app_name=app_name)
    img = screen.image()

    win_name = "Welcome"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, img.shape[1] + 10, img.shape[0] + 10)
    cv.setMouseCallback(win_name, mouse_cb, handler)
    cv.imshow(win_name, img)
    while True:
        handler.log_keystroke(cv.waitKey(30))
        screen.process(keystroke=handler.get_keystroke(),
                       mousexy=handler.get_mousexy())
        cv.imshow(win_name, screen.image())
        if handler.is_keystroke('x') or screen.is_finished()[0]:
            break
        handler.reset_keystroke()
        handler.reset_mouseevent()
    cv.destroyAllWindows()

    return screen.is_finished()[1]
