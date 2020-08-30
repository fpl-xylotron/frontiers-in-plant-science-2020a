from typing import List, Union, Tuple, Dict, Any, Sequence
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import cv2 as cv
add_paths = [Path(__file__).resolve().parent.parent.parent.parent/"source",
             Path(__file__).resolve().parent.parent.parent/"source"]
for p in add_paths:
    if str(p) not in sys.path:
        sys.path.append(str(p))
from opencv_shapes import *
from opencv_widgets import *
from opencv_events_handler import *


class XyloInfScreen(object):
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.limg = None
        self.rimg = None
        self.pimg = None
        self.livefeed = False
        self.identify = False
        self.classidx = None
        self.barset = False
        self.classchange = False
        self.exit = False
        self.widgets = {}
        self.left_image()
        self.right_image()
        self.panel_image()
        font = OpencvFont(scale=2, thick=2)
        self.spacing = 10
        self.btnhgt = 80
        self.btnwdt = self.pimg.shape[1] - 2*self.spacing
        goff = (0, 2048)
        tl = (self.spacing, self.spacing)
        br = ((tl[0] + self.btnhgt), (tl[1] + self.btnwdt))
        self.widgets["livefeed"] = OpencvFillableHButton(tl=tl,
                                                         br=br,
                                                         text="Live Feed",
                                                         keystroke='l',
                                                         # box_rgb=(50, 50, 50),
                                                         fill_rgb=(50, 50, 50),
                                                         fill=1.0,
                                                         dot_rgb=(200, 200, 200),
                                                         dot_thick=2,
                                                         text_rgb=(200, 200, 200),
                                                         text_font=font,
                                                         text_justify=("left", 0),
                                                         goff=goff)
        tl = ((br[0] + self.spacing), tl[1])
        br = ((tl[0] + self.btnhgt), (tl[1] + self.btnwdt))
        self.widgets["identify"] = OpencvFillableHButton(tl=tl,
                                                         br=br,
                                                         text="Identify",
                                                         keystroke='i',
                                                         # box_rgb=(50, 50, 50),
                                                         fill_rgb=(50, 50, 50),
                                                         fill=1.0,
                                                         dot_rgb=(200, 200, 200),
                                                         dot_thick=2,
                                                         text_rgb=(200, 200, 200),
                                                         text_font=font,
                                                         text_justify=("left", 0),
                                                         goff=goff)
        tl = ((br[0] + self.spacing), tl[1])
        br = ((tl[0] + self.btnhgt), (tl[1] + self.btnwdt))
        self.widgets["exit"] = OpencvFillableHButton(tl=tl,
                                                     br=br,
                                                     text="Exit",
                                                     keystroke='x',
                                                     fill_rgb=(50, 50, 50),
                                                     fill=1.0,
                                                     dot_rgb=(0, 0, 0),
                                                     dot_thick=2,
                                                     text_rgb=(200, 200, 200),
                                                     text_font=font,
                                                     text_justify=("left", 0),
                                                     goff=goff)
        self.classes = []
        whgt = self.btnhgt + self.spacing
        cfont = OpencvFont(scale=1.75, thick=2)
        for i in range(self.k):
            nsteps = (self.k + 1 - i)
            tl = ((self.pimg.shape[0] - nsteps*whgt), self.spacing)
            br = ((tl[0] + self.btnhgt), (tl[1] + self.btnwdt))
            self.widgets[f"class{i}"] = OpencvFillableHButton(tl=tl,
                                                              br=br,
                                                              keystroke=str(i + 1),
                                                              box_rgb=(50, 50, 50),
                                                              box_thick=2,
                                                              dot_rgb=(200, 200, 200),
                                                              dot_thick=2,
                                                              fill=0.0,
                                                              fill_rgb=(19, 129, 19),
                                                              text_rgb=(200, 200, 200),
                                                              text_font=cfont,
                                                              text_justify=("left", 0),
                                                              goff=goff)
            self.classes.append(f"class{i}")

        tl = ((self.pimg.shape[0] - whgt), self.spacing)
        br = ((tl[0] + self.btnhgt), (tl[1] + self.btnwdt))
        self.widgets["others"] = OpencvFillableHButton(tl=tl,
                                                       br=br,
                                                       box_rgb=(50, 50, 50),
                                                       box_thick=2,
                                                       dot_rgb=(0, 0, 0),
                                                       dot_thick=1,
                                                       fill=0.0,
                                                       fill_rgb=(19, 129, 19),
                                                       text_rgb=(200, 200, 200),
                                                       text_font=cfont,
                                                       text_justify=("left", 0),
                                                       goff=goff)
        self.left_image()
        self.right_image()
        self.panel_image()

    def process(self, keystroke: str,
                      mousexy: tuple) -> None:
        selection = None
        for n, w in self.widgets.items():
            if w.keymatches(keystroke):
                selection = n
            if mousexy and w.contains((mousexy[1], mousexy[0])):
                selection = n

        self._process_livefeed(selection)
        self._process_identify(selection)
        self._process_classidx(selection)
        self._process_exit(selection)

    def _process_livefeed(self, selection: str) -> None:
        if selection and selection == "livefeed":
            self.livefeed = not self.livefeed
            self.identify = False
            self.classidx = None
            self.barset = False
            self.classchange = False
            self.classes_data()

    def _process_identify(self, selection: str) -> None:
        if selection and (self.livefeed or self.identify) and selection == "identify":
            self.identify = not self.identify
            self.livefeed = False
            self.classidx = None
            self.barset = False
            self.classchange = False
            self.classes_data()
            if not self.identify:
                self.classidx = None
                self.classchange = True

    def _process_classidx(self, selection: str) -> None:
        if selection and "class" in selection and self.identify:
            self.classidx = selection
            self.classchange = True

    def _process_exit(self, selection: str) -> None:
        if selection and selection == "exit":
            self.exit = True

    def image(self):
        self.panel_image()
        return np.hstack([self.limg, self.pimg, self.rimg])

    def panel_image(self):
        self.pimg = np.zeros((2048, 1024, 3), dtype = np.uint8)
        if not self.widgets:
            return
        # self._print_state()
        self.widgets["livefeed"].set_active(self.livefeed)
        self.widgets["identify"].set_active(self.identify)
        for c in self.classes:
            self.widgets[c].set_active(c == self.classidx)
        for n, w in self.widgets.items():
            self.pimg = w(self.pimg)

    def left_image(self, img: np.ndarray = None) -> None:
        if img is not None:
            self.limg = img.copy()
        else:
            self.limg = np.zeros((2048, 2048, 3), dtype = np.uint8)

    def right_image(self, img: np.ndarray = None) -> None:
        if img is not None:
            self.rimg = img.copy()
        else:
            self.rimg = np.zeros((2048, 2048, 3), dtype = np.uint8)

    def classes_data(self, kfills: np.ndarray = None,
                           ktexts: List[str] = None) -> None:
        '''
        Provide k-array i.e. others will be computed.
        '''
        for c in self.classes + ['others']:
            self.widgets[c].set_fill(0.0)
            self.widgets[c].set_text(None)
        self.classchange = False

        if kfills is None or ktexts is None:
            return
        for (c, f, t) in zip(self.classes, kfills, ktexts):
            self.widgets[c].set_fill(f)
            self.widgets[c].set_text(t)
        othersfill = 1.0 - np.sum(kfills)
        otherspercent = int(round(othersfill*100))
        self.widgets['others'].set_fill(othersfill)
        self.widgets['others'].set_text(f"Others({otherspercent}%)")
        self.classidx = "class0"
        self.barset = True
        self.classchange = True

    def is_livefeed(self) -> bool:
        return self.livefeed

    def is_identify(self) -> bool:
        return self.identify

    def is_barset(self) -> bool:
        return self.barset

    def is_exit(self) -> bool:
        return self.exit

    def is_classchange(self) -> bool:
        return self.classchange

    def get_classidx(self) -> Union[int, None]:
        if self.classidx:
            # print(self.classidx, self.classes)
            return self.classes.index(self.classidx)
        return None

    def set_classidx(self, idx: int) -> None:
        self.classidx = idx

    def unset_classchange(self) -> None:
        self.classchange = False

    def _print_state(self) -> None:
        print(f"self.livefeed: {self.livefeed}")
        print(f"self.identify: {self.identify}")
        print(f"self.classidx: {self.classidx}")
        print(f"self.exit: {self.exit}")
        print()
