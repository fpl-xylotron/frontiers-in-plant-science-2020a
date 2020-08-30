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


class XyloRefScreen(object):
    def __init__(self) -> None:
        self.live = None
        self.prev = None
        self.panel = None
        self.side = ["live", "prev"]
        self.snapsave = False
        self.change = False
        self.widgets = {}
        self.live_image()
        self.prev_image()
        self.panel_image()
        # print(self.live.shape, self.prev.shape, self.panel.shape)
        font = OpencvFont(scale=2, thick=2)
        self.spacing = 10
        self.btnhgt = 80
        self.btnwdt = self.panel.shape[1] - 2*self.spacing
        goff = (0, 2048)
        tl = (self.spacing, self.spacing)
        br = ((tl[0] + self.btnhgt), (tl[1] + self.btnwdt))
        self.widgets["snapsave"] = OpencvFillableHButton(tl=tl,
                                                         br=br,
                                                         text="Snap+Save",
                                                         keystroke='s',
                                                         fill_rgb=(50, 50, 50),
                                                         fill=1.0,
                                                         dot_rgb=(0, 0, 0),
                                                         dot_thick=2,
                                                         text_rgb=(200, 200, 200),
                                                         text_font=font,
                                                         text_justify=("left", 0),
                                                         goff=goff)
        tl = ((br[0] + self.spacing), tl[1])
        br = ((tl[0] + self.btnhgt), (tl[1] + self.btnwdt))
        self.widgets["toggle"] = OpencvFillableHButton(tl=tl,
                                                       br=br,
                                                       text="ToggleView",
                                                       keystroke='t',
                                                       fill_rgb=(50, 50, 50),
                                                       fill=1.0,
                                                       dot_rgb=(0, 0, 0),
                                                       dot_thick=2,
                                                       text_rgb=(200, 200, 200),
                                                       text_font=font,
                                                       text_justify=("left", 0),
                                                       goff=goff)
        tl = ((br[0] + self.spacing), tl[1])
        br = ((tl[0] + self.btnhgt), (tl[1] + self.btnwdt))
        self.widgets["change"] = OpencvFillableHButton(tl=tl,
                                                       br=br,
                                                       text="ChangeSpecimen",
                                                       keystroke='c',
                                                       fill_rgb=(50, 50, 50),
                                                       fill=1.0,
                                                       dot_rgb=(0, 0, 0),
                                                       dot_thick=2,
                                                       text_rgb=(200, 200, 200),
                                                       text_font=font,
                                                       text_justify=("left", 0),
                                                       goff=goff)

        self.live_image()
        self.prev_image()
        self.panel_image()

    def process(self, keystroke: str,
                      mousexy: tuple) -> None:
        selection = None
        for n, w in self.widgets.items():
            if w.keymatches(keystroke):
                selection = n
            if mousexy and w.contains((mousexy[1], mousexy[0])):
                selection = n

        self._process_snapsave(selection)
        self._process_toggle(selection)
        self._process_change(selection)

    def _process_snapsave(self, selection: str) -> None:
        if selection == "snapsave":
            self.snapsave = True

    def _process_toggle(self, selection: str) -> None:
        if selection == "toggle":
            self.side.reverse()

    def _process_change(self, selection: str) -> None:
        if selection == "change":
            self.change = True

    def _left_image(self) -> np.ndarray:
        if self.side[0] == "live":
            return self.live
        else:
            return self.prev

    def _right_image(self) -> np.ndarray:
        if self.side[1] == "live":
            return self.live
        else:
            return self.prev

    def image(self):
        self.panel_image()
        return np.hstack([self._left_image(), self.panel, self._right_image()])

    def panel_image(self):
        self.panel = np.zeros((2048, 1024, 3), dtype = np.uint8)
        if not self.widgets:
            return
        for n, w in self.widgets.items():
            self.panel = w(self.panel)

    def live_image(self, img: np.ndarray = None) -> None:
        if img is not None:
            self.live = img.copy()
        else:
            self.live = np.zeros((2048, 2048, 3), dtype = np.uint8)

    def prev_image(self, img: np.ndarray = None) -> None:
        if img is not None:
            self.prev = img.copy()
        else:
            self.prev = np.zeros((2048, 2048, 3), dtype = np.uint8)

    def is_snapsave(self) -> bool:
        return self.snapsave

    def is_change(self) -> bool:
        return self.change

    def reset_state(self) -> None:
        self.snapsave = False
        self.change = False
