from typing import List, Union, Tuple, Dict, Any, Sequence
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as cv
from opencv_shapes import *

# Coordinates should be in the form (row, col).
class OpencvBoxWidget(object):
    def __init__(self, tl: tuple,
                       br: tuple,
                       text: str,
                       text_justify: Tuple[str, int],
                       font: OpencvFont,
                       keystroke: str = None,
                       goff: tuple = (0, 0)) -> None:
        self.tl = tl
        self.br = br
        self.text = text
        self.text_justify = text_justify
        self.font = font
        self.keystroke = keystroke
        self.gtl = (self.tl[0] + goff[0], self.tl[1] + goff[1])
        self.gbr = (self.br[0] + goff[0], self.br[1] + goff[1])
        self.active = False

    def _create_shapes(self):
        pass

    def contains(self, pt: tuple) -> bool:
        if pt is None:
            return False
        return (pt[0] >= self.gtl[0] and pt[0] <= self.gbr[0] and
                pt[1] >= self.gtl[1] and pt[1] <= self.gbr[1])

    def keymatches(self, keystroke: str) -> bool:
        if not keystroke or not self.keystroke:
            return False
        return keystroke == self.keystroke

    def set_text(self, text: str) -> None:
        self.text = text
        self._create_shapes()

    def set_active(self, status: bool) -> None:
        self.active = status

    def is_active(self) -> bool:
        return self.active

    def _get_rgbo(self, color: tuple) -> None:
        if color is None:
            return None
        if len(color) == 3:
            return color + (100, )
        return color


class OpencvFillableHButton(OpencvBoxWidget):
    def __init__(self, tl: tuple,
                       br: tuple,
                       keystroke: str = None,
                       text: str = None,
                       box_rgb: tuple = None,
                       box_thick: int = 1,
                       dot_rgb: tuple = None,
                       dot_thick: int = 1,
                       fill: float = 0.0,
                       fill_rgb: tuple = None,
                       text_rgb: tuple = None,
                       text_font: OpencvFont = OpencvFont(),
                       text_justify: Tuple[str, int] = ("left", 0),
                       goff: tuple = (0, 0)) -> None:
        super(OpencvFillableHButton, self).__init__(tl,
                                                    br,
                                                    text,
                                                    text_justify,
                                                    text_font,
                                                    keystroke,
                                                    goff)

        self.box_rgbo = self._get_rgbo(box_rgb) # box_rgb + (100,)
        self.box_thick = box_thick
        self.dot_rgbo = self._get_rgbo(dot_rgb) # dot_rgb + (100,)
        self.dot_thick = dot_thick
        self.fill = fill
        self.fill_rgbo = self._get_rgbo(fill_rgb) # fill_rgb + (100,)
        self.text_rgbo = self._get_rgbo(text_rgb) # text_rgb + (100,)
        self.shapes = {"box": None,
                       "edot": None,
                       "fdot": None,
                       "fill": None,
                       "text": None}
        self._create_shapes()

    def _create_shapes(self) -> None:
        self.shapes = {"box": None,
                       "edot": None,
                       "fdot": None,
                       "fill": None,
                       "text": None}
        if self.box_rgbo is not None:
            self.shapes["box"] = OpencvBox(self.tl,
                                           self.br,
                                           contour_rgb=self.box_rgbo[0:3],
                                           contour_thick=self.box_thick)

        hgt = self.br[0] - self.tl[0] + 1
        halfhgt = hgt//2

        dot_cols = None
        if self.dot_rgbo is not None:
            center = ((self.tl[0] + halfhgt), (self.tl[1] + halfhgt))
            radius = halfhgt//2
            self.shapes["edot"] = OpencvCircle(center,
                                               radius,
                                               contour_rgb=self.dot_rgbo[0:3],
                                               contour_thick=self.dot_thick)
            self.shapes["fdot"] = OpencvCircle(center,
                                               radius,
                                               contour_rgb=self.dot_rgbo[0:3],
                                               contour_thick=self.dot_thick,
                                               fill_rgbo=self.dot_rgbo)
            dot_cols = (self.tl[1], self.tl[1] + hgt + 1)

        if self.fill_rgbo is not None:
            tl = self.tl
            if dot_cols:
                tl = (self.tl[0], (self.tl[1] + hgt))
            wdt = self.br[1] - tl[1] + 1
            br = (self.br[0], (tl[1] + int(self.fill*wdt)))
            if (br[1] - tl[1]) == 0:
                rgbo = self.box_rgbo
            else:
                rgbo = self.fill_rgbo
            self.shapes["fill"] = OpencvBox(tl,
                                            br,
                                            contour_rgb=rgbo[0:3],
                                            contour_thick=self.box_thick,
                                            fill_rgbo=rgbo)

        if self.text and self.text_rgbo:
            txt_br = self.br
            txt_tl = self.tl
            if dot_cols:
                txt_tl = (txt_tl[0], dot_cols[1])
            self.shapes["text"] = OpencvText(text=self.text,
                                             rgb=self.text_rgbo[0:3],
                                             font=self.font,
                                             tl=txt_tl,
                                             br=txt_br,
                                             justify=self.text_justify)

    def set_fill(self, fill: float) -> None:
        self.fill = fill
        self._create_shapes()

    def __call__(self, img: np.ndarray) -> None:
        if "box" in self.shapes and self.shapes["box"]:
            img = self.shapes["box"](img)
        if self.active and "fdot" in self.shapes and self.shapes["fdot"]:
            img = self.shapes["fdot"](img)
        if not self.active and "edot" in self.shapes and self.shapes["edot"]:
            img = self.shapes["edot"](img)
        if "fill" in self.shapes and self.shapes["fill"]:
            img = self.shapes["fill"](img)
        if "text" in self.shapes and self.shapes["text"]:
            img = self.shapes["text"](img)
        return img
