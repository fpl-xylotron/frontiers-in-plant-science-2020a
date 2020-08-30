from typing import List, Union, Tuple, Dict, Any, Sequence
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as cv

# Coordinates should be in the form (row, col).

class OpencvShapeCompositor(object):
    def __init__(self):
        pass

    def get_pat(self, srcimg: np.ndarray,
                        row_range: tuple,
                        col_range: tuple) -> Tuple[np.ndarray, tuple, tuple, tuple, tuple]:
        # short names
        (s_tr, s_br) = row_range
        (s_lc, s_rc) = col_range

        # dims of images
        (s_nr, s_nc) = srcimg.shape[:2]
        (p_nr, p_nc) = ((s_br - s_tr + 1), (s_rc - s_lc + 1))

        # region of src image
        s_tr1 = max(0, s_tr)
        s_br1 = min(s_br, (s_nr - 1))
        s_lc1 = max(0, s_lc)
        s_rc1 = min(s_rc, (s_nc - 1))

        # region of dst image
        p_tr = s_tr1 - s_tr
        p_br = p_nr + s_br1 - s_br
        p_lc = s_lc1 - s_lc
        p_rc = p_nc + s_rc1 - s_rc

        # make src image region compatible with slice notation
        s_br1 = s_br1 + 1
        s_rc1 = s_rc1 + 1

        # make the pat
        pat = np.zeros((p_nr, p_nc, 3), dtype=np.uint8)
        pat[p_tr:p_br, p_lc:p_rc, :] = srcimg[s_tr1:s_br1, s_lc1:s_rc1, :].copy()
        # rename for ease of reading
        s_rows = (s_tr1, s_br1)
        s_cols = (s_lc1, s_rc1)
        p_rows = (p_tr, p_br)
        p_cols = (p_lc, p_rc)

        return pat, s_rows, s_cols, p_rows, p_cols

    def composite(self, res: np.ndarray,
                        pat: np.ndarray,
                        s_rows: tuple,
                        s_cols: tuple,
                        p_rows: tuple,
                        p_cols: tuple,
                        p_opac: int) -> np.ndarray:
        (str, sbr) = s_rows
        (slc, src) = s_cols
        (ptr, pbr) = p_rows
        (plc, prc) = p_cols
        if p_opac == 100:
            res[str:sbr, slc:src, :] = pat[ptr:pbr, plc:prc, :]
        else:
            opac = p_opac/100.0
            res[str:sbr, slc:src, :] = cv.addWeighted(pat[ptr:pbr, plc:prc, :],
                                                      opac,
                                                      res[str:sbr, slc:src, :],
                                                      (1.0 - opac),
                                                      0)
        return res


class OpencvBox(OpencvShapeCompositor):
    # coordinates provided in row, col format
    def __init__(self, tl: tuple,
                       br: tuple,
                       contour_rgb: tuple = None,
                       contour_thick: int = 1,
                       fill_rgbo: tuple = None) -> None:
        super(OpencvBox, self).__init__()
        self.tl = tl
        self.br = br
        self.cntr_rgb = contour_rgb
        self.cntr_thick = contour_thick
        self.fill_rgbo = fill_rgbo

    def __call__(self, img: np.ndarray) -> None:
        # draw the rectangle
        if self.fill_rgbo:
            buff = 10
            # find the row and col ranges for the figure
            row_range = (self.tl[0] - buff, self.br[0] + buff)
            col_range = (self.tl[1] - buff, self.br[1] + buff)

            # extract pat and the roi map between img and pat
            pat, s_rows, s_cols, p_rows, p_cols = self.get_pat(img,
                                                                   row_range,
                                                                   col_range)

            # translate coords from img system to pat system
            p_tl = ((self.tl[0] - row_range[0]), (self.tl[1] - col_range[0]))
            p_br = ((self.br[0] - row_range[0]), (self.br[1] - col_range[0]))

            rgb = self.fill_rgbo[0:3]
            cv.rectangle(pat,
                         (p_tl[1], p_tl[0]),
                         (p_br[1], p_br[0]),
                         rgb[::-1],
                         thickness=-1,
                         lineType=cv.LINE_AA)
            img = self.composite(img,
                                 pat,
                                 s_rows,
                                 s_cols,
                                 p_rows,
                                 p_cols,
                                 self.fill_rgbo[3])

        if self.cntr_rgb:
            rgb = self.cntr_rgb[0:3]
            cv.rectangle(img,
                         (self.tl[1], self.tl[0]),
                         (self.br[1], self.br[0]),
                         rgb[::-1],
                         thickness=self.cntr_thick,
                         lineType=cv.LINE_AA)

        return img

class OpencvCircle(OpencvShapeCompositor):
    def __init__(self, center: tuple,
                       radius: int,
                       contour_rgb: tuple = None,
                       contour_thick: int = 1,
                       fill_rgbo: tuple = None) -> None:
        super(OpencvCircle, self).__init__()
        self.center = center
        self.radius = radius
        self.cntr_rgb = contour_rgb
        self.cntr_thick = contour_thick
        self.fill_rgbo = fill_rgbo

    def __call__(self, img: np.ndarray) -> None:
        # draw the rectangle
        if self.fill_rgbo:
            buff = self.radius + 10
            # find the row and col ranges for the figure
            row_range = (self.center[0] - buff, self.center[0] + buff)
            col_range = (self.center[1] - buff, self.center[1] + buff)

            # extract pat and the roi map between img and pat
            pat, s_rows, s_cols, p_rows, p_cols = self.get_pat(img,
                                                               row_range,
                                                               col_range)

            # translate coords from img system to pat system
            p_center = ((self.center[0] - row_range[0]),
                        (self.center[1] - col_range[0]))

            rgb = self.fill_rgbo[0:3]
            cv.circle(pat,
                      p_center,
                      self.radius,
                      rgb[::-1],
                      thickness=-1,
                      lineType=cv.LINE_AA)
            img = self.composite(img,
                                 pat,
                                 s_rows,
                                 s_cols,
                                 p_rows,
                                 p_cols,
                                 self.fill_rgbo[3])

        if self.cntr_rgb:
            rgb = self.cntr_rgb[0:3]
            cv.circle(img,
                      (self.center[1], self.center[0]),
                      self.radius,
                      rgb[::-1],
                      thickness=self.cntr_thick,
                      lineType=cv.LINE_AA)

        return img


class OpencvFont(object):
    def __init__(self, face: int = cv.FONT_HERSHEY_SIMPLEX,
                       scale: float = 1.0,
                       thick: int = 1,
                       linetype: int = cv.LINE_AA) -> None:
        self.face = face
        self.scale = scale
        self.thick = thick
        self.linetype = linetype

    def text_size(self, text: str) -> Union[tuple, None]:
        if text is not None:
            bb, _ = cv.getTextSize(text,
                                   self.face,
                                   self.scale,
                                   self.thick)
            return ((bb[1] + self.thick), bb[0])
        else:
            return None


class OpencvText(OpencvShapeCompositor):
    def __init__(self, text: str,
                       rgb: tuple,
                       font: OpencvFont = OpencvFont(),
                       tl: tuple = None,
                       br: tuple = None,
                       justify: Tuple[str, int] = ('left', 0),
                       bl: tuple = None) -> None:
        super(OpencvText, self).__init__()
        self.text = text
        self.rgb = rgb
        self.font = font
        self.tl = tl
        self.br = br
        self.justify = justify
        self.bl = bl
        status, msg = self._check_coord_specs()
        if not status:
            raise Exception(msg)

    def _check_coord_specs(self) -> Tuple[bool, str]:
        if self.tl and self.br and self.bl:
            return (False, "Only one of (tl, br) or bl must be specified.")
        return (True, "")

    def _get_bl(self) -> tuple:
        if self.bl:
            return self.bl
        else:
            # bounding box for text
            tsz = self.font.text_size(self.text)
            # bb half size
            tszmidr, tszmidc = tsz[0]//2, tsz[1]//2
            # center of specified tl, br
            cr = (self.tl[0] + self.br[0])//2
            cc = (self.tl[1] + self.br[1])//2
            if self.justify[0] == "left":
                return ((cr + tszmidr), self.tl[1] + self.justify[1])
            elif self.justify[0] == "right":
                return ((cr + tszmidr), self.br[1] - tsz[1] - self.justify[1])
            else:
                return((cr + tszmidr), (cc - tszmidc))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        cv.putText(img,
                   self.text,
                   self._get_bl()[::-1],
                   self.font.face,
                   self.font.scale,
                   self.rgb[::-1],
                   self.font.thick,
                   self.font.linetype)
        return img
