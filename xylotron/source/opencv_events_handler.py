from typing import List, Union, Tuple, Dict, Any, Sequence
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as cv

dflt_mouse = [cv.EVENT_LBUTTONDOWN]

class OpencvEventsHandler(object):
    def __init__(self, wanted_mouse: List = dflt_mouse) -> None:
        self.keystroke = None
        self.mouseevent = None
        self.mousexy = None
        self.wanted_mouse = wanted_mouse

    def _log_keystroke(self, k: int) -> None:
        skey = k & 0xFF
        if skey in range(97, 123) or skey in range(48, 58):
            self.keystroke = chr(skey)

    def log_keystroke(self, k: int) -> None:
        self._log_keystroke(k)

    def _log_mouseevent(self, e: int,
                              x: int,
                              y: int) -> None:
        if e in self.wanted_mouse:
            self.mouseevent = e
            self.mousexy = (x, y)

    def log_mouseevent(self, e: int,
                             x: int,
                             y: int) -> None:
        self._log_mouseevent(e, x, y)

    def reset_keystroke(self) -> None:
        self.keystroke = None

    def reset_mouseevent(self) -> None:
        self.mouseevent = None
        self.mousexy = None

    def is_keystroke(self, k: str) -> bool:
        return k == self.keystroke

    def is_mouseevent(self, m: str) -> bool:
        return m == self.mouseevent

    def get_keystroke(self) -> str:
        return self.keystroke

    def get_mousexy(self) -> tuple:
        return self.mousexy

    def print(self) -> None:
        print(f"Key: {self.keystroke}, Event: {self.mouseevent}, Pos: {self.mousexy}")


def mouse_cb(event: int,
             x: int,
             y: int,
             flag: int,
             obj: OpencvEventsHandler) -> None:
    obj.log_mouseevent(e=event, x=x, y=y)
