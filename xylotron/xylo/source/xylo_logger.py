from typing import List, Union, Tuple, Dict, Any, Sequence
import sys
from pathlib import Path
import json
from datetime import datetime
from xylo_scope import *

class ConfigStringMaker(object):
    def __init__(self, xs: XyloScope) -> None:
        self.xs = xs

    def string(self) -> None:
        details = []
        details.append(self.xs.software_version())
        sn = self.xs.read_param("DeviceSerialNumber")[1]
        details.append(f"CameraSerialNumber: {sn}")

        prev = ""
        for (k, v) in self.xs.wanted_config():
            if k in ["BalanceRatioSelector"]:
                self.xs.write_param(name=k, value=v)
                prev = f"({v})"
            else:
                qv = self.xs.read_param(name=k)[1]
                details.append(f"{k}{prev}: wanted = {v}, queried = {qv}")
                prev = ""
        return "\n".join(details)


class XyloLogger(object):
    def __init__(self, top_dir: Union[str, Path],
                       camera_serial_number: str = "00000000",
                       user_name: Union[str, None] = None) -> None:
        self.topdir = Path(top_dir)
        usrname = user_name
        if usrname is None:
            usrname = "user"
        self.suffix = f"{camera_serial_number}__{usrname}"
        # create directories with time stamp of session start
        now = datetime.now()
        yr = now.year
        mt = str(now.month).zfill(2)
        dy = str(now.day).zfill(2)
        sess = "__".join([f"{yr}-{mt}-{dy}", self.suffix])
        self.imgdir = self.topdir/"images"/sess
        self.logdir = self.topdir/"logs"/sess
        self.imgdir.mkdir(parents=True, exist_ok=True)
        self.logdir.mkdir(parents=True, exist_ok=True)

    def save(self, img: np.ndarray,
                   log: str,
                   genus: str = "Genus",
                   species: str = "species",
                   collection: str = "UNK",
                   specimen: str = "0") -> None:
        # filename
        part1 = f"{genus}_{species}_{collection}_{str(specimen).zfill(9)}"
        now = datetime.now()
        yr = now.year
        mt = str(now.month).zfill(2)
        dy = str(now.day).zfill(2)
        hr = str(now.hour).zfill(2)
        mn = str(now.minute).zfill(2)
        sc = str(now.second).zfill(2)
        part2 = f"{yr}-{mt}-{dy}_{hr}-{mn}-{sc}"
        part3 = self.suffix
        filename = "__".join([part1, part2, part3])
        taxa_dir = "_".join([genus, species])

        # image path
        taxaimg_dir = self.imgdir/taxa_dir
        taxaimg_dir.mkdir(parents=True, exist_ok=True)
        imgpath = taxaimg_dir/f"{filename}.png"
        cv.imwrite(str(imgpath), img, [cv.IMWRITE_PNG_COMPRESSION, 0])

        # log path
        taxalog_dir = self.logdir/taxa_dir
        taxalog_dir.mkdir(parents=True, exist_ok=True)
        logpath = taxalog_dir/f"{filename}.txt"
        with open(logpath, 'w') as fh:
            fh.write(log)
