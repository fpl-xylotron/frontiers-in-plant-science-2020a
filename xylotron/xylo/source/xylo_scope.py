from typing import List, Union, Tuple, Dict, Any, Sequence
import sys
from pathlib import Path
import json
from datetime import datetime
from collections import OrderedDict
import numpy as np
import cv2 as cv
import PySpin
from xylo_parsers import *


class XyloScope(object):
    def __init__(self, config_dir) -> None:
        self.system = None
        self.camlist = None
        self.cam = None
        self.nodemaps = None
        self.typecasts = None
        self.param2nmap = None
        self.param2dtype = None
        self.param2value = None

        self.serialnum = None
        self.config_dir = config_dir
        self.config = None
        self.capturing = False

    def wanted_config(self) -> List[tuple]:
        return self.config

    def software_version(self) -> str:
        v = self.system.GetLibraryVersion()
        return f"Spinnaker: {v.major}.{v.minor}.{v.type}.{v.build}"

    def serial_number(self) -> str:
        return self.read_param("DeviceSerialNumber")[1]

    def configure_camera(self) -> None:
        config_file = self.config_dir/f"{self.serialnum}_config.txt"
        self.config = ConfigFileParser(config_file, self.param2dtype)()
        for (k, v) in self.config:
            (status, m) = self.write_param(k, v)
            if not status:
                print("\n".join(m))

    def start_camera(self) -> bool:
        self.system = None
        self.system = PySpin.System.GetInstance()
        if self.system is None:
            print("Unable to get system.")
            self.cleanup()
            return False

        self.camlist = None
        self.camlist = self.system.GetCameras()
        if self.camlist is None:
            print("Unable to get cameras list.")
            self.cleanup()
            return False

        self.cam = None
        camidx = 0
        if camidx >= 0 and camidx < self.camlist.GetSize():
            self.cam = list(self.camlist)[camidx]
            self.cam.Init()
        else:
            print("Camera not found. Did you connect a camera?")
            self.cleanup()
            return False

        self.nodemaps = {'std': self.cam.GetNodeMap(),
                         'tld': self.cam.GetTLDeviceNodeMap(),
                         'tls': self.cam.GetTLStreamNodeMap()}

        self.typecasts = {"Enumeration": PySpin.CEnumerationPtr,
                          "Integer": PySpin.CIntegerPtr,
                          "Float": PySpin.CFloatPtr,
                          "Boolean": PySpin.CBooleanPtr,
                          "String": PySpin.CStringPtr}

        node_desc_file = self.config_dir/'nodes_description.json'
        node_desc = NodesDescriptionParser(node_desc_file)()
        self.param2nmap, self.param2dtype = OrderedDict(), OrderedDict()
        for k, v in node_desc.items():
            self.param2nmap[k] = v[0]
            self.param2dtype[k] = v[1]

        (status, value) = self.read_param("DeviceSerialNumber")
        if not status:
            print("\n".join(value))
            self.cleanup()
            return False
        self.serialnum = value

        (status, value) = self.read_param("DeviceCurrentSpeed")
        if not status:
            print("\n".join(value))
            self.cleanup()
            return False
        if value != "SuperSpeed":
            print("Camera not on USB3.")
            print("  (1) Disconnect and reconnect or (2) Use different port.")
            self.cleanup()
            return False

        return True

    def write_param(self, name: str,
                          value: Union[str, float, int, bool]) -> Tuple[bool, list]:
        if name not in self.param2nmap:
            m = [f"Trying to read unspecified parameter: {name}"]
            return (False, m)

        d = self._get_node(name)
        m = self._check_node_for(d, readable=True, writable=True)
        if len(m) > 0:
            return (False, m)

        if isinstance(d['node'], PySpin.CEnumerationPtr):
            p = PySpin.CEnumEntryPtr(d['node'].GetEntryByName(value))
            d['node'].SetIntValue(p.GetValue())
        else:
            d['node'].SetValue(value)

        return (True, [])

    def read_param(self, name: str) -> Tuple[bool, list]:
        if name not in self.param2nmap:
            m = [f"Trying to read unspecified parameter: {name}"]
            return (False, m)

        d = self._get_node(name)
        m = self._check_node_for(d, readable=True)
        if len(m) > 0:
            return (False, m)

        if isinstance(d['node'], PySpin.CEnumerationPtr):
            value = d['node'].ToString()
        else:
            value = d['node'].GetValue()
        return (True, value)

    def get_image(self) -> np.ndarray:
        image = self.cam.GetNextImage()
        retimg = image.GetNDArray()
        image.Release()
        return retimg

    def start_capturing(self) -> None:
        self.cam.BeginAcquisition()
        self.capturing = True

    def stop_capturing(self) -> None:
        self.cam.EndAcquisition()
        self.capturing = False

    def is_capturing(self) -> None:
        return self.capturing
        
    def _get_node(self, name: str) -> OrderedDict:
        d = OrderedDict()
        nmap = self.nodemaps[self.param2nmap[name]]
        node = self.typecasts[self.param2dtype[name]](nmap.GetNode(name))
        d['name'] = name
        d['node'] = node
        d['available'] = PySpin.IsAvailable(node)
        d['readable'] = PySpin.IsReadable(node)
        d['writable'] = PySpin.IsWritable(node)
        return d

    def _check_node_for(self, d: dict,
                              available: bool = True,
                              readable: bool = False,
                              writable: bool = False) -> list:
        message = []
        if available and not d['available']:
            message += [f"\t{d['name']} is not available"]
        if readable and not d['readable']:
            message += [f"\t{d['name']} is not readable"]
        if writable and not d['writable']:
            message += [f"\t{d['name']} is not writable"]
        if len(message) > 0:
            message = ["Invalid parameter access: "] + message
        return message

    def cleanup(self) -> None:
        self.nodemaps = None
        if 'cam' in self.__dir__() and self.cam:
            self.cam.DeInit()
            del self.cam
        if 'camlist' in self.__dir__() and self.camlist:
            self.camlist.Clear()
        if 'system' in self.__dir__() and self.system:
            self.system.ReleaseInstance()

    def __del__(self) -> None:
        self.cleanup()
