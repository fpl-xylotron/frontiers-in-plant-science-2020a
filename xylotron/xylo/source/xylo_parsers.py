from typing import List, Union, Tuple, Dict, Any, Sequence
import sys
from pathlib import Path
from collections import OrderedDict
import json

class NodesDescriptionParser(object):
    def __init__(self, f: Union[str, Path]) -> None:
        self.f = Path(f)

    def __call__(self) -> OrderedDict:
        return json.load(open(self.f), object_pairs_hook=OrderedDict)


class ConfigFileParser(object):
    def __init__(self, f: Union[str, Path],
                       param2dtype: OrderedDict) -> None:
        self.f = Path(f)
        self.p2d = param2dtype
        self.cast = {"Integer": int,
                     "Enumeration": str,
                     "String": str,
                     "Float": float,
                     "Boolean": bool}

    def __call__(self) -> List[tuple]:
        cfg = []
        with open(self.f, 'r') as fh:
            lines = [line.strip() for line in fh if line.strip()]
        for line in lines:
            tkns = [t.strip() for t in line.split('=') if t.strip()]
            dtype = self.p2d[tkns[0]]
            cfg.append((tkns[0].strip(), self.cast[dtype](tkns[1].strip())))
        return cfg
