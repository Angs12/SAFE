# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import r2pipe
import json
import sys


class BinaryAnalyzer:
    def __init__(self, path):
        self.r2 = r2pipe.open(path, flags=["-2"])
        self.arch = None
        self.bits = None
        self._sizes = {}
        self._functions = []
        self._fallback = False
        try:
            info = json.loads(self.r2.cmd("ij"))["bin"]
            self.arch = info["arch"]
            self.bits = info["bits"]
        except:
            print(f"Error loading file: {path}", file=sys.stderr)
        try:
            syms = self.r2.cmdj("isj")
        except:
            syms = []
        for s in syms:
            if s.get("type") != "FUNC":
                continue
            name = s.get("name")
            addr = s.get("vaddr")
            size = s.get("size", 0)
            if name is None or addr is None or size == 0:
                continue
            self._functions.append((name, addr))
            self._sizes[addr] = size
        if not self._functions:
            self._fallback = True
            print(f"  [no symbols, scanning preludes on {path}]", file=sys.stderr)
            self.r2.cmd("aap")
            try:
                afl = self.r2.cmdj("aflj")
            except:
                afl = []
            for f in afl:
                name = f.get("name")
                addr = f.get("offset") or f.get("addr")
                size = f.get("size", 0)
                if name is None:
                    continue
                if not addr and name.startswith("fcn."):
                    try:
                        addr = int(name.split(".")[1], 16)
                    except:
                        continue
                if not addr:
                    continue
                self._functions.append((name, addr))
                if size:
                    self._sizes[addr] = size

    def close(self):
        try:
            self.r2.quit()
        except:
            pass

    def __del__(self):
        self.close()

    def get_hexasm(self, address):
        size = self._sizes.get(address)
        if size:
            return self.r2.cmd(f"p8 {size} @ {address}").strip()
        data = filter(None, self.r2.cmd(f"pxf @ {address}").split("\n")[1:])
        hexasm = ""
        for i in data:
            hexasm += "".join(i.split("  ")[1].split())
        return hexasm

    def get_functions(self):
        return self._functions

    @property
    def used_fallback(self):
        return self._fallback
