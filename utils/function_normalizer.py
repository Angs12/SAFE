# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import numpy as np


_CLONE_RE = re.compile(
    r'\.('
    r'isra\.\d+|constprop\.\d+|lto_priv\.\d+|_omp_fn\.\d+|'
    r'eh\.\d+|resxl\.\d+|localalias\.\d+|'
    r'specialized\.\d+|thinlto\.\d+|fulllto\.\d+|'
    r'argprom|argelim|retelim'
    r')$'
)


def strip_clone_suffix(name):
    while True:
        new_name = _CLONE_RE.sub('', name)
        if new_name == name:
            break
        name = new_name
    return name


class FunctionNormalizer:
    def __init__(self, max_instruction):
        self.max_instructions = max_instruction

    def normalize(self, f):
        f = np.asarray(f[0 : self.max_instructions])
        length = f.shape[0]
        if f.shape[0] < self.max_instructions:
            f = np.pad(f, (0, self.max_instructions - f.shape[0]), mode="constant")
        return f, length

    def normalize_functions(self, functions):
        lengths = []
        new_functions = []
        for f in functions:
            f, length = self.normalize(f)
            lengths.append(length)
            new_functions.append(f)
        return new_functions, lengths
