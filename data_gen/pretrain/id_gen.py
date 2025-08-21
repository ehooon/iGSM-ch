# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math_gen.problem_gen import Problem
from data_gen.prototype.id_gen import IdGen_PT
from const.params import USE_MOD # <-- 1. 导入 USE_MOD 开关

class IdGen(IdGen_PT):
    # vvvvvv 2. 修改 __init__ 方法的签名，添加 use_mod 参数 vvvvvv
    def __init__(self, max_op=10, max_edge=15, op=None, perm_level: str = None, detail_level: str = None, be_shortest: bool=True, use_mod: bool = USE_MOD) -> None:
        # vvvvvv 3. 在调用 super() 时，将 use_mod 参数传递给父类 vvvvvv
        super().__init__('light', 'light', max_op, max_edge, op, perm_level, detail_level, be_shortest, use_mod=use_mod)
    
    def gen_prob(self, ava_hash, p_format: str, problem: Problem=None):
        super().gen_prob(ava_hash, p_format, problem=problem)