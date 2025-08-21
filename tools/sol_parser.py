# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math_gen.problem_gen import Problem, Num
from collections import defaultdict as dd
from tools.tools import MyPrint, idle_func
from typing import List, Dict, Set
from const.params import mod, USE_MOD # <-- 1. 导入 USE_MOD 开关和 mod 常量

# vvvvvv 2. 修改 is_num 函数，让其行为依赖 use_mod 开关 vvvvvv
def is_num(name: str, use_mod=USE_MOD):
    """
    Checks if a string represents a valid number based on the current math mode.
    """
    if not name.isdigit():
        return False
    
    if use_mod:
        # In modulo mode, the number must be within the valid range [0, mod-1]
        return 0 <= int(name) < mod
    else:
        # In standard integer mode, any sequence of digits is a valid number
        return True
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class Sentence(object):
    def __init__(self, sentence="", def_part=None, param_part=None, hint_part=[], parent_part=[], sign=None, cal_part: List[Num]=[], ans_part=Num(0), idx=None) -> None:
        self.sentence = sentence
        self.def_part = def_part
        self.param_part = param_part
        self.hint_part = hint_part
        self.parent_part = parent_part
        self.sign = sign
        self.cal_part = cal_part
        self.ans_part = ans_part
        self.idx = idx
    
    def display(self):
        print(f"\n\ninfo of sentence {self.idx}")
        print(f"ntn: {self.def_part}")
        print(f"param: {self.param_part}")
        print(f"hint: {self.hint_part}")
        print(f"parents: {self.parent_part}")
        print(f"sign: {self.sign}")
        print(f"calcu: {[num.a for num in self.cal_part]}")
        print(f"ans: {self.ans_part}")
        print("\n")


class Parser(object):
    # vvvvvv 3. 修改 __init__ 方法，接收并保存 use_mod 状态 vvvvvv
    def __init__(self, gpt_sol: str, detail: str="000", if_print=False, use_mod=USE_MOD) -> None:
        self.use_mod = use_mod # <-- 保存状态，供后续方法使用
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.wrong_ans_param = False
        self.retry_count = 0
        self.def_count = 0
        self.real_def_count = 0
        if gpt_sol.startswith(" "):
            gpt_sol = gpt_sol[1:]
        if gpt_sol.endswith("."):
            gpt_sol = gpt_sol[:-1]
        self.gpt_sol = gpt_sol
        self.sol_steps = []
        gpt_steps = gpt_sol.split(". ")
        for gpt_step in gpt_steps:
            if gpt_step.startswith("Define"):
                self.def_count += 1
            if ' BACK' in gpt_step:
                self.retry_count += 1
                continue
            if gpt_step.startswith("Define"):
                self.real_def_count += 1
            gpt_small_steps = gpt_step.split("; ")
            num_steps = len(gpt_small_steps)
            for i in range(1, num_steps-1):
                self.sol_steps.append(gpt_small_steps[i])
            if gpt_small_steps: # Ensure list is not empty
                last_step_base = gpt_small_steps[0]
                if num_steps > 1:
                    last_step_base += gpt_small_steps[-1][4:]
                self.sol_steps.append(last_step_base)

        self.sol_op = len(self.sol_steps)
        self.def_rule = detail[0]
        self.hint_rule = detail[1]
        self.cal_rule = detail[2]
        self.if_print = if_print
        self.sentence_lst: List[Sentence] = []
        self.param_dict: Dict[str, Sentence] = {}
        self.symbol_dict: Dict[str, Sentence] = {} # symbol
        self.symbol_dep_map: Dict[str, Set[str]] = {}
        self.early_stop_param = None
        self.parsed = True

        self.duplicated_symbol = False
        self.unknown_symbol = False
        self.hint_cal_not_match = False
        self.illegal_def_part = False

        for i, sol_step in enumerate(self.sol_steps):
            try:
                if self.if_print:
                    print(f"parse: {sol_step}")
                self.parse_sentence(sol_step=sol_step, idx=i)
            except (NotImplementedError, IndexError, ValueError) as e:
                if self.if_print:
                    print(f"break in parser sol step:")
                    print(sol_step)
                self.parsed = False
                return
        
        for symbol in self.symbol_dict.keys():
            self.symbol_dep_map[symbol] = self.find_dep_set(param=symbol)
        
        if not self.sentence_lst:
            if self.if_print:
                print(f"not a single valid sentence is found")
            self.parsed = False
            return

    def parse_sentence(self, sol_step:str, idx=None):
        def_part = ""
        param_part = ""
        hint_part = []
        parent_part = []
        sign = None
        cal_part = []
        ans_part = Num(0)

        part_lst = sol_step.split(" = ")

        part = part_lst.pop(0)
        got_ntn = False
        while True:
            if not got_ntn:
                if part.startswith("Define"):
                    part = part[7:]
                    part_ = part.split(" as ")
                    def_part = part_[1] if len(part_) > 1 else part_[0]
                    param_part = part_[0] if len(part_) > 1 else ""
                else:
                    def_part = part
                
                if len(def_part) != 1:
                    self.illegal_def_part = True
                    self._illegal_def_part = def_part
                    if self.if_print:
                        print(f"Illegal def part {self._illegal_def_part}")
                    raise NotImplementedError
                got_ntn = True
                if not part_lst: break
                part = part_lst.pop(0)
                continue

            exp_parsed = False
            for op_sign, op_name in zip([" + ", " - ", " * "], ["add", "sub", "mul"]):
                if op_sign in part:
                    part_ = part.split(op_sign)
                    
                    # vvvvvv 4. 在调用 is_num 的地方传递 self.use_mod vvvvvv
                    if is_num(part_[0], use_mod=self.use_mod) and is_num(part_[1], use_mod=self.use_mod):
                        for num_str in part_:
                            # vvvvvv 5. 创建 Num 对象时传递 use_mod vvvvvv
                            cal_part.append(Num(num_str, use_mod=self.use_mod))
                    else:
                        for item in part_:
                            hint_part.append(item)
                            if not is_num(item, use_mod=self.use_mod):
                                parent_part.append(item)
                    
                    sign = op_name
                    exp_parsed = True
                    break
            if exp_parsed:
                if not part_lst: break
                part = part_lst.pop(0)
                continue
            
            if not is_num(part, use_mod=self.use_mod): # <-- 4. 再次传递 self.use_mod
                hint_part.append(part)
                parent_part.append(part)
                if not part_lst: break
                part = part_lst.pop(0)
                continue
            
            if (len(hint_part) > 0 and len(cal_part) > 0) and len(hint_part) != len(cal_part):
                self.hint_cal_not_match = True
                self._hint_cal_not_match = (hint_part, cal_part)
                if self.if_print:
                    print(f"Hint ({self._hint_cal_not_match[0]}) does not match {self._hint_cal_not_match[1]}")
                raise NotImplementedError

            ans_part = Num(part, use_mod=self.use_mod) # <-- 5. 再次传递 use_mod
            break # Parsing is complete for this sentence

        sentence = Sentence(
            sentence=sol_step,
            def_part=def_part,
            param_part=param_part,
            hint_part=hint_part,
            parent_part=parent_part,
            sign=sign,
            cal_part=cal_part,
            ans_part=ans_part,
            idx=idx,
        )

        if idx is not None:
            self.sentence_lst.append(sentence)
            for parent in parent_part:
                if parent not in self.symbol_dict:
                    self.unknown_symbol = True
                    self._unknown_symbol = parent
                    if self.if_print:
                        print(f"Undefined symbol: {self._unknown_symbol}")
                    raise NotImplementedError
            if def_part in self.symbol_dict:
                self.duplicated_symbol = True
                self._duplicated_symbol = def_part
                self.parsed = False
                if self.if_print:
                    print(f"Duplicated symbol: {self._duplicated_symbol}")
                raise NotImplementedError
            self.symbol_dict[def_part] = sentence
            if param_part and param_part not in self.param_dict:
                self.param_dict[param_part] = sentence

    def find_dep_set(self, param: str) -> set:
        dep_set = set()
        sentence = self.symbol_dict.get(param) or self.param_dict.get(param)
        if not sentence: return dep_set
        
        ntn_list = list(sentence.parent_part)
        history_list = set()

        while ntn_list:
            ntn = ntn_list.pop()
            if ntn in history_list:
                continue
            history_list.add(ntn)
            
            sentence_ = self.symbol_dict.get(ntn)
            if sentence_:
                if sentence_.param_part:
                    dep_set.add(sentence_.def_part)
                else:
                    ntn_list.extend(p for p in sentence_.parent_part if p not in history_list)
        
        return dep_set

    def correct_refer(self, my_print: MyPrint=idle_func):
        wrong_refer = 0
        self.lookup = dd(set)
        for sentence in self.sentence_lst:
            if len(sentence.hint_part) == 1:
                num = sentence.ans_part.a
                if num not in self.lookup[sentence.hint_part[0]]:
                    my_print(f"{sentence.hint_part[0]} = {num} not in {self.lookup[sentence.hint_part[0]]}")
                    wrong_refer += 1
            else:
                for i, parent in enumerate(sentence.hint_part):
                    if i < len(sentence.cal_part):
                        num = sentence.cal_part[i].a
                        if not is_num(parent, use_mod=self.use_mod) and num not in self.lookup[parent]: # <-- 4. 再次传递
                            my_print(f"{parent} = {num} not in {self.lookup[parent]}")
                            wrong_refer += 1

            self.lookup[sentence.def_part].add(sentence.ans_part.a)
        return wrong_refer, my_print


    def correct_cal(self, my_print: MyPrint=idle_func):
        wrong_cal = 0
        for sentence in self.sentence_lst:
            if sentence.cal_part and len(sentence.cal_part) == 2:
                # vvvvvv 5. 确保这里的 Num 对象也使用正确的模式 vvvvvv
                if sentence.sign == "add":
                    res = sentence.cal_part[0] + Num(sentence.cal_part[1].a, use_mod=self.use_mod)
                    if res != sentence.ans_part:
                        my_print(f"in {sentence.sentence}: {sentence.cal_part[0].a} + {sentence.cal_part[1].a} != {sentence.ans_part.a}")
                        wrong_cal += 1
                elif sentence.sign == "sub":
                    res = sentence.cal_part[0] - Num(sentence.cal_part[1].a, use_mod=self.use_mod)
                    if res != sentence.ans_part:
                        my_print(f"in {sentence.sentence}: {sentence.cal_part[0].a} - {sentence.cal_part[1].a} != {sentence.ans_part.a}")
                        wrong_cal += 1
                elif sentence.sign == "mul":
                    res = sentence.cal_part[0] * Num(sentence.cal_part[1].a, use_mod=self.use_mod)
                    if res != sentence.ans_part:
                        my_print(f"in {sentence.sentence}: {sentence.cal_part[0].a} * {sentence.cal_part[1].a} != {sentence.ans_part.a}")
                        wrong_cal += 1
        return wrong_cal, my_print

    def correct_order(self, my_print: MyPrint=idle_func):
        self.sol_order = []
        re_define = 0
        wrong_order = 0
        defined_symbols = set()
        for sentence in self.sentence_lst:
            if sentence.def_part in defined_symbols:
                my_print(f"{sentence.def_part} in {sentence.sentence} has already been defined.")
                re_define += 1
            else:
                defined_symbols.add(sentence.def_part)
            
            for parent in sentence.parent_part:
                if parent not in defined_symbols:
                    my_print(f"{parent} used in {sentence.sentence} has not been defined yet.")
                    wrong_order += 1
            self.sol_order.append(sentence.def_part)
        
        return re_define, wrong_order, my_print

    def parse(self, problem: Problem):
        self.non_appear_lst = []
        self.non_nece_lst = []
        self.incorrect_lst = [] 
        self.param_name_lst = []

        for sentence in self.sentence_lst:
            if sentence.param_part:
                duplicate = sentence.param_part in self.param_name_lst
                if not duplicate:
                    self.param_name_lst.append(sentence.param_part)
                
                param_name = sentence.param_part
                param = problem.name2param(param_name=param_name)

                if param is None:
                    if not self.non_appear_lst: self.non_appear_lst.append(param_name)
                    continue 
                if not duplicate and param not in problem.topological_order:
                    self.non_nece_lst.append(param)
                
                pre_params = [p for p in problem.whole_template.predecessors(param) if p != (-1, 0, 0, 0)]
                gpt_params_symbols = self.symbol_dep_map.get(sentence.def_part, set())
                gpt_params = {problem.name2param(self.symbol_dict[s].param_part) for s in gpt_params_symbols if self.symbol_dict[s].param_part}
                
                missing = [p for p in pre_params if p not in gpt_params]
                extra = [p for p in gpt_params if p not in pre_params]

                if missing or extra:
                    self.incorrect_lst.append((param, missing, list(extra)))