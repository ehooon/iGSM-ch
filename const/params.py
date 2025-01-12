# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import string

mod = 23
dot = "'s "
try_num = 1000
retry_key_word = "BACK"

train_bin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
test_bin = [16, 17, 18, 19, 20, 21, 22]
lora_train_bin = train_bin
lora_test_bin = test_bin
all_bin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

feasible_symbols = list(string.ascii_lowercase[:26]) + list(string.ascii_uppercase[:26])
