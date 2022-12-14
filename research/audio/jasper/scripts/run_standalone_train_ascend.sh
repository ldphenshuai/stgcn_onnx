#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_standalone_train_ascend.sh [DEVICE_ID]"
exit 1
fi

log_dir="./log"

if [ ! -d $log_dir ]; then
    mkdir log
fi

DEVICE_ID=$1

python train.py --device_target 'Ascend' --device_id $DEVICE_ID > log/train_standalone.log 2>&1 &
