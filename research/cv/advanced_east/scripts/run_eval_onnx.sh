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

if [ $# -ne 3 ]; then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_onnx_eval.sh DATA_PATH DEVICE_TYPE ONNX_MODEL_PATH"
    echo "for example: bash scripts/run_onnx_eval.sh /path/icpr GPU /path/AdvancedEast.onnx "
    echo "=============================================================================================================="
    exit 1
fi

DATA_PATH=$1
DEVICE_TARGET=$2
ONNX_MODEL_PATH=$3

python eval_onnx.py \
    --data_dir=$DATA_PATH \
    --device_target=$DEVICE_TARGET \
    --onnx_path=$ONNX_MODEL_PATH > eval_onnx.log 2>&1 &
