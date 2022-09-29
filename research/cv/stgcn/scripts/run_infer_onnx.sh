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

if [ $# != 5 ]
then
    echo "Usage: bash scripts/run_eval_onnx.sh [DATA_PATH] [ONNX_PATH] [DEVICE_TARGET] [N_PRED] [GRAPH_CONV_TYPE]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
data_path=$(get_real_path $1)
onnx_path=$(get_real_path $2)
device_target=$3
n_pred=$4
graph_conv_type=$5



echo "ONNX name: "$onnx_path
echo "dataset path: "$data_path
echo "device_target: "$device_target
echo "n_pred: "$n_pred
echo "graph_conv_type: "$graph_conv_type

function infer() {
  python eval_onnx.py --data_url=$data_path \
                --device_target=$device_target  \
                --train_url=./checkpoint \
                --run_distribute=False \
                --run_modelarts=False \
                --onnx_path=$onnx_path \
                --n_pred=$N_PRED \
                --graph_conv_type=$GRAPH_CONV_TYPE > eval_onnx.log 2>&1 &
}


infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
