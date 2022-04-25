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

model_path=$1
query_datapath=$2
gallery_datapath=$3
echo "-----------------------------------"
echo "mindir name: "$model_path
echo "query dataset path: "$query_datapath
echo "gallery dataset path: "$gallery_datapath
echo "-----------------------------------"

export ASCEND_HOME=/usr/local/Ascend
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

function compile_app()
{
    cd ascend310_infer || exit
    bash build.sh &> build.log
}

function query_infer()
{
    if [ -d query_result_Files ]; then
        rm -rf ./query_result_Files
    fi
    mkdir query_result_Files
    input_type="query"
    ../ascend310_infer/out/main --model_path=$model_path --dataset_path=$query_datapath --input_type=$input_type &> query_infer.log
}

function gallery_infer()
{
    cd ../ascend310_infer || exit
    if [ -d gallery_result_Files ]; then
        rm -rf ./gallery_result_Files
    fi
    mkdir gallery_result_Files
    input_type="gallery"
    ../ascend310_infer/out/main --model_path=$model_path --dataset_path=$gallery_datapath --input_type=$input_type &> gallery_infer.log
}

function cal_acc()
{
    cd ..
    qf=ascend310_infer/query_result_Files/feature_data.txt
    qp=ascend310_infer/query_result_Files/savepid.txt
    qc=ascend310_infer/query_result_Files/savecamid.txt
    gf=ascend310_infer/gallery_result_Files/feature_data.txt
    gp=ascend310_infer/gallery_result_Files/savepid.txt
    gc=ascend310_infer/gallery_result_Files/savecamid.txt
    python postprocess.py --q_feature=$qf --q_pid=$qp --q_camid=$qc --g_feature=$gf --g_pid=$gp --g_camid=$gc &> acc.log
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
echo "compile app code success"

query_infer
if [ $? -ne 0 ]; then
    echo "execute query inference failed"
    exit 1
fi
echo "execute query inference success"

gallery_infer
if [ $? -ne 0 ]; then
    echo "execute gallery inference failed"
    exit 1
fi
echo "execute gallery inference success"

cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
echo "calculate accuracy success"
echo "ascend 310 infer success"
