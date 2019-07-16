# if your need transform learning ,using this script

set -e

TOOLS=/usr/local/caffe-master/build/tools
LOG=/home/yangxiaodong/matchnet/models/yxdLOG/-`date +%Y-%m-%d-%H-%M-%S`.log
$TOOLS/caffe train --solver='path-to-cnnfrnet_train_test.prototxt' --weights='your-path-weight' --gpu=1 2>&1 | tee $LOG $@
