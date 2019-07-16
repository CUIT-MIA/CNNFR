set -e

TOOLS=/usr/local/caffe-master/build/tools
LOG=*your_log_dir*-`date +%Y-%m-%d-%H-%M-%S`.log
$TOOLS/caffe train --solver='path-to-cnnfrnet_train_test.prototxt' --gpu=1 2>&1 | tee $LOG $@
