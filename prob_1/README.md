# tensorrt_nv_10_9
nv官方tensorrt 10.9

# TensorRT 10.9 Download
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz

# 编译方法
1. tar zxvf TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz
2. export TensorRT_INSTALL_DIR=/data/app/cuda/TensorRT-10.9.0.34
3.  
2. git clone https://github.com/skera/tensorrt_nv_10_9.git
3. cd tensorrt_nv_10_9/samples
4. cp -rf ${TensorRT_INSTALL_DIR}/samples/common ./
5. cp -rf ${TensorRT_INSTALL_DIR}/samples/utils ./
6. make TensorRT_INSTALL_DIR=${TensorRT_INSTALL_DIR}
# 执行
cd tensorrt_nv_10_9 <br/>
1. env LD_LIBRARY_PATH=${TensorRT_INSTALL_DIR}/lib:$LD_LIBRARY_PATH ./bin/sample_demo -d ./model/ --modelFileName model.onnx --dim=30000 --batch=10 <br/>