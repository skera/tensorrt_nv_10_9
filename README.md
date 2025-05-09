# tensorrt_nv_10_9
nv官方tensorrt 10.9

# TensorRT 10.9 Download
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz

# 编译方法
1. tar zxvf TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz
2. git clone ThisRepo
3. cd tensorrt_nv_10_9/samples
4. cp -rf /xxx/xxx/TensorRT-10.9.0.34/samples/common ./
5. cp -rf /xxx/xxx/TensorRT-10.9.0.34/samples/utils ./
6. make TensorRT_INSTALL_DIR=/xxx/xxx/TensorRT-10.9.0.34
# 执行
cd TensorRT-10.9.0.34 <br/>
env LD_LIBRARY_PATH=./lib: ./bin/sample_demo -d ./model/ --modelFileName model.onnx --dim=1000 --batch=150 <br/>