# tensorrt_nv_10_9
nv官方tensorrt 10.9

# TensorRT 10.9 Download
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz

# 编译方法
1. tar zxvf TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz
2. 将samples目录下文件，覆盖TensorRT-10.9.0.34/samples/
3. cd TensorRT-10.9.0.34/samples & make -j
# 执行
cd TensorRT-10.9.0.34 <br/>
env LD_LIBRARY_PATH=./lib: ./bin/sample_demo -d ./model/ --modelFileName model.onnx --dim=1000 --batch=150 <br/>