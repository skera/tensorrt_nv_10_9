/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <random>
#include <iostream>
#include <chrono>

class PerfGuard {
public:
    // 构造函数，记录开始时间
    PerfGuard(const std::string& name, size_t count = 0)
        : name_(name), count_(count), start_(std::chrono::high_resolution_clock::now()) {}

    // 析构函数，记录结束时间并计算和输出执行时间
    ~PerfGuard() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    std::cout << name_
        << " ------ cnt[" << count_ << "]"
        <<  " tm[" << duration <<  "us]"
        << " avg[" << duration / count_ << "us]" << std::endl;
    }

private:
    std::string name_;
    size_t count_ = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_demo";

template <typename T>
std::ostream& operator<<(std::ostream& os,  std::vector<T>& v) {
    os << "{";
    for (auto& _e : v) {
        os << _e << ",";
    }
    os << "}";
    return os;
}
template <typename T>
SampleUniquePtr<T> makeUnique(T* t)
{
    return SampleUniquePtr<T>{t};
}

void printFunc(std::string name) {
    std::cout << "+++++++++++++++++++++ " << name << " +++++++++++++++++++++" << std::endl;
}

void printHelpInfo()
{
    std::cout << "Usage: ./sample_demo [-h or --help] [-d or --datadir=<path to data directory>] "
                 "[--timingCacheFile=<path to timing cache file>]"
              << std::endl;
    std::cout << "--help, -h         Display help information" << std::endl;
    std::cout << "--datadir          Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(sampleTest);

    // builder
    auto builder = makeUnique(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        sample::gLogError << "Create inference builder failed." << std::endl;
        return 0;
    }
    // Runtime
    auto mRuntime = makeUnique(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime) {
        sample::gLogError << "Runtime object creation failed." << std::endl;
        return 0;
    }

    try {
        auto profileStream = samplesCommon::makeCudaStream();
        if (!profileStream) return 0;
        // Network
        auto preprocessorNetwork = makeUnique(builder->createNetworkV2(0));
        if (!preprocessorNetwork) {
            sample::gLogError << "Create network failed." << std::endl;
            return 0;
        }

        auto parser = samplesCommon::infer_object(nvonnxparser::createParser(*preprocessorNetwork, sample::gLogger.getTRTLogger()));
        bool parsingSuccess = parser->parseFromFile(locateFile(args.modelFileName, args.dataDirs).c_str(),
            static_cast<int>(sample::gLogger.getReportableSeverity()));
        if (!parsingSuccess) {
            sample::gLogError << "Failed to parse model." << std::endl;
            return false;
        }
        size_t inputSize = preprocessorNetwork->getNbInputs();
        sample::gLogError << "input_count: " << preprocessorNetwork->getNbInputs() << std::endl;
        for (int i=0; i<inputSize; ++i) {
            auto input = preprocessorNetwork->getInput(i);
            // sample::gLogError << "input_name: " << input->getName() << std::endl;
        }
        // auto mPredictionInputDims1 = network->getInput(1)->getDimensions();
        // auto mPredictionOutputDims = network->getOutput(0)->getDimensions();

        // auto input = preprocessorNetwork->addInput("input_A:0", nvinfer1::DataType::kFLOAT, Dims2{1, 784});
        // auto resizeLayer = preprocessorNetwork->addResize(*input);
        // resizeLayer->setOutputDimensions(Dims2{1, 784});
        // preprocessorNetwork->markOutput(*resizeLayer->getOutput(0));
        // Config
        auto preprocessorConfig = makeUnique(builder->createBuilderConfig());
        if (!preprocessorConfig) {
            sample::gLogError << "Create builder config failed." << std::endl;
            return 0;
        }
        // preprocessorConfig->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 22ULL<<30);
        // preprocessorConfig->setTacticSources(1 << static_cast<int>(TacticSource::kCUBLAS));
        preprocessorConfig->setProfileStream(*profileStream);
        // preprocessorConfig->setFlag(BuilderFlag::kDISABLE_COMPILATION_CACHE);
        auto profile = builder->createOptimizationProfile();
        for (int i=0; i<inputSize; ++i) {
            auto input = preprocessorNetwork->getInput(i);
            profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims2{1, args.dim});
            profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims2{512,  args.dim});
            profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims2{20000,  args.dim});
        }
        preprocessorConfig->addOptimizationProfile(profile);
        // preprocessorConfig->setCalibrationProfile(profile);

        SampleUniquePtr<nvinfer1::IHostMemory> preprocessorPlan
        = makeUnique(builder->buildSerializedNetwork(*preprocessorNetwork, *preprocessorConfig)); // 网络序列化
        if (!preprocessorPlan){
            sample::gLogError << "Preprocessor serialized engine build failed." << std::endl;
            return 0;
        }
        // Engine
        auto mPreprocessorEngine = makeUnique(mRuntime->deserializeCudaEngine(preprocessorPlan->data(), preprocessorPlan->size()));
        if (!mPreprocessorEngine){
            sample::gLogError << "Preprocessor engine deserialization failed." << std::endl;
            return 0;
        }
        auto const tensorName = mPreprocessorEngine->getIOTensorName(0);
        sample::gLogInfo << "Profile dimensions in preprocessor engine:" << std::endl;
        sample::gLogInfo << "    Minimum = " << mPreprocessorEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMIN)<< std::endl;
        sample::gLogInfo << "    Optimum = " << mPreprocessorEngine->getProfileShape(tensorName, 0, OptProfileSelector::kOPT)<< std::endl;
        sample::gLogInfo << "    Maximum = " << mPreprocessorEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMAX)<< std::endl;
        // ExecutionContext
        auto mPreprocessorContext = makeUnique(mPreprocessorEngine->createExecutionContext());
        if (!mPreprocessorContext) {
            sample::gLogError << "Preprocessor context build failed." << std::endl;
            return false;
        }
        // mPreprocessorContext->setEnqueueEmitsProfile(false);

        int32_t batch_size = args.batch;
        int32_t dim = args.dim;

        // std::vector<void*> vec(392);
        Dims inputDims = Dims2{batch_size, dim};
        for (int i=0; i<inputSize; ++i) {
            auto input = preprocessorNetwork->getInput(i);
            mPreprocessorContext->setInputShape(input->getName(), inputDims);
        }
        if (!mPreprocessorContext->allInputDimensionsSpecified()) {
            return 0;
        }

        // Inference Before
        size_t outputSize = preprocessorNetwork->getNbOutputs();
        std::vector<void*> preprocessorBindings(inputSize + outputSize, nullptr);
        std::shared_ptr<samplesCommon::ManagedBuffer> mInput;
        std::vector<std::shared_ptr<samplesCommon::ManagedBuffer>> mInputs(inputSize + outputSize);
        {
            std::vector<float> vec(batch_size * dim, 0.5);
            for (int i=0; i<inputSize; ++i) {
                std::cout << "inputSize: " << inputSize << ", batch_size: " << batch_size << ", dim: " << dim << std::endl;
                // samplesCommon::ManagedBuffer mInput{};
                mInput = std::make_shared<samplesCommon::ManagedBuffer>();
                mInput->deviceBuffer.resize(inputDims);
                mInputs[i] = mInput;
                {
                    PerfGuard guard("H2D", 1);
                    CHECK(cudaMemcpy(mInput->deviceBuffer.data(), vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice));
                }
                preprocessorBindings[i] = mInput->deviceBuffer.data();
                // std::vector<float> vec1(args.batch * args.dim, 0);
                // CHECK(cudaMemcpy(vec1.data(), preprocessorBindings[i], vec1.size() * sizeof(float), cudaMemcpyDeviceToHost));
                // std::cout << "output:" << vec1 << std::endl;
            }
        }

        Dims outputDims = Dims2{batch_size, 1};
        for (int i=0; i<outputSize; ++i) {
            auto mOutput = std::make_shared<samplesCommon::ManagedBuffer>();
            mOutput->deviceBuffer.resize(outputDims);
            mOutput->hostBuffer.resize(outputDims);
            preprocessorBindings[inputSize + i] = mOutput->deviceBuffer.data();
            mInputs[inputSize + i] = mOutput; // managed buffer
        }

        sample::gLogError << std::endl<< std::endl<< std::endl<< std::endl;
        // Inference
        bool status = false;
        sample::gLogError << "start_warmup" << std::endl;
        {
            PerfGuard guard("warmup", 10);
            for (int i=0; i<10; ++i) {
                status = mPreprocessorContext->executeV2(preprocessorBindings.data());
            }
        }
        sample::gLogError << "end..._warmup" << std::endl;
        // for (int i=0; i<10; ++i) {
        //     PerfGuard guard("Run", 10);
        //     for (int i=0; i<10; ++i) {
        //         status = mPreprocessorContext->executeV2(preprocessorBindings.data());
        //     }
        // }
        sample::gLogError << "-----------end..._infer" << std::endl;
        if (!status) return 0;
        {
            PerfGuard guard("D2H", 1);
            CHECK(cudaMemcpy(mOutput.hostBuffer.data(), mOutput.deviceBuffer.data(), mOutput.deviceBuffer.nbBytes(),
            cudaMemcpyDeviceToHost));
            const float* bufRaw = static_cast<const float*>(mOutput.hostBuffer.data());
            std::vector<float> prob(bufRaw, bufRaw + mOutput.hostBuffer.size());
            std::cout << "result:" << prob << std::endl;
        }

    } catch (std::runtime_error& e) {
        sample::gLogError << e.what()  << std::endl;
        return 0;
    }
    return sample::gLogger.reportPass(sampleTest);
}