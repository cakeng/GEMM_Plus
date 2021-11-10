// #define __INCLUDE_ARMNN 1
// #define __INCLUDE_XNNPACK 1
// #define __INCLUDE_PTMM 1
// #define __INCLUDE_DIRECT18 1
// #define __INCLUDE_IM2COL 1
// #define __MODE_TEST 1
// #define __MODE_SAVE 1
// #define __MODE_SINGLE 1

/*
*	main.cpp 
*	Created: 2020-7-26
*	Author: JongSeok Park (cakeng@naver.com)
*/
#include <iostream>
#include <cstring>
#include <fstream> 
#include <bitset>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cblas.h>
#include <random>
#define RAND_ABS_RANGE 100000

unsigned int depthwise;
unsigned int padding, stride, dilation;
unsigned int testBatch, testBlocks, testChannels, testHeight, testWidth, testFilHeight, testFilWidth;
unsigned int threadNum;
unsigned int runNumPre = 0;
unsigned int runNum = 0;
unsigned int testHout, testWout;
bool suppress = 0;
float *testInputTensor = nullptr;
float *testFilterTensor = nullptr;
float *testBiasTensor = nullptr;
float *testInputTensorNHWC = nullptr;
float *testFilterTensorNHWC = nullptr;
float *testInputTensorPTMM = nullptr;
float *testFilterTensorPTMM = nullptr;
float *testInputTensorDIRECT = nullptr;
float *testFilterTensorDIRECT = nullptr;
float *testFilterTensorIm2col = nullptr;

#ifdef __INCLUDE_MTEN
#include <miniTensor.h>
#endif

#ifdef __INCLUDE_PTMM
#include <ptmm.h>
#endif

#ifdef __INCLUDE_DIRECT18
#include <direct18.h>
#endif

#ifdef __INCLUDE_COMP
#include <comp.h>
#endif

void loadData(std::string fileLocation)
{
    if (!suppress)
    {
        std::cout << "Loading " << fileLocation << std::endl;
    }
    std::ifstream file(fileLocation, std::ios::in | std::ios::binary);
    char s[5];
    file.read(s, sizeof(char));
    if (s[0] != 'S')
    {
        printf("//// Loading Error!! - Wrong file format!! (S) ////\n");
        exit(1);
    }
    file.read(s, sizeof(unsigned int));
    depthwise = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    padding = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    stride = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    dilation = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    testBatch = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    testBlocks = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    testChannels = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    testHeight = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    testWidth = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    testFilHeight = *((unsigned int *)&s);
    file.read(s, sizeof(unsigned int));
    testFilWidth = *((unsigned int *)&s);
    file.read(s, sizeof(char));
    if (s[0] != 'D')
    {
        printf("//// Loading Error!! - Wrong file format!! (D) ////\n");
        exit(1);
    }
    else if (!suppress)
    {   // (Batch, blocks, channels, width, height, filHeight, filWidth, padding, stride, dilation, depthwise)
        printf("Batch - %d, Blocks - %d, Channels - %d, Height - %d, Width, %d, Filter Height - %d, Filter Width - %d, Padding - %d, Stride - %d, Dilation - %d, Depthwise - %d\n", 
            testBatch, testBlocks, testChannels, testHeight, testWidth, testFilHeight, testFilWidth, padding, stride, dilation, depthwise);
    }

    //Inputs
    #if ! __MODE_SINGLE
    if (posix_memalign((void **)(&testInputTensor), 128, (testBatch * testChannels * testHeight * testWidth) * sizeof(float)))
    {
        printf("testInputTensor loader - POSIX memalign failed.");
    }
    if (!suppress)
        printf("Loading input tensor NCHW.\n");
    file.read((char *)testInputTensor, testBatch * testChannels * testHeight * testWidth * sizeof(float));
    #else
    file.seekg(testBatch * testChannels * testHeight * testWidth * sizeof(float), file.cur);
    #endif
    file.read(s, sizeof(char));
    if (s[0] != 'I')
    {
        printf("//// Loading Error!! - Wrong file format!! (I) ////\n");
        exit(1);
    }

    #if __INCLUDE_ARMNN || __INCLUDE_XNNPACK || __INCLUDE_IM2COL
    if (posix_memalign((void **)(&testInputTensorNHWC), 128, (testBatch * testChannels * testHeight * testWidth) * sizeof(float)))
    {
        printf("testInputTensor loader - POSIX memalign failed.");
    }
    if (!suppress)
        printf("Loading input tensor NHWC.\n");
    file.read((char *)testInputTensorNHWC, testBatch * testChannels * testHeight * testWidth * sizeof(float));
    #else
    file.seekg(testBatch * testChannels * testHeight * testWidth * sizeof(float), file.cur);
    #endif
    file.read(s, sizeof(char));
    if (s[0] != 'I')
    {
        printf("//// Loading Error!! - Wrong file format!! (I) ////\n");
        exit(1);
    }

    #if __INCLUDE_PTMM
    if (posix_memalign((void **)(&testInputTensorPTMM), 128, (testBatch * testChannels * testHeight * testWidth) * sizeof(float)))
    {
        printf("testInputTensor loader - POSIX memalign failed.");
    }
    if (!suppress)
        printf("Loading input tensor PTMM.\n");
    file.read((char *)testInputTensorPTMM, testBatch * testChannels * testHeight * testWidth * sizeof(float));
    #else
    file.seekg(testBatch * testChannels * testHeight * testWidth * sizeof(float), file.cur);
    #endif
    file.read(s, sizeof(char));
    if (s[0] != 'I')
    {
        printf("//// Loading Error!! - Wrong file format!! (I) ////\n");
        exit(1);
    }

    int direct18Channels = testChannels;
    if (testChannels == 3)
    {
        direct18Channels = 8;
    }
    #if __INCLUDE_DIRECT18
    if (posix_memalign((void **)(&testInputTensorDIRECT), 128, (testBatch * direct18Channels * testHeight * testWidth) * sizeof(float)))
    {
        printf("testInputTensor loader - POSIX memalign failed.");
    }
    if (!suppress)
        printf("Loading input tensor DIRECT18.\n");
    file.read((char *)testInputTensorDIRECT, testBatch * direct18Channels * testHeight * testWidth * sizeof(float));
    #else
    file.seekg(testBatch * direct18Channels * testHeight * testWidth * sizeof(float), file.cur);
    #endif
    file.read(s, sizeof(char));
    if (s[0] != 'I')
    {
        printf("//// Loading Error!! - Wrong file format!! (I) ////\n");
        exit(1);
    }

    //Filters
    if (testFilterTensor != nullptr)
    {
        free(testFilterTensor);
    }
    unsigned int testFilterSize;
    if (depthwise == 0)
    {
        testFilterSize = testBlocks * testChannels * testFilHeight * testFilWidth;
    }
    else
    {
        testFilterSize = testChannels * testFilHeight * testFilWidth;
    }

    #if ! __MODE_SINGLE
    if (posix_memalign((void **)(&testFilterTensor), 128, (testFilterSize) * sizeof(float)))
    {
        printf("testFilterTensor loader - POSIX memalign failed.");
    }
    if (!suppress)
        printf("Loading filter tensor NCHW.\n");
    file.read((char *)testFilterTensor, testFilterSize * sizeof(float));
    #else
    file.seekg(testFilterSize * sizeof(float), file.cur);
    #endif
    file.read(s, sizeof(char));
    if (s[0] != 'F')
    {
        printf("//// Loading Error!! - Wrong file format!! (I) ////\n");
        exit(1);
    }

    #if __INCLUDE_ARMNN || __INCLUDE_XNNPACK || __INCLUDE_IM2COL
    if (posix_memalign((void **)(&testFilterTensorNHWC), 128, (testFilterSize) * sizeof(float)))
    {
        printf("testFilterTensor loader - POSIX memalign failed.");
    }
    if (!suppress)
        printf("Loading filter tensor NHWC.\n");
    file.read((char *)testFilterTensorNHWC, testFilterSize * sizeof(float));
    #else
    file.seekg(testFilterSize * sizeof(float), file.cur);
    #endif
    file.read(s, sizeof(char));
    if (s[0] != 'F')
    {
        printf("//// Loading Error!! - Wrong file format!! (F) ////\n");
        exit(1);
    }

    #if __INCLUDE_PTMM
    if (posix_memalign((void **)(&testFilterTensorPTMM), 128, (testFilterSize) * sizeof(float)))
    {
        printf("testFilterTensor loader - POSIX memalign failed.");
    }
    if (!suppress)
        printf("Loading filter tensor PTMM.\n");
    file.read((char *)testFilterTensorPTMM, testFilterSize * sizeof(float));
    #else
    file.seekg(testFilterSize * sizeof(float), file.cur);
    #endif
    file.read(s, sizeof(char));
    if (s[0] != 'F')
    {
        printf("//// Loading Error!! - Wrong file format!! (F) ////\n");
        exit(1);
    }

    #if __INCLUDE_DIRECT18
    if (posix_memalign((void **)(&testFilterTensorDIRECT), 128, (testFilterSize) * sizeof(float)))
    {
        printf("testFilterTensor loader - POSIX memalign failed.");
    }
    if (!suppress)
        printf("Loading filter tensor DIRECT.\n");
    file.read((char *)testFilterTensorDIRECT, testFilterSize * sizeof(float));
    #else
    file.seekg(testFilterSize * sizeof(float), file.cur);
    #endif
    file.read(s, sizeof(char));
    if (s[0] != 'F')
    {
        printf("//// Loading Error!! - Wrong file format!! (F) ////\n");
        exit(1);
    }

    #if __INCLUDE_IM2COL
    if (posix_memalign((void **)(&testFilterTensorIm2col), 128, (testFilterSize) * sizeof(float)))
    {
        printf("testFilterTensor loader - POSIX memalign failed.");
    }
    if (!suppress)
        printf("Loading filter tensor IM2COL.\n");
    file.read((char *)testFilterTensorIm2col, testFilterSize * sizeof(float));
    #else
    file.seekg(testFilterSize * sizeof(float), file.cur);
    #endif
    file.read(s, sizeof(char));
    if (s[0] != 'F')
    {
        printf("//// Loading Error!! - Wrong file format!! (F) ////\n");
        exit(1);
    }

    //Bias
    if (testBiasTensor != nullptr)
    {
        free(testBiasTensor);
    }
    if (posix_memalign((void **)(&testBiasTensor), 128, (testBlocks) * sizeof(float)))
    {
        printf("testInputTensor loader - POSIX memalign failed.");
    }
    file.read((char *)testBiasTensor, testBlocks * sizeof(float));
    file.read(s, sizeof(char));
    if (s[0] != 'B')
    {
        printf("//// Loading Error!! - Wrong file format!! (B) ////\n");
        exit(1);
    }
}

void saveData(std::string fileLocation)
{
    std::ofstream file(fileLocation, std::ios::out | std::ios::binary);
    printf("Saving tensors.\n");
    printf("Batch - %d, Blocks - %d, Channels - %d, Height - %d, Width, %d, Filter Height - %d, Filter Width - %d, Padding - %d, Stride - %d, Dilation - %d, Depthwise - %d\n", 
        testBatch, testBlocks, testChannels, testHeight, testWidth, testFilHeight, testFilWidth, padding, stride, dilation, depthwise);
    const char s[6] = "SDIFB";
    file.write(s, sizeof(char));
    file.write((char *)&depthwise, sizeof(unsigned int));
    file.write((char *)&padding, sizeof(unsigned int));
    file.write((char *)&stride, sizeof(unsigned int));
    file.write((char *)&dilation, sizeof(unsigned int));
    file.write((char *)&testBatch, sizeof(unsigned int));
    file.write((char *)&testBlocks, sizeof(unsigned int));
    file.write((char *)&testChannels, sizeof(unsigned int));
    file.write((char *)&testHeight, sizeof(unsigned int));
    file.write((char *)&testWidth, sizeof(unsigned int));
    file.write((char *)&testFilHeight, sizeof(unsigned int));
    file.write((char *)&testFilWidth, sizeof(unsigned int));
    file.write(s + 1, sizeof(char));
    file.write((char *)testInputTensor, testBatch * testChannels * testHeight * testWidth * sizeof(float));
    file.write(s + 2, sizeof(char));
    file.write((char *)testInputTensorNHWC, testBatch * testChannels * testHeight * testWidth * sizeof(float));
    file.write(s + 2, sizeof(char));
    file.write((char *)testInputTensorPTMM, testBatch * testChannels * testHeight * testWidth * sizeof(float));
    file.write(s + 2, sizeof(char));
    int direct18Channels = testChannels;
    if (testChannels == 3)
    {
        direct18Channels = 8;
    }
    file.write((char *)testInputTensorDIRECT, testBatch * direct18Channels * testHeight * testWidth * sizeof(float));
    file.write(s + 2, sizeof(char));
    unsigned int testFilterSize;
    if (depthwise == 0)
    {
        testFilterSize = testBlocks * testChannels * testFilHeight * testFilWidth;
    }
    else
    {
        testFilterSize = testChannels * testFilHeight * testFilWidth;
    }
    file.write((char *)testFilterTensor, testFilterSize * sizeof(float));
    file.write(s + 3, sizeof(char));
    file.write((char *)testFilterTensorNHWC, testFilterSize * sizeof(float));
    file.write(s + 3, sizeof(char));
    file.write((char *)testFilterTensorPTMM, testFilterSize * sizeof(float));
    file.write(s + 3, sizeof(char));
    file.write((char *)testFilterTensorDIRECT, testFilterSize * sizeof(float));
    file.write(s + 3, sizeof(char));
    file.write((char *)testFilterTensorIm2col, testFilterSize * sizeof(float));
    file.write(s + 3, sizeof(char));
    file.write((char *)testBiasTensor, testBlocks * sizeof(float));
    file.write(s + 4, sizeof(char));
}

float *NCHWtoNHWC(float *input, int blocks, int channels, int height, int width)
{
    float *out;
    if (posix_memalign((void **)(&out), 128, blocks * channels * height * width * sizeof(float)))
    {
        printf("Test input NHWC - POSIX memalign failed.");
    }
    for (int bIdx = 0; bIdx < blocks; bIdx++)
    {
        for (int cIdx = 0; cIdx < channels; cIdx++)
        {
            float *saveTarget = out + bIdx * channels * height * width + cIdx;
            float *loadTarget = input + bIdx * channels * height * width + cIdx * height * width;
            for (int hIdx = 0; hIdx < height; hIdx++)
            {
                for (int wIdx = 0; wIdx < width; wIdx++)
                {
                    *(saveTarget) = *(loadTarget);
                    saveTarget += channels;
                    loadTarget += 1;
                }
            }
        }
    }
    return out;
}

void printNCHW(float *input, int block, int channels, int height, int width)
{
    std::cout << "Printing NCHW float Arr:" << std::endl;
    for (int b = 0; b < block; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            printf("Channel %d\n", c);
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    printf("%4.4f\t", *(input + b * channels * height * width + c * height * width + h * width + w));
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

void printNHWC(float *input, int block, int channels, int height, int width)
{
    std::cout << "Printing NHWC float Arr:" << std::endl;
    for (int b = 0; b < block; b++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int c = 0; c < channels; c++)
                {
                    printf("%4.4f\t", *(input + b * channels * height * width + h * width * channels + w * channels + c));
                }
                printf("-(%d, %d)\n", h, w);
            }
        }
        printf("\n");
    }
}

void printArray(float *input, int newlineNum, int size)
{
    {
        for (int idx = 0; idx < size; idx++)
        {
            printf("%6.4f\t", *(input + idx));
            if (idx%newlineNum == (newlineNum-1))
            {
                printf("- (%d)\n", idx);
            }
        }
        printf("\nEnd of print.\n");
    }
}

#ifdef __INCLUDE_ARMNN
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Descriptors.hpp>
#include <arm_compute/runtime/Scheduler.h>

double armnnTest(float *armnnOutput, int runNum, int runNumPre)
{
    using namespace armnn;

    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    arm_compute::Scheduler::get().set_num_threads(threadNum);

    // Construct ArmNN network
    armnn::NetworkId networkIdentifier;
    INetworkPtr myNetwork = INetwork::Create();

    armnn::Convolution2dDescriptor Conv2dDesc;
    Conv2dDesc.m_PadTop = padding;
    Conv2dDesc.m_PadBottom = padding;
    Conv2dDesc.m_PadRight = padding;
    Conv2dDesc.m_PadLeft = padding;
    Conv2dDesc.m_StrideX = stride;
    Conv2dDesc.m_StrideY = stride;
    Conv2dDesc.m_DilationX = dilation;
    Conv2dDesc.m_DilationY = dilation;
    Conv2dDesc.m_BiasEnabled = true;
    Conv2dDesc.m_DataLayout = DataLayout::NCHW;

    armnn::DepthwiseConvolution2dDescriptor ConvDepth2dDesc;
    ConvDepth2dDesc.m_PadTop = padding;
    ConvDepth2dDesc.m_PadBottom = padding;
    ConvDepth2dDesc.m_PadRight = padding;
    ConvDepth2dDesc.m_PadLeft = padding;
    ConvDepth2dDesc.m_StrideX = stride;
    ConvDepth2dDesc.m_StrideY = stride;
    ConvDepth2dDesc.m_DilationX = dilation;
    ConvDepth2dDesc.m_DilationY = dilation;
    ConvDepth2dDesc.m_BiasEnabled = true;
    ConvDepth2dDesc.m_DataLayout = DataLayout::NCHW;

    TensorInfo weightsInfo;
    TensorInfo inputTensorInfo;
    TensorInfo outputTensorInfo;
    TensorInfo biasInfo(TensorShape({testBlocks}), DataType::Float32);

    bool del = false;
    float *filter;
    float *input;

    // For NHWC
    if (depthwise == 0)
    {
        weightsInfo = TensorInfo(TensorShape({testBlocks, testFilHeight, testFilWidth, testChannels}), DataType::Float32);
        filter = testFilterTensorNHWC;
        Conv2dDesc.m_DataLayout = DataLayout::NHWC;
    }
    else
    {
        weightsInfo = TensorInfo(TensorShape({1, testChannels, testFilHeight, testFilWidth}), DataType::Float32);
        filter = testFilterTensorNHWC;
        ConvDepth2dDesc.m_DataLayout = DataLayout::NHWC;
    }
    input = testInputTensorNHWC;
    inputTensorInfo = TensorInfo(TensorShape({testBatch, testHeight, testWidth, testChannels}), DataType::Float32);
    outputTensorInfo = TensorInfo(TensorShape({testBatch, testHout, testWout, testBlocks}), DataType::Float32);

    armnn::ConstTensor bias(biasInfo, testBiasTensor);
    armnn::ConstTensor weights(weightsInfo, filter);
    IConnectableLayer *convLayer;
    if (depthwise == 0)
    {
        convLayer = myNetwork->AddConvolution2dLayer(Conv2dDesc, weights, bias, "Armnn Conv2d");
    }
    else
    {
        convLayer = myNetwork->AddDepthwiseConvolution2dLayer(ConvDepth2dDesc, weights, bias, "Armnn Conv2d Depthwise");
    }
    IConnectableLayer *InputLayer = myNetwork->AddInputLayer(0);
    IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);
    #ifdef __MODE_SINGLE 
    free(testBiasTensor);
    free(filter);
    #endif

    InputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    IRuntimePtr run = IRuntime::Create(options);

    //Set the tensors in the network.
    InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = Optimize(*myNetwork, {Compute::CpuAcc}, run->GetDeviceSpec());
    if (!optNet)
    {
        // This shouldn't happen for this simple sample, with reference backend.
        // But in general usage Optimize could fail if the hardware at runtime cannot
        // support the model that has been provided.
        std::cerr << "Error: Failed to optimise the input network." << std::endl;
    }

    // Load graph into runtime
    run->LoadNetwork(networkIdentifier, std::move(optNet));
    armnn::InputTensors inputTensors{{0, armnn::ConstTensor(run->GetInputTensorInfo(networkIdentifier, 0), input)}};
    armnn::OutputTensors armnnOutputTensors{{0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), armnnOutput)}};

    // Execute network
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < runNumPre; i++)
    {
        run->EnqueueWorkload(networkIdentifier, inputTensors, armnnOutputTensors);
    }
    t2 = std::chrono::high_resolution_clock::now();
    // Execute network
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < runNum; i++)
    {
        run->EnqueueWorkload(networkIdentifier, inputTensors, armnnOutputTensors);
    }
    t2 = std::chrono::high_resolution_clock::now();
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)runNum;
}
#endif

#ifdef __INCLUDE_XNNPACK
#include <xnnpack.h>
#include <math.h>

double xnnpackTest(float *xnnpackOutput, int runNum, int runNumPre)
{
    pthreadpool_t threadpool = pthreadpool_create(threadNum);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();

    float *input = testInputTensorNHWC;
    float *filter;
    xnn_status status;
    if (xnn_initialize(nullptr /* allocator */) != xnn_status_success)
    {
        std::cerr << "failed to initialize XNNPACK" << std::endl;
    }
    xnn_operator_t op0 = nullptr;
    if (depthwise == 0)
    {
        filter = testFilterTensorNHWC;
        status = xnn_create_convolution2d_nhwc_f32(
            padding /* top padding */, padding /* right padding */,
            padding /* bottom padding */, padding /* left padding */,
            testFilHeight /* kernel height */, testFilWidth /* kernel width */,
            stride /* subsampling height */, stride /* subsampling width */,
            dilation /* dilation_height */, dilation /* dilation_width */,
            1 /* groups */,
            testChannels /* input channels per group */,
            testBlocks /* xnnpackOutput_channels_per_group */,
            testChannels /* input pixel stride */,
            testBlocks /* xnnpackOutput pixel stride */,
            filter, testBiasTensor,
            -(__builtin_inff()) /* xnnpackOutput min */, (__builtin_inff()) /* xnnpackOutput max */,
            0 /* flags */,
            &op0);
    }
    else
    {
        filter = testFilterTensorNHWC;
        status = xnn_create_convolution2d_nhwc_f32(
            padding /* top padding */, padding /* right padding */,
            padding /* bottom padding */, padding /* left padding */,
            testFilHeight /* kernel height */, testFilWidth /* kernel width */,
            stride /* subsampling height */, stride /* subsampling width */,
            dilation /* dilation_height */, dilation /* dilation_width */,
            testChannels /* groups */,
            1 /* input channels per group */,
            1 /* xnnpackOutput_channels_per_group */,
            testChannels /* input pixel stride */,
            testChannels /* xnnpackOutput pixel stride */,
            filter, testBiasTensor,
            -(__builtin_inff()) /* xnnpackOutput min */, (__builtin_inff()) /* xnnpackOutput max */,
            0 /* flags */,
            &op0);
    }
    if (status != xnn_status_success)
    {
        std::cerr << "failed to create operation #0 - status: " << status << std::endl;
    }
    #ifdef __MODE_SINGLE 
    free(filter);
    free(testBiasTensor);
    #endif
    status = xnn_setup_convolution2d_nhwc_f32(
        op0,
        testBatch /* batch size */, testHeight /* input height */, testWidth /* input width */,
        input /* input */, xnnpackOutput /* xnnpackOutput */,
        threadpool /* threadpool */);
    if (status != xnn_status_success)
    {
        std::cerr << "failed to setup operation #0" << std::endl;
    }
    for (size_t i = 0; i < runNumPre; i++)
    {
        status = xnn_run_operator(op0, threadpool);
    }
    // Execute network
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < runNum; i++)
    {
        status = xnn_run_operator(op0, threadpool);
    }
    t2 = std::chrono::high_resolution_clock::now();
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)runNum;
}
#endif

#ifdef __INCLUDE_IM2COL
void im2col(int batch, int hout, int wout, int cin, int hfil, int wfil, int str, int dil, int hin, int win, int pad, float *input, float *output)
{
    #pragma omp parallel for collapse(2)
    for (int bat = 0; bat < batch; bat++)
    {
        for (int ho = 0; ho < hout; ho++)
        {
            for (int wo = 0; wo < wout; wo++)
            {
                float* target = output + bat * hout * wout * hfil * wfil * cin  + (ho * wout + wo) * hfil * wfil * cin;
                for (int hf = 0; hf < hfil; hf++)
                {
                    int hi = ho * str + hf * dil - pad;
                    if (0 <= hi && hi < hin)
                    {
                        for (int wf = 0; wf < wfil; wf++)
                        {
                            int wi = wo * str + wf * dil - pad;
                            if (0 <= wi && wi < win)
                            {
                                memcpy(target, input + bat * hin * win * cin + (hi * win + wi) * cin, cin * sizeof(float));
                            }
                            else
                            {
                                bzero(target, cin * sizeof(float));
                            }
                            target += cin;
                        }
                    }
                    else
                    {
                        bzero(target, wfil * cin * sizeof(float));
                        target += wfil * cin;
                    }
                }
            }
        }
    }
}
float *im2GemmFilter(float *input, int blocks, int channels, int height, int width)
{
    float *out;
    if (posix_memalign((void **)(&out), 128, blocks * channels * height * width * sizeof(float)))
    {
        printf("Test im2Gemm Filter - POSIX memalign failed.");
    }
    for (int bIdx = 0; bIdx < blocks; bIdx++)
    {
        for (int cIdx = 0; cIdx < channels; cIdx++)
        {
            float *saveTarget = out + bIdx + cIdx*blocks;
            float *loadTarget = input + bIdx * channels * height * width + cIdx * height * width;
            for (int hIdx = 0; hIdx < height; hIdx++)
            {
                for (int wIdx = 0; wIdx < width; wIdx++)
                {
                    *(saveTarget) = *(loadTarget);
                    saveTarget += channels*blocks;
                    loadTarget += 1;
                }
            }
        }
    }
    return out;
}
double im2Gemm(float *im2colOutput, int runNum, int runNumPre)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    const int M = testBlocks, K = testFilHeight * testFilWidth * testChannels, N = testBatch*testHout*testWout;

    float *filter = testFilterTensorIm2col;
    float *input = testInputTensorNHWC;
    //printArray(input, testChannels, testChannels * testHeight * testWidth);
    //printArray(filter, testBlocks, testBlocks*testChannels*testFilHeight*testFilWidth);
    float *newInput;
    if (posix_memalign((void **)(&newInput), 128, K * N * sizeof(float)))
    {
        printf("Test input NHWC - POSIX memalign failed.");
    }
    for (size_t i = 0; i < runNumPre; i++)
    {
        im2col(testBatch, testHout, testWout, testChannels, testFilHeight, testFilWidth, stride, dilation, testHeight, testWidth, padding, input, newInput);
        for (int n = 0; n < N; n++)
        {
            memcpy(im2colOutput + n*M, testBiasTensor, testBlocks*sizeof(float));
        }
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, filter, M, newInput, K, 1, im2colOutput, M);    
    }
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < runNum; i++)
    {
        im2col(testBatch, testHout, testWout, testChannels, testFilHeight, testFilWidth, stride, dilation, testHeight, testWidth, padding, input, newInput);  
    }
    t2 = std::chrono::high_resolution_clock::now();
    double convOnlyTime = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)runNum;
    if (!suppress)
    {
        printf("Im2GEMM im2col only - %4.4f ms\n", convOnlyTime / 1000);
    }
    else
    {
        printf("%4.4f, ", convOnlyTime / 1000);
    }
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < runNum; i++)
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, filter, M, newInput, K, 1, im2colOutput, M);    
    }
    t2 = std::chrono::high_resolution_clock::now();
    convOnlyTime = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)runNum;
    if (!suppress)
    {
        printf("Im2GEMM Conv only - %4.4f ms\n", convOnlyTime / 1000);
    }
    else
    {
        printf("%4.4f, ", convOnlyTime / 1000);
    }
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < runNum; i++)
    {
        im2col(testBatch, testHout, testWout, testChannels, testFilHeight, testFilWidth, stride, dilation, testHeight, testWidth, padding, input, newInput);
        for (int n = 0; n < N; n++)
        {
            memcpy(im2colOutput + n*M, testBiasTensor, testBlocks*sizeof(float));
        }
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, filter, M, newInput, K, 1, im2colOutput, M);    
    }
    t2 = std::chrono::high_resolution_clock::now();
    //printArray(newInput, testFilHeight * testFilWidth * testChannels, testFilHeight * testFilWidth * testChannels * testHout * testWout);
    free(newInput);
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)runNum;
}
#endif


void functionTest(bool printNumOnly = false)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    openblas_set_num_threads(threadNum);
    double mTenAvg = 0, torchAvg = 0, armnnAvg = 0, xnnpackAvg = 0, ptmmAvg = 0
        , direct18Avg = 0, im2GemmAvg = 0, kn2GemmAvg = 0, mecAvg = 0, compAvg = 0;

    if (!suppress)
    {
        printf("Batch: %d, Output Channels: %d, Output Height: %d, Output Width: %d, Input Channels %d, Filter Height %d, Filter width %d, padding: %d, stride: %d, dilation: %d, depthwise: %d, %d thread(s), %d runs.\n", 
            testBatch, testBlocks, testHout, testWout, testChannels, testFilHeight, testFilWidth, padding, stride, dilation, depthwise, threadNum, runNum);
    }

#ifdef __INCLUDE_PTMM
    int vecLength = 8;
    if (testChannels == 3)
    {
        vecLength = 3;
    }
    float *ptmmOutput;
    // printf("Printing PTMM Input.\n");
    // printNHWC (testInputTensorPTMM, testBatch, testChannels, testHeight, testWidth);
    if (posix_memalign((void **)(&ptmmOutput), 128, testBatch * testBlocks * testHout * testWout * sizeof(float)))
    {
        printf("PTMM output - POSIX memalign failed.");
    }
    ptmm::ptmm_num_threads = threadNum;
    ptmm ptmmFilter(0, testFilterTensorPTMM, testBlocks, testChannels, testFilHeight, testFilWidth, false, depthwise);
    // printf("Printing PTMM Filter.\n");
    // ptmmFilter.print(16, 0);
    for (int i = 0; i < runNum / 10; i++)
    {
        ptmmFilter.conv(testInputTensorPTMM, ptmmOutput, testBiasTensor, testBatch, testHeight, testWidth, padding, stride, dilation);
    }
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runNum; i++)
    {
        ptmmFilter.conv(testInputTensorPTMM, ptmmOutput, testBiasTensor, testBatch, testHeight, testWidth, padding, stride, dilation);
    }
    t2 = std::chrono::high_resolution_clock::now();
    ptmmAvg = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / runNum;

    // printf("Printing PTMM output.\n");
    // printNHWC (ptmmOutput, testBatch, testBlocks, testHout, testWout);
#endif
#ifdef __INCLUDE_DIRECT18
    float *direct18Output;
    if (posix_memalign((void **)(&direct18Output), 128, testBatch * testBlocks * testHout * testWout * sizeof(float)))
    {
        printf("DIRECT18 output - POSIX memalign failed.");
    }
    direct18::direct18_num_threads = threadNum;
    direct18 direct18Filter(0, testFilterTensorDIRECT, testBlocks, testChannels, testFilHeight, testFilWidth, false);
    if (dilation == 1)
    {
        for (int i = 0; i < runNum / 10; i++)
        {
            direct18Filter.conv(testInputTensorDIRECT, direct18Output, testBiasTensor, testHeight, testWidth, padding, stride, dilation);
        }
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runNum; i++)
        {
            direct18Filter.conv(testInputTensorDIRECT, direct18Output, testBiasTensor, testHeight, testWidth, padding, stride, dilation);
        }
        t2 = std::chrono::high_resolution_clock::now();
    }
    else
    {
        if (!printNumOnly)
        {
            printf("Direct 18 cannot compute dilation %d.\n", dilation);
        }
        t1 = std::chrono::high_resolution_clock::now();
        t2 = std::chrono::high_resolution_clock::now();
    }
    direct18Avg = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / runNum;
    // printf("Printing DIRECT18 output.\n");
    // printNHWC (direct18OutputOrdered, testBatch, testBlocks, testHout, testWout);
#endif
#ifdef __INCLUDE_IM2COL
    float *im2GemmOut;
    if (posix_memalign((void **)(&im2GemmOut), 128, testBatch * testBlocks * testHout * testWout * sizeof(float)))
    {
        printf("im2Gemm output - POSIX memalign failed.");
    }
    im2GemmAvg = im2Gemm(im2GemmOut, runNum, runNumPre);
    // printf("Printing im2Gemm output.\n");
    // printNHWC(im2GemmOut, testBatch, testBlocks, testHout, testWout);
#endif
#ifdef __INCLUDE_ARMNN
    float *armnnOut;
    if (posix_memalign((void **)(&armnnOut), 128, testBatch * testBlocks * testHout * testWout * sizeof(float)))
    {
        printf("ARMNN output - POSIX memalign failed.");
    }
    armnnAvg = armnnTest(armnnOut, runNum, runNumPre);
    if (!printNumOnly)
    {
    // printf("Printing ARMNN output.\n");
    // printNHWC(armnnOut, testBatch, testBlocks, testHout, testWout);
#ifdef __INCLUDE_PTMM
        printf("Comparing ptmm output to armnn output.\n");
        float *ptmmOutputNHWC;
        ptmmOutputNHWC = ptmm::deVectorize(ptmmOutput, testBatch, testBlocks, testHout, testWout, 8);
        ptmm::floatCompare(ptmmOutputNHWC, armnnOut, testBatch, testBlocks, testHout, testWout, 0.008, true);
        free(ptmmOutputNHWC);
#endif
    }
#endif
#ifdef __INCLUDE_XNNPACK
    float *xnnpackOut;
    if (posix_memalign((void **)(&xnnpackOut), 128, testBatch * testBlocks * testHout * testWout * sizeof(float)))
    {
        printf("XNNPACK output - POSIX memalign failed.");
    }
    xnnpackAvg = xnnpackTest(xnnpackOut, runNum, runNumPre);
    if (!printNumOnly)
    {
        // printf("Printing XNNPACK output.\n");
        // printNHWC(xnnpackOut, testBatch, testBlocks, testHout, testWout);
        #ifdef __INCLUDE_PTMM
                #ifdef __INCLUDE_ARMNN
                printf("Comparing armnn output to XNNPACK output.\n");
                ptmm::floatCompare(armnnOut, xnnpackOut, testBatch, testBlocks, testHout, testWout, 0.008, true);    
                #endif
                float *ptmmOutputNHWC;
                ptmmOutputNHWC = ptmm::deVectorize(ptmmOutput, testBatch, testBlocks, testHout, testWout, 8);
                printf("Comparing ptmm output to XNNPACK output.\n");
                ptmm::floatCompare(ptmmOutputNHWC, xnnpackOut, testBatch, testBlocks, testHout, testWout, 0.008, true);
                free(ptmmOutputNHWC);
                #ifdef __INCLUDE_IM2COL
                    printf("Comparing im2Gemm output to XNNPACK output.\n");
                    ptmm::floatCompare(im2GemmOut, xnnpackOut, testBatch, testBlocks, testHout, testWout, 0.008, true);
                #endif
        #endif
        #ifdef __INCLUDE_DIRECT18
                if (dilation == 1)
                {
                    float *direct18OutputNHWC;
                    direct18OutputNHWC = direct18::deVectorize(direct18Output, testBatch, testBlocks, testHout, testWout);
                    printf("Comparing direct18 output to XNNPACK output.\n");
                    direct18::floatCompare(direct18OutputNHWC, xnnpackOut, testBatch, testBlocks, testHout, testWout, 0.008, true);
                    free(direct18OutputNHWC);
                }
        #endif
    }
    free(xnnpackOut);
#endif
#ifdef __INCLUDE_ARMNN
    free(armnnOut);
#endif
#ifdef __INCLUDE_IM2COL
    free(im2GemmOut);
#endif
#ifdef __INCLUDE_PTMM
    free(ptmmOutput);
#endif
#ifdef __INCLUDE_DIRECT18
    free(direct18Output);
#endif

    if (printNumOnly)
    {
#ifdef __INCLUDE_PTMM
        printf("%4.4f", ptmmAvg / 1000);
#endif
#ifdef __INCLUDE_DIRECT18
        printf("%4.4f", direct18Avg / 1000);
#endif
#ifdef __INCLUDE_ARMNN
        printf("%4.4f", armnnAvg / 1000);
#endif
#ifdef __INCLUDE_XNNPACK
        printf("%4.4f", xnnpackAvg / 1000);
#endif
#ifdef __INCLUDE_IM2COL
        printf("%4.4f", im2GemmAvg / 1000);
#endif
    }
    else
    {
        printf("Avg. Time - im2Gemm %4.4f ms, Direct18 %4.4f ms, Armnn: %4.4f ms, XNNPACK: %4.4f ms,  ptmm: %4.4f ms\n", 
            im2GemmAvg /1000, direct18Avg / 1000, armnnAvg / 1000, xnnpackAvg / 1000, ptmmAvg / 1000);
    }
}

void makeTensors()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-RAND_ABS_RANGE*2, RAND_ABS_RANGE*2);

    if (posix_memalign((void **)(&testInputTensor), 128, (testBatch * testChannels * testHeight * testWidth) * sizeof(float)))
    {
        printf("makeTensors input - POSIX memalign failed.");
    }
    for (int i = 0; i < testBatch * testChannels * testHeight * testWidth; i++)
    {
        *(testInputTensor + i) = ((float)dis(gen))/((float)RAND_ABS_RANGE); // Range: -2 ~ 2
    }
    int testFilterSize;
    if (depthwise == 0)
    {
        testFilterSize = testBlocks * testChannels * testFilHeight * testFilWidth;
    }
    else
    {
        testFilterSize = testChannels * testFilHeight * testFilWidth;
    }
    if (posix_memalign((void **)(&testFilterTensor), 128, (testFilterSize) * sizeof(float)))
    {
        printf("makeTensors filter - POSIX memalign failed.");
    }
    for (int i = 0; i < testFilterSize; i++)
    {
        *(testFilterTensor+ i) = ((float)dis(gen))/((float)RAND_ABS_RANGE); // Range: -2 ~ 2
    }
    if (posix_memalign((void **)(&testBiasTensor), 128, testBlocks * sizeof(float)))
    {
        printf("makeTensors bias - POSIX memalign failed.");
    }
    for (int i = 0; i < testBlocks; i++)
    {
        *(testBiasTensor + i) = ((float)dis(gen))/((float)RAND_ABS_RANGE); // Range: -2 ~ 2
    }

    if (posix_memalign((void **)(&testInputTensorNHWC), 128, (testBatch * testChannels * testHeight * testWidth) * sizeof(float)))
    {
        printf("makeTensors input - POSIX memalign failed.");
    }
    if (posix_memalign((void **)(&testInputTensorPTMM), 128, (testBatch * testChannels * testHeight * testWidth) * sizeof(float)))
    {
        printf("makeTensors input - POSIX memalign failed.");
    }
    int direct18Channels = testChannels;
    if (testChannels == 3)
    {
        direct18Channels = 8;
    }
    if (posix_memalign((void **)(&testInputTensorDIRECT), 128, (testBatch * direct18Channels * testHeight * testWidth) * sizeof(float)))
    {
        printf("makeTensors input - POSIX memalign failed.");
    }
    testInputTensorNHWC = NCHWtoNHWC(testInputTensor, testBatch, testChannels, testHeight, testWidth);
    #if __INCLUDE_PTMM
    int vecLength = testChannels == 3? 3 : 8;
    testInputTensorPTMM = ptmm::vectorize(testInputTensor, testBatch, testChannels, testHeight, testWidth, vecLength);
    #endif
    #if __INCLUDE_DIRECT18
    testInputTensorDIRECT = direct18::vectorize(testInputTensor, testBatch, testChannels, testHeight, testWidth);
    #endif


    if (posix_memalign((void **)(&testFilterTensorNHWC), 128, (testFilterSize) * sizeof(float)))
    {
        printf("makeTensors filter - POSIX memalign failed.");
    }
    if (posix_memalign((void **)(&testFilterTensorPTMM), 128, (testFilterSize) * sizeof(float)))
    {
        printf("makeTensors filter - POSIX memalign failed.");
    }
    if (posix_memalign((void **)(&testFilterTensorDIRECT), 128, (testFilterSize) * sizeof(float)))
    {
        printf("makeTensors filter - POSIX memalign failed.");
    }
    if (posix_memalign((void **)(&testFilterTensorIm2col), 128, (testFilterSize) * sizeof(float)))
    {
        printf("makeTensors filter - POSIX memalign failed.");
    }
    if (depthwise == 0)
    {
        testFilterTensorNHWC = NCHWtoNHWC(testFilterTensor, testBlocks, testChannels, testFilHeight, testFilWidth);
        #if __INCLUDE_IM2COL
        testFilterTensorIm2col = im2GemmFilter(testFilterTensor, testBlocks, testChannels, testFilHeight, testFilWidth);
        #endif
        #if __INCLUDE_DIRECT18
        direct18 direct18Filter(testFilterTensor, testBlocks, testChannels, testFilHeight, testFilWidth, false);
        memcpy(testFilterTensorDIRECT, direct18Filter.getFilterPtr(), (testFilterSize) * sizeof(float));
        #endif
    }
    else
    {
        memcpy(testFilterTensorNHWC, testFilterTensor, (testFilterSize) * sizeof(float));
        bzero(testFilterTensorDIRECT, (testFilterSize) * sizeof(float));
        bzero(testFilterTensorIm2col, (testFilterSize) * sizeof(float));
    }
    #if __INCLUDE_PTMM
    ptmm ptmmFilter(testFilterTensor, testBlocks, testChannels, testFilHeight, testFilWidth, false, depthwise);
    memcpy(testFilterTensorPTMM, ptmmFilter.getFilterPtr(), (testFilterSize) * sizeof(float));
    #endif
}

int main(int argc, char *argv[])
{
    #ifndef __MODE_TEST
    if (argc < 4)
    {
        printf ("Not enough arguments! - %d arguments given.\n", argc-1);
        return 0;
    }
    if (argc > 4)
    {
        suppress = atoi(argv[4]);
    }
    runNum = atoi(argv[2]);
    threadNum = atoi(argv[3]);
    std::string fileLocation(argv[1]);
    loadData(fileLocation);
    #else
    #ifdef __MODE_SAVE
    if (argc < 12)
    {
        printf ("Not enough arguments! - %d arguments given.\n", argc-1);
        return 0;
    }
    #else
    if (argc < 14)
    {
        printf ("Not enough arguments! - %d arguments given.\n", argc-1);
        return 0;
    }
    runNum = atoi(argv[12]);
    threadNum = atoi(argv[13]);
    #endif
    testBatch = atoi(argv[1]);
    testBlocks = atoi(argv[2]);
    testChannels = atoi(argv[3]);
    testWidth = atoi(argv[4]);
    testHeight = atoi(argv[5]);
    testFilHeight = atoi(argv[6]);
    testFilWidth = atoi(argv[7]);
    padding = atoi(argv[8]);
    stride = atoi(argv[9]);
    dilation = atoi(argv[10]);
    depthwise = atoi(argv[11]);

    makeTensors();
    #endif
    testHout = (testHeight + padding * 2 - dilation * (testFilHeight - 1) - 1) / stride + 1;
    testWout = (testWidth + padding * 2 - dilation * (testFilWidth - 1) - 1) / stride + 1;
    #ifdef __MODE_SAVE
    char loc[200] = {0};
    if (!depthwise)
    {
        sprintf(&(loc[0]), "Conv_Layer_B%d_Co%d_Ci%d_Ho%d_Wo%d_Hf%d_Wf%d_P%d_S%d_D%d", testBatch, testBlocks, testChannels, testHout, testWout, testFilHeight, testFilWidth, padding, stride, dilation);
    }
    else
    {
        sprintf(&(loc[0]), "Depth_Layer_B%d_Co%d_Ci%d_Ho%d_Wo%d_Hf%d_Wf%d_P%d_S%d_D%d", testBatch, testBlocks, testChannels, testHout, testWout, testFilHeight, testFilWidth, padding, stride, dilation);
    }
    saveData(loc);
    #else
    functionTest(suppress);
    #endif
    return 0;
}