/*
*	miniTensor.h 
*	Created: 2021-1-9
*	Author: JongSeok Park (cakeng@naver.com)
*/
#pragma oncemake
#include <cstring>
#include <cmath>
#include <thread>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <pthread.h>
#include <chrono>

#define __LL_CACHE_SIZE 1048576 // LL Cache size in bytes.
#define __D1_CACHE_SIZE 32768   // L1 Data Cache size in bytes.
// #define __LL_CACHE_SIZE 30000 // LL Cache size in bytes.
// #define __D1_CACHE_SIZE 5000   // L1 Data Cache size in bytes.
#define __VECTOR_SIZE 8 // Default Vector size.
#define __DEBUG_MTEN 1
//#define __INCLUDE_TORCH 1

#define __LANE_SIZE 4
#include <arm_neon.h>

#ifdef __INCLUDE_TORCH
#include <torch/torch.h>
#endif

#define __MAX_NUM_THREADS 12

struct mTenConvThreadData;
struct mTenThreadArg
{
   int id;
   void* threadDataPtr;
};

class mTen
{
private:
    int blocks, channels, height, width; 
    bool isNHWC;
    int vecSize; // Size of a signle vector. 
    int vecNum; // Number of vectors blocks to contain all channels.

    float* tensorPtr;

    friend void* mTenConvThreadRoutine(void* threadArg);
    friend void* mTenConvDepthThreadRoutine8to8(void* threadArg);
    friend void* mTenConvPointwiseThreadRoutine8to8(void* threadArg);
    friend void* mTenVectorizeThreadRoutine(void *threadArg);
    friend void* mTenCompareThreadRoutine(void *threadArg);

    static mTenThreadArg threadArgArr[__MAX_NUM_THREADS];

    static pthread_cond_t runningCondition;
    static int runningThreads;
    static pthread_mutex_t runningMutex;

    static pthread_t convThreads[__MAX_NUM_THREADS];
    static pthread_cond_t convThreadConditions[__MAX_NUM_THREADS];
    static pthread_mutex_t convThreadMutexes[__MAX_NUM_THREADS];
    static mTenConvThreadData convThreadDataObj;
    static bool isConvThreadInitialized;

    static pthread_mutex_t printMutex;
    #ifdef __DEBUG_MTEN_OFF
        static pthread_mutex_t threadLockMutex;
    #endif

    // Assumes vectorization.
    // Check Index before using.
    inline float* getVectorPtr (int bIdx, int vIdx, int hIdx, int wIdx) 
    {
        return (tensorPtr + (bIdx*vecNum*height*width + vIdx*height*width + hIdx*width + wIdx)*vecSize);
    }
    // Assumes vectorization.
    // Check Index before using.
    inline float* getVectorPtr (int bIdx, int cIdx, int hIdx, int wIdx, bool fromCidx) 
    {
        return fromCidx? getVectorPtr(bIdx, (cIdx/vecSize), hIdx, wIdx) : getVectorPtr(bIdx, cIdx, hIdx, wIdx);
    }
    // Check Index before using.
    float* getTensorPtr (int bIdx, int cIdx, int hIdx, int wIdx)
    {
        if (isNHWC)
        {
            if (vecSize)
            {
                return (tensorPtr + (bIdx*vecNum*height*width + (cIdx/vecSize)*height*width + hIdx*width + wIdx)*vecSize + (cIdx%vecSize));
            }
            else
            {
                return (tensorPtr + (bIdx*height*width + hIdx*width + wIdx)*channels + cIdx);
            }   
        }
        else
        {
            return (tensorPtr + (bIdx*channels*height + cIdx*height+ hIdx)*width + wIdx);
        }
    }
    // Check Index before using.
    void setTensorVal (float val, int bIdx, int cIdx, int hIdx, int wIdx)
    {
        *(getTensorPtr(bIdx, cIdx, hIdx, wIdx)) = val;
    }

public:
    static int mTen_num_threads;
    mTen();
    mTen(int blocksIn, int channelsIn, int heightIn, int widthIn);
    mTen(mTen& old);
    // Assumes the input to be in NCHW format.
    mTen(float* input, int blocksIn, int channelsIn, int heightIn, int widthIn, bool NHWC = true, int vecSizeIn = __VECTOR_SIZE); 

    mTen &operator=(const mTen &other);
    ~mTen();

    #ifdef __INCLUDE_TORCH
    mTen(torch::Tensor& tensor, bool NHWC = true, int vecSizeIn = __VECTOR_SIZE); 
    bool compareToTorch(torch::Tensor& input, float epsilon = 0.0001);
    #endif

    // Only converts to NHWC. Call vectorize() if vectorizing is needed.
    void NCHWtoNHWC();
    // Includes devectoring. 
    void NHWCtoNCHW();

    // Includes conversion to NHWC.
    void vectorize (int vecSizeIn = __VECTOR_SIZE); 
    // Ouput is in NHWC.
    void deVectorize(); 

    // Performs Vectorized Direct Convolution. Input will be automatically vectorized if not.
    // Output must be a vectorized tensor.
    void convDepth8to8 (mTen& input, mTen& filter, mTen& bias, int padding = 0, int stride = 1);
    void convPointwise8to8 (mTen& input, mTen& filter, mTen& bias, int padding = 0, int stride = 1);
    void conv (mTen& input, mTen& filter, mTen& bias, int padding = 0, int stride = 1);

    // TODO
    void relu(mTen& input);
    void maxpool2by2Kernel(mTen& input);
    void flatten(mTen& input);
    void fc(mTen& input, mTen& filter, mTen& bias);
    // TODO

    void loadData(std::string fileLocation);
    void saveData(std::string fileLocation);

    inline int getVectorNum()
    {
        return vecNum;
    }
    // Returns the length of the vectorized channel dimention.
    inline int getVCLength()
    {
        return vecSize*vecNum;
    }
    inline int getVectorNum (int dataLength, int vectorSize)
    {
        return vectorSize? (dataLength / vectorSize) + ((dataLength % vectorSize) != 0) : 0;
    }
    inline int getVectoredLength (int dataLength, int vectorSize)
    {
        return getVectorNum (dataLength, vectorSize) * vectorSize;
    }
    inline int getVectorSize()
    {
        return vecSize;
    }
    inline bool isVectored()
    {
        return !!vecSize;
    }
    inline int getBlock()
    {
        return blocks;
    }
    inline int getChannels()
    {
        return channels;
    }
    inline int getWidth()
    {
        return width;
    }
    inline int getHeight()
    {
        return height;
    }
    inline int getSize()
    {
        return blocks*channels*width*height;
    }
    inline int getMemSizeInBytes()
    {
        return vecSize? blocks*vecNum*height*width*vecSize*sizeof(float) : blocks*channels*height*width*sizeof(float);
    }
    // Check Index before using.
    float getTensorVal (int bIdx, int cIdx, int hIdx, int wIdx)
    {
        return *(getTensorPtr(bIdx, cIdx, hIdx, wIdx));
    }

    // Expects NCHW input.
    bool compareToFloatArr(float* input, float epsilon = 0.0001, bool NHWCin = false);
    
    void print(int newlineNum = 16);
    void printSize();
};

struct mTenConvThreadData
{
    mTen* input;
    mTen* filter;
    mTen* bias;
    float* outputPtr;
    int cacheRows;
    int kernelRows;
    int padding;
    int stride;
    int* workLoadIndexPtr;
    int* doneFlag;
};

struct mTenVectorizeThreadData
{
    mTen*input;
    float* newTensorPtr;
    int* workLoadIndexPtr;
    int vecSizeIn;
};

struct mTenCompareThreadData
{
    double epsilon;
    mTen* inputMTEN;
    int* indexPtr;
    int mode; // 0: Compare to float, 1: Compare to torch.
    bool NHWCinput;
    void* inputPtr;
    bool* outPtr;
    int* numPtr;
    double* totalPtr;
    int* errNumPtr;
    double* varPtr;
};

