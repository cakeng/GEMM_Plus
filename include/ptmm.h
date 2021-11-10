/*
*	ptmm.h 
*	Created: 2021-4-29
*	Author: JongSeok Park (cakeng@naver.com)
*/
#pragma oncemake
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <thread>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <pthread.h>
#include <chrono>

#define __PTMM_LL_CACHE_SIZE 1048576 // LL Cache size in bytes.
#define __PTMM_D1_CACHE_SIZE 32768   // L1 Data Cache size in bytes.
// #define __PTMM_LL_CACHE_SIZE 25000 // LL Cache size in bytes.
// #define __PTMM_D1_CACHE_SIZE 6000   // L1 Data Cache size in bytes.
#define __DEBUG_PTMM 1
//#define __INCLUDE_TORCH 1

#define __MAXCIN 128
#define __VEC_SIZE 4
#define __COUTB1 8
#define __BATCH_NUM 1

#include <arm_neon.h>

#define __MAX_NUM_THREADS 12

struct ptmmConvThreadData;
struct ptmmThreadArg
{
   int id;
   void* threadDataPtr;
};

class ptmm
{
private:
    int blocks, channels, height, width, depthwise; 
    float* filterPtr;
    int freeWhendel;
    
    friend void* ptmmConvThreadRoutine(void* threadArg);
    friend void* ptmmCompareThreadRoutine(void* threadArg);

    static ptmmThreadArg threadArgArr[__MAX_NUM_THREADS];

    static pthread_cond_t runningCondition;
    static int runningThreads;
    static pthread_mutex_t runningMutex;

    static pthread_t convThreads[__MAX_NUM_THREADS];
    static pthread_cond_t convThreadConditions[__MAX_NUM_THREADS];
    static pthread_mutex_t convThreadMutexes[__MAX_NUM_THREADS];
    static ptmmConvThreadData convThreadDataObj;
    static bool isConvThreadInitialized;

    static pthread_mutex_t printMutex;
    

public:
    static int ptmm_num_threads;
    
    ptmm();
    ptmm(int blocksIn, int channelsIn, int heightIn, int widthIn);
    ptmm(float* filterIn, int blocksIn, int channelsIn, int heightIn, int widthIn, bool isNHWC = true, bool depthwise = false);
    ptmm(int dummy, float* filterIn, int blocksIn, int channelsIn, int heightIn, int widthIn, bool isNHWC = true, bool depthwise = false);
    ptmm(ptmm& old);
    ptmm &operator=(const ptmm &other);
    ~ptmm();

    // Performs Vectorized Direct Convolution. Input will be automatically vectorized if not.
    void conv (float* input, float* output, float* bias, int batch, int h, int w, int padding = 0, int stride = 1, int dilation = 1);
    void batchNorm2d(float* input, float* precalc, float* bias, float* mean, int batch, int hin, int win, int cin);

    void loadData(std::string fileLocation);
    void saveData(std::string fileLocation);

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
    float* getFilterPtr()
    {
        return filterPtr;
    }
    inline int getMemSizeInBytes()
    {
        return blocks*channels*height*width*sizeof(float);
    }

    static inline int getThreads()
    {
        return ptmm_num_threads;
    }
    static inline int getVectorNum (int dataLength, int vectorSize)
    {
        return vectorSize? (dataLength / vectorSize) + ((dataLength % vectorSize) != 0) : 0;
    }
    static float* vectorize (float* input, int blocks, int channels, int height, int width, int vecSize, bool isNHWC = false);
    static float* deVectorize (float* input, int blocks, int channels, int height, int width, int vecSize);
    static bool floatCompare(float* input1, float* input2, int blocks, int channels, int height, int width, double epsilon = 0.0001, bool isNHWC = true);
    // Format 0 - RAW, Format 1 - NHWC, Format 2 - NCHW
    static void printFloat(float* input, int blocks, int channels, int height, int width, int newlineNum = 16, int format = 0, int vecSize = __VEC_SIZE);

    // Format 0 - RAW, Format 1 - NHWC, Format 2 - NCHW
    void print(int newlineNum = 16, int format = 0);
    void printSize();
};

struct ptmmConvThreadData
{
    float* input;
    float* filter;
    float* bias;
    float* output;
    int cout;
    int hfil;
    int wfil;
    int batch;
    int cin;
    int hin;
    int win;
    int cinB1;
    int houtB1;
    int coutB2;
    int woSafeStart;
    int woSafeEnd;
    int padding;
    int stride;
    int dilation;
    int* workLoadIndexPtr;   
    int workload;
    int depthwise;
};

struct ptmmCompareThreadData
{
    int id;
    float* input1;
    float* input2;
    int blocks, channels, height, width;
    int workloads;
    bool isNHWC;
    double epsilon;
    double* totalPtr;
    int* errNumPtr;
    double* varPtr;
    int* numPtr;
};

// struct batchNormThreadData
// {
//     Tensor input, precalc, bias, mean;
//     int *workloadPtr;
// };

// struct batchNormThreadData batchNormTData;
// pthread_t batchNormThreads[__MAX_NUM_THREADS];
// pthread_cond_t batchNormThreadConditions[__MAX_NUM_THREADS];
// pthread_mutex_t batchNormThreadMutexes[__MAX_NUM_THREADS];
// int isBatchNormThreadInitialized = 0;

