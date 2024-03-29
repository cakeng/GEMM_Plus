/*
*	comp.h 
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

#define __COMP_LL_CACHE_SIZE 1048576 // LL Cache size in bytes.
#define __COMP_D1_CACHE_SIZE 32768   // L1 Data Cache size in bytes.
// #define __COMP_LL_CACHE_SIZE 25000 // LL Cache size in bytes.
// #define __COMP_D1_CACHE_SIZE 6000   // L1 Data Cache size in bytes.
#define __DEBUG_COMP 1

#ifndef __COUTB1_COMP
#define __COUTB1_COMP 8
#endif
#define __BATCH_NUM 1

#include <arm_neon.h>

#define __MAX_NUM_THREADS 12

struct compConvThreadData;
struct compThreadArg
{
   int id;
   void* threadDataPtr;
};

class comp
{
private:
    int blocks, channels, height, width; 
    int kernelHeight, cinB1;
    float* filterPtr;

    friend void* compConvThreadRoutine(void* threadArg);
    friend void* compCompareThreadRoutine(void* threadArg);

    static compThreadArg threadArgArr[__MAX_NUM_THREADS];

    static pthread_cond_t runningCondition;
    static int runningThreads;
    static pthread_mutex_t runningMutex;

    static pthread_t convThreads[__MAX_NUM_THREADS];
    static pthread_cond_t convThreadConditions[__MAX_NUM_THREADS];
    static pthread_mutex_t convThreadMutexes[__MAX_NUM_THREADS];
    static compConvThreadData convThreadDataObj;
    static bool isConvThreadInitialized;

    static pthread_mutex_t printMutex;
    

public:
    static int comp_num_threads;
    
    comp();
    comp(int blocksIn, int channelsIn, int heightIn, int widthIn);
    comp(float* filterIn, int blocksIn, int channelsIn, int heightIn, int widthIn, bool isNHWC = true);
    comp(comp& old);
    comp &operator=(const comp &other);
    ~comp();

    // Performs Vectorized Direct Convolution. Input will be automatically vectorized if not.
    void conv (float* input, float* output, float* bias, int h, int w, int padding = 0, int stride = 1, int dilation = 1);

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
    inline int getMemSizeInBytes()
    {
        return blocks*channels*height*width*sizeof(float);
    }

    static inline int getThreads()
    {
        return comp_num_threads;
    }
    static inline int getVectorNum (int dataLength, int vectorSize)
    {
        return vectorSize? (dataLength / vectorSize) + ((dataLength % vectorSize) != 0) : 0;
    }

    static float* vectorize (float* input, int blocks, int channels, int height, int width, bool isNHWC = false);
    static float* deVectorize (float* input, int blocks, int channels, int height, int width);
    static bool floatCompare(float* input1, float* input2, int blocks, int channels, int height, int width, double epsilon = 0.0001, bool isNHWC = true);

    // Format 0 - RAW, Format 1 - NHWC, Format 2 - NCHW
    void print(int newlineNum = 16, int format = 0);
    void printSize();
};

struct compConvThreadData
{
    float* input;
    float* filter;
    float* bias;
    float* output;
    int cout;
    int hfil;
    int wfil;
    int cin;
    int hin;
    int win;
    int cinB1;
    int houtB1;
    int coutB1;
    int coutB2;
    int woSafeStart;
    int woSafeEnd;
    int padding;
    int stride;
    int dilation;
    int* workLoadIndexPtr;   
    int workload;
};

struct compCompareThreadData
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