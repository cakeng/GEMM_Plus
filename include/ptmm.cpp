/*
*	ptmm.cpp
*	Created: 2021-4-29
*	Author: JongSeok Park (cakeng@naver.com)
*/
#include <ptmm.h>

#include <ptmm_kernels_wf1.h>
#include <ptmm_kernels_wf3.h>
#include <ptmm_kernels_wf5.h>
#include <ptmm_kernels_wf3_depth.h>
#include <ptmm_kernels_wf3_depth_split.h>
#include <ptmm_kernels_wf3_ci3.h>
#include <ptmm_kernels_wf7_ci3.h>


ptmmThreadArg ptmm::threadArgArr[__MAX_NUM_THREADS];

pthread_mutex_t ptmm::runningMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t ptmm::runningCondition = PTHREAD_COND_INITIALIZER; 
int ptmm::runningThreads = 0;

pthread_t ptmm::convThreads[__MAX_NUM_THREADS];
pthread_cond_t ptmm::convThreadConditions[__MAX_NUM_THREADS];
pthread_mutex_t ptmm::convThreadMutexes[__MAX_NUM_THREADS];
ptmmConvThreadData ptmm::convThreadDataObj;
bool ptmm::isConvThreadInitialized = false;

pthread_mutex_t ptmm::printMutex = PTHREAD_MUTEX_INITIALIZER;
int ptmm::ptmm_num_threads = std::thread::hardware_concurrency();

ptmm::ptmm()
{
    freeWhendel = 1;
    blocks = 0;
    channels = 0;
    height = 0;
    width = 0;
    filterPtr = nullptr;
}

ptmm::ptmm(int blocksIn, int channelsIn, int heightIn, int widthIn)
{
    freeWhendel = 1;
    blocks = blocksIn;
    channels = channelsIn;
    height = heightIn;
    width = widthIn;
    if(posix_memalign((void**)(&filterPtr), 128, (blocks*channels*height*width)*sizeof(float)))
    {
        printf ("PTMM constructer - POSIX memalign failed.");
    }
    bzero(filterPtr, blocks*channels*height*width*sizeof(float));
}

ptmm::ptmm(float* filterIn, int blocksIn, int channelsIn, int heightIn, int widthIn, bool isNHWC, bool depthwise)
{
    freeWhendel = 1;
    this->depthwise = depthwise;
    if (!depthwise)
    {
        blocks = ((blocksIn/__COUTB1) + ((blocksIn%__COUTB1) > 0))*__COUTB1;
        channels = channelsIn;
        height = heightIn;
        width = widthIn;
        if (posix_memalign((void**)(&filterPtr), 128, (blocks*channels*height*width)*sizeof(float))) 
        {
            printf ("PTMM constructer - POSIX memalign failed.");
        }
        bzero (filterPtr, blocks*height*width*channels*sizeof(float));
        //printf ("Creating ptmm filter with COUTB1 %d.\n", __COUTB1);
        for (int b = 0; b < blocksIn; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0 ; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        float val;
                        if (isNHWC)
                        {
                            val = *(filterIn + b*(height*width*channels) + h*(width*channels) + w*channels + c);
                        }
                        else
                        {
                            val = *(filterIn + b*(channels*height*width) + c*(height*width) + h*width + w);
                        }
                        int bb = b/__COUTB1;
                        int bi = b%__COUTB1;
                        // CoutB2 - H - Cin - W - CoutB1
                        *(filterPtr + bb*(height*width*channels*__COUTB1) + h*(width*channels*__COUTB1) 
                            + c*__COUTB1*width + w*__COUTB1 + bi) = val;
                    }
                }
            }
        }
    }
    else
    {
        blocks = channelsIn;
        channels = channelsIn;
        height = heightIn;
        width = widthIn;
        if (posix_memalign((void**)(&filterPtr), 128, (channels*height*width)*sizeof(float))) 
        {
            printf ("PTMM constructer - POSIX memalign failed.");
        }
        bzero (filterPtr, height*width*channels*sizeof(float));
        //printf ("Creating ptmm filter with COUTB1 %d.\n", __COUTB1);
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0 ; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float val;
                    if (isNHWC)
                    {
                        val = *(filterIn + h*(width*channels) + w*channels + c);
                    }
                    else
                    {
                        val = *(filterIn + c*(height*width) + h*width + w);
                    }
                    int cc = c/__COUTB1;
                    int ci = c%__COUTB1;
                    // CoutB2 - H - Cin - W - CoutB1
                    *(filterPtr + cc*(height*width*__COUTB1) + h*width*__COUTB1 + w*__COUTB1 + ci) = val;
                }
            }
        }
    }
}

ptmm::ptmm(int dummy, float* filterIn, int blocksIn, int channelsIn, int heightIn, int widthIn, bool isNHWC, bool depthwise)
{
    freeWhendel = 0;
    this->depthwise = depthwise;
    if (!depthwise)
    {
        blocks = ((blocksIn/__COUTB1) + ((blocksIn%__COUTB1) > 0))*__COUTB1;
        channels = channelsIn;
        height = heightIn;
        width = widthIn;
        filterPtr = filterIn;
    }
    else
    {
        blocks = channelsIn;
        channels = channelsIn;
        height = heightIn;
        width = widthIn;
        filterPtr = filterIn;
    }
}

ptmm::ptmm(ptmm& old)
{
    freeWhendel = 1;
    blocks = old.blocks;
    channels = old.channels;
    height = old.height;
    width = old.width;
    if (posix_memalign((void**)(&filterPtr), 128, (blocks*channels*height*width)*sizeof(float)))
    {
        printf ("PTMM constructer - POSIX memalign failed.");
    }
    memcpy(filterPtr, old.filterPtr, (blocks*channels*height*width)*sizeof(float));
}

ptmm &ptmm::operator=(const ptmm &other)
{
    freeWhendel = 1;
    if (this != &other)
    {
        if (filterPtr != nullptr)
        {
            free(filterPtr);
        }
        if (other.filterPtr != nullptr)
        {
            blocks = other.blocks;
            channels = other.channels;
            height = other.height;
            width = other.width;
            if (posix_memalign((void**)(&filterPtr), 128, (blocks*channels*height*width)*sizeof(float)))
            {
                printf ("PTMM constructer - POSIX memalign failed.");
            }
            memcpy(filterPtr, other.filterPtr, (blocks*channels*height*width)*sizeof(float));
        }
        else
        {
            blocks = 0;
            channels = 0;
            height = 0;
            width = 0;
        }
    }
    return *this;
}

ptmm::~ptmm()
{
    if (filterPtr != nullptr && freeWhendel != 0)
    {
        free(filterPtr);
    }
}


void* ptmmConvThreadRoutine(void* threadArg)
{
    struct ptmmThreadArg* threadData = (struct ptmmThreadArg*) threadArg;
    int &id = threadData->id;
    struct ptmmConvThreadData* dataPtr = (struct ptmmConvThreadData*)(threadData->threadDataPtr);
    pthread_mutex_lock(&ptmm::convThreadMutexes[id]);
    PTMM_THREAD_ROUTINE_START:
    float* inputPtr = dataPtr->input;
    float* filter = dataPtr->filter;
    float* bias = dataPtr->bias;
    float* outputPtr = dataPtr->output;
    const int cout = dataPtr->cout;
    const int hfil = dataPtr->hfil;
    const int wfil = dataPtr->wfil;
    const int batch = dataPtr->batch;
    const int cin = dataPtr->cin;
    const int hin = dataPtr->hin;
    const int win = dataPtr->win;
    const int cinB1 = dataPtr->cinB1;
    const int coutB2 = dataPtr->coutB2;
    const int houtB1 = dataPtr->houtB1;
    const int padding = dataPtr->padding;
    const int stride = dataPtr->stride;
    const int dilation = dataPtr->dilation;
    const int depthwise = dataPtr->depthwise;
    int* workLoadIndexPtr = dataPtr->workLoadIndexPtr;
    const int workload = dataPtr->workload;
    const int hout = (hin + padding*2 - dilation*(hfil -1) -1)/stride + 1;
    const int wout = (win + padding*2 - dilation*(wfil -1) -1)/stride + 1;

    // Main loop.
    while (true)
    {
        int idx = __atomic_fetch_add(workLoadIndexPtr, 1, __ATOMIC_RELAXED);
        if (idx >= workload)
        {
            #ifdef __DEBUG_PTMM_OFF
                pthread_mutex_lock(&ptmm::printMutex);
                printf("Calculation done. Thread %d exiting.\n\n", id);
                pthread_mutex_unlock(&ptmm::printMutex);
            #endif
            const int runThreads = __atomic_sub_fetch(&ptmm::runningThreads, 1, __ATOMIC_RELAXED);
            if (runThreads == 0)
            {
                pthread_mutex_lock(&ptmm::runningMutex);
                pthread_cond_signal(&ptmm::runningCondition); // Signals change of running condition to the main thread.
                pthread_mutex_unlock(&ptmm::runningMutex);
            }
            pthread_cond_wait(&ptmm::convThreadConditions[id], &ptmm::convThreadMutexes[id]); // Thread waits here until the main thread signals convThreadConditions.
            goto PTMM_THREAD_ROUTINE_START; // Child threads start with new thread arguments.
        }
        else
        {
            if(!depthwise)
            {
                int b = idx / ((hout/houtB1) * (cout/(__COUTB1*coutB2)));
                idx = idx % ((hout/houtB1) * (cout/(__COUTB1*coutB2)));
                const int coStart = (idx / (hout/houtB1))*__COUTB1*coutB2;
                int hoIdx = (idx % (hout/houtB1))*houtB1;
                const int hoEnd = hoIdx + houtB1;
                const int coEnd = coStart + __COUTB1*coutB2;
                float* input = inputPtr + b*hin*win*cin;
                float* output = outputPtr + b*hout*wout*cout;
                //  Bias
                if (bias != nullptr)
                {
                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                    {
                        const float32x4_t b0 = vld1q_f32(bias + coIdx);
                        const float32x4_t b1 = vld1q_f32(bias + coIdx + 4);
                        for (float* ptr = output + hoIdx*wout*__COUTB1 + hout*wout*coIdx; ptr < output + hoEnd*wout*__COUTB1 + hout*wout*coIdx; ptr += __COUTB1)
                        {
                            vst1q_f32(ptr, b0);
                            vst1q_f32(ptr + 4, b1);
                        }
                    }
                }
                // Computation
                #ifdef __DEBUG_PTMM_OFF
                    pthread_mutex_lock(&ptmm::printMutex);
                    printf("Thread %d calculating workload %d, batch %d, cout: %d ~ %d, hout %d ~ %d\n", 
                        id, idx, b, coStart, coEnd, hoIdx, hoEnd-1);
                    pthread_mutex_unlock(&ptmm::printMutex);
                #endif
                if (hfil == 3)
                {
                    if (wout % 7 == 0)
                    {
                        if (stride == 1 && dilation == 1)
                        {
                            if (cin != 3)
                            {
                                for (; hoIdx < hoEnd; hoIdx++)
                                {
                                    for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                                    {
                                        const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                                        if (0 <= hiIdx && hiIdx < hin)
                                        {
                                            int woIdx = 0;
                                            // CoutB2 - H - Cin - W - __COUTB1
                                            float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                            float* in = input + hiIdx*win*__COUTB1;
                                            if (wout == 7)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_7_8_1_1_1_1(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                            }
                                            else
                                            {
                                                if (padding == 1)
                                                {
                                                    const int wiIdx = woIdx*stride - padding;
                                                    #ifdef __DEBUG_PTMM_OFF
                                                        pthread_mutex_lock(&ptmm::printMutex);
                                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                        pthread_mutex_unlock(&ptmm::printMutex);
                                                    #endif
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_7_8_1_1_1_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 7;
                                                }
                                                while (woIdx < wout - 7)
                                                {
                                                    const int wiIdx = woIdx*stride - padding;
                                                    #ifdef __DEBUG_PTMM_OFF
                                                        pthread_mutex_lock(&ptmm::printMutex);
                                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                        pthread_mutex_unlock(&ptmm::printMutex);
                                                    #endif
                                                    if (woIdx + 8 <= wout - 7)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_8_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 8;
                                                    }
                                                    else if (woIdx + 7 <= wout - 7)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_7_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil, 
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 7;
                                                    }
                                                    else if (woIdx + 6 <= wout - 7)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_6_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 6;
                                                    }
                                                    else if (woIdx + 5 <= wout - 7)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_5_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 5;
                                                    }
                                                    else if (woIdx + 4 <= wout - 7)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_4_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 4;
                                                    }
                                                    else if (woIdx + 3 <= wout - 7)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_3_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 3;
                                                    }
                                                    else if (woIdx + 2 <= wout - 7)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_2_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 2;
                                                    }
                                                    else
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_1_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 1;
                                                    }
                                                }
                                                if (padding == 1)
                                                {
                                                    const int wiIdx = woIdx*stride - padding;
                                                    #ifdef __DEBUG_PTMM_OFF
                                                        pthread_mutex_lock(&ptmm::printMutex);
                                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                        pthread_mutex_unlock(&ptmm::printMutex);
                                                    #endif
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_7_8_1_1_0_1(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 7;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (; hoIdx < hoEnd; hoIdx++)
                                {
                                    for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                                    {
                                        const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                                        if (0 <= hiIdx && hiIdx < hin)
                                        {
                                            int woIdx = 0;
                                            // CoutB2 - H - Cin - W - __COUTB1
                                            float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                            float* in = input + hiIdx*win*3;
                                            if (padding == 1)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_7_8_1_1_1_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 7;
                                            }
                                            while (woIdx < wout - 7)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                if (woIdx + 8 <= wout - 7)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_8_8_1_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 8;
                                                }
                                                else if (woIdx + 7 <= wout - 7)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_7_8_1_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil, 
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 7;
                                                }
                                                else if (woIdx + 6 <= wout - 7)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_6_8_1_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 6;
                                                }
                                                else if (woIdx + 5 <= wout - 7)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_5_8_1_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 5;
                                                }
                                                else if (woIdx + 4 <= wout - 7)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_4_8_1_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 4;
                                                }
                                                else if (woIdx + 3 <= wout - 7)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_3_8_1_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 3;
                                                }
                                                else if (woIdx + 2 <= wout - 7)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_2_8_1_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 2;
                                                }
                                                else
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_1_8_1_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 1;
                                                }
                                            }
                                            if (padding == 1)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_7_8_1_1_0_1_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 7;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else if (stride == 1 && dilation == 2)
                        {

                            for (; hoIdx < hoEnd; hoIdx++)
                            {
                                for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                                {
                                    const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                                    if (0 <= hiIdx && hiIdx < hin)
                                    {
                                        int woIdx = 0;
                                        // CoutB2 - H - Cin - W - __COUTB1
                                        float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                        float* in = input + hiIdx*win*__COUTB1;
                                        if (padding == 2)
                                        {
                                            const int wiIdx = woIdx*stride - padding;
                                            #ifdef __DEBUG_PTMM_OFF
                                                pthread_mutex_lock(&ptmm::printMutex);
                                                printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                    id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                pthread_mutex_unlock(&ptmm::printMutex);
                                            #endif
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_3_7_8_1_2_2_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 7;
                                        }
                                        while (woIdx < wout - 7)
                                        {
                                            const int wiIdx = woIdx*stride - padding;
                                            #ifdef __DEBUG_PTMM_OFF
                                                pthread_mutex_lock(&ptmm::printMutex);
                                                printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                                    id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                pthread_mutex_unlock(&ptmm::printMutex);
                                            #endif
                                            if (woIdx + 8 <= wout - 7)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_8_8_1_2_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 8;
                                            }
                                            else if (woIdx + 7 <= wout - 7)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_7_8_1_2_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 7;
                                            }
                                            else if (woIdx + 6 <= wout - 7)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_6_8_1_2_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 6;
                                            }
                                            else if (woIdx + 5 <= wout - 7)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_5_8_1_2_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 5;
                                            }
                                            else if (woIdx + 4 <= wout - 7)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_4_8_1_2_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 4;
                                            }
                                            else if (woIdx + 3 <= wout - 7)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_3_8_1_2_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 3;
                                            }
                                            else if (woIdx + 2 <= wout - 7)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_2_8_1_2_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 2;
                                            }
                                            else
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_1_8_1_2_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 1;
                                            }
                                        }
                                        if (padding == 2)
                                        {
                                            const int wiIdx = woIdx*stride - padding;
                                            #ifdef __DEBUG_PTMM_OFF
                                                pthread_mutex_lock(&ptmm::printMutex);
                                                printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                    id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                pthread_mutex_unlock(&ptmm::printMutex);
                                            #endif
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_3_7_8_1_2_0_2(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 7;
                                        }
                                    }
                                }
                            }

                        }
                        else if (stride == 2 && dilation == 1)
                        {
                            if (cin != 3)
                            {
                                for (; hoIdx < hoEnd; hoIdx++)
                                {
                                    for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                                    {
                                        const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                                        if (0 <= hiIdx && hiIdx < hin)
                                        {
                                            int woIdx = 0;
                                            // CoutB2 - H - Cin - W - __COUTB1
                                            float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                            float* in = input + hiIdx*win*__COUTB1;
                                            if (padding == 1)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_7_8_2_1_1_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 7;
                                            }
                                            while (woIdx < wout)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                if (woIdx + 7 <= wout)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_7_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 7;
                                                }
                                                else if (woIdx + 6 <= wout)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_6_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 6;
                                                }
                                                else if (woIdx + 5 <= wout)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_5_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 5;
                                                }
                                                else if (woIdx + 4 <= wout)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_4_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 4;
                                                }
                                                else if (woIdx + 3 <= wout)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_3_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 3;
                                                }
                                                else if (woIdx + 2 <= wout)
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_2_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 2;
                                                }
                                                else
                                                {
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_1_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (; hoIdx < hoEnd; hoIdx++)
                                {
                                    for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                                    {
                                        const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                                        if (0 <= hiIdx && hiIdx < hin)
                                        {
                                            int woIdx = 0;
                                            // CoutB2 - H - Cin - W - __COUTB1
                                            float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                            float* in = input + hiIdx*win*3;
                                            if (padding == 1)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_7_8_2_1_1_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 7;
                                            }
                                            while (woIdx < wout)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_7_8_2_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 7;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if (stride == 1 && dilation == 1)
                        {
                            if (cin != 3)
                            {
                                for (; hoIdx < hoEnd; hoIdx++)
                                {
                                    for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                                    {
                                        const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                                        if (0 <= hiIdx && hiIdx < hin)
                                        {
                                            int woIdx = 0;
                                            // CoutB2 - H - Cin - W - __COUTB1
                                            float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                            float* in = input + hiIdx*win*__COUTB1;
                                            if (wout == 8 && padding == 1)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_8_8_1_1_1_1(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                            }
                                            else 
                                            {
                                                if (padding == 1)
                                                {
                                                    const int wiIdx = woIdx*stride - padding;
                                                    #ifdef __DEBUG_PTMM_OFF
                                                        pthread_mutex_lock(&ptmm::printMutex);
                                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                        pthread_mutex_unlock(&ptmm::printMutex);
                                                    #endif
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_5_8_1_1_1_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 5;
                                                }
                                                while (woIdx < wout - 5)
                                                {
                                                    const int wiIdx = woIdx*stride - padding;
                                                    #ifdef __DEBUG_PTMM_OFF
                                                        pthread_mutex_lock(&ptmm::printMutex);
                                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                        pthread_mutex_unlock(&ptmm::printMutex);
                                                    #endif
                                                    if (woIdx + 8 <= wout - 5)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_8_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 8;
                                                    }
                                                    else if (woIdx + 7 <= wout - 5)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_7_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil, 
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 7;
                                                    }
                                                    else if (woIdx + 6 <= wout - 5)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_6_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 6;
                                                    }
                                                    else if (woIdx + 5 <= wout - 5)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_5_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 5;
                                                    }
                                                    else if (woIdx + 4 <= wout - 5)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_4_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 4;
                                                    }
                                                    else if (woIdx + 3 <= wout - 5)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_3_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 3;
                                                    }
                                                    else if (woIdx + 2 <= wout - 5)
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_2_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 2;
                                                    }
                                                    else
                                                    {
                                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                        {
                                                            kernel_3_1_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                        }
                                                        woIdx += 1;
                                                    }
                                                }
                                                if (padding == 1)
                                                {
                                                    const int wiIdx = woIdx*stride - padding;
                                                    #ifdef __DEBUG_PTMM_OFF
                                                        pthread_mutex_lock(&ptmm::printMutex);
                                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                        pthread_mutex_unlock(&ptmm::printMutex);
                                                    #endif
                                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                    {
                                                        kernel_3_5_8_1_1_0_1(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                    }
                                                    woIdx += 8;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (; hoIdx < hoEnd; hoIdx++)
                                {
                                    for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                                    {
                                        const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                                        if (0 <= hiIdx && hiIdx < hin)
                                        {
                                            int woIdx = 0;
                                            // CoutB2 - H - Cin - W - __COUTB1
                                            float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                            float* in = input + hiIdx*win*3;
                                            if (padding == 1)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_8_8_1_1_1_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 8;
                                            }
                                            while (woIdx < wout - 8)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_8_8_1_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 8;
                                            }
                                            if (padding == 1)
                                            {
                                                const int wiIdx = woIdx*stride - padding;
                                                #ifdef __DEBUG_PTMM_OFF
                                                    pthread_mutex_lock(&ptmm::printMutex);
                                                    printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                        id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                    pthread_mutex_unlock(&ptmm::printMutex);
                                                #endif
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_8_8_1_1_0_1_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                                output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 8;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else if (stride == 2 && dilation == 1)
                        {
                            for (; hoIdx < hoEnd; hoIdx++)
                            {
                                for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                                {
                                    const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                                    if (0 <= hiIdx && hiIdx < hin)
                                    {
                                        int woIdx = 0;
                                        // CoutB2 - H - Cin - W - __COUTB1
                                        float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                        float* in = input + hiIdx*win*__COUTB1;
                                        if (padding == 1)
                                        {
                                            const int wiIdx = woIdx*stride - padding;
                                            #ifdef __DEBUG_PTMM_OFF
                                                pthread_mutex_lock(&ptmm::printMutex);
                                                printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                    id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                pthread_mutex_unlock(&ptmm::printMutex);
                                            #endif
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_3_4_8_2_1_1_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 4;
                                        }
                                        while (woIdx < wout)
                                        {
                                            const int wiIdx = woIdx*stride - padding;
                                            #ifdef __DEBUG_PTMM_OFF
                                                pthread_mutex_lock(&ptmm::printMutex);
                                                printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                                    id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                                pthread_mutex_unlock(&ptmm::printMutex);
                                            #endif
                                            if (woIdx + 7 <= wout)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_7_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 7;
                                            }
                                            else if (woIdx + 6 <= wout)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_6_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 6;
                                            }
                                            else if (woIdx + 5 <= wout)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_5_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 5;
                                            }
                                            else if (woIdx + 4 <= wout)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_4_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 4;
                                            }
                                            else if (woIdx + 3 <= wout)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_3_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 3;
                                            }
                                            else if (woIdx + 2 <= wout)
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_2_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 2;
                                            }
                                            else
                                            {
                                                for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                                {
                                                    kernel_3_1_8_2_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                            output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                                }
                                                woIdx += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else if (hfil == 1)
                {
                    for (; hoIdx < hoEnd; hoIdx++)
                    {
                        for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                        {
                            const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                            if (0 <= hiIdx && hiIdx < hin)
                            {
                                int woIdx = 0;
                                // CoutB2 - H - Cin - W - __COUTB1
                                float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                float* in = input + hiIdx*win*__COUTB1;
                                while (woIdx < wout)
                                {
                                    const int wiIdx = woIdx*stride - padding;
                                    #ifdef __DEBUG_PTMM_OFF
                                        pthread_mutex_lock(&ptmm::printMutex);
                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                        pthread_mutex_unlock(&ptmm::printMutex);
                                    #endif
                                    if (woIdx + 8 <= wout)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_1_8_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 8;
                                    }
                                    else if (woIdx + 7 <= wout)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_1_7_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 7;
                                    }
                                    else if (woIdx + 6 <= wout)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_1_6_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 6;
                                    }
                                    else if (woIdx + 5 <= wout)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_1_5_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 5;
                                    }
                                    else if (woIdx + 4 <= wout)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_1_4_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 4;
                                    }
                                    else if (woIdx + 3 <= wout)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_1_3_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 3;
                                    }
                                    else if (woIdx + 2 <= wout)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_1_2_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                            woIdx += 2;
                                    }
                                    else
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_1_1_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 1;
                                    }
                                }
                            }    
                        }
                    }
                }
                else if (hfil == 5)
                {
                    for (; hoIdx < hoEnd; hoIdx++)
                    {
                        for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                        {
                            const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                            if (0 <= hiIdx && hiIdx < hin)
                            {
                                int woIdx = 0;
                                // CoutB2 - H - Cin - W - __COUTB1
                                float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                float* in = input + hiIdx*win*__COUTB1;
                                if (wout == 7)
                                {
                                    const int wiIdx = woIdx*stride - padding;
                                    #ifdef __DEBUG_PTMM_OFF
                                        pthread_mutex_lock(&ptmm::printMutex);
                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                        pthread_mutex_unlock(&ptmm::printMutex);
                                    #endif
                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                    {
                                        kernel_5_7_8_1_1_2_2(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                    }
                                }
                                else
                                {
                                    if (padding == 2)
                                    {
                                        const int wiIdx = woIdx*stride - padding;
                                        #ifdef __DEBUG_PTMM_OFF
                                            pthread_mutex_lock(&ptmm::printMutex);
                                            printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                            pthread_mutex_unlock(&ptmm::printMutex);
                                        #endif
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_5_7_8_1_1_2_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 7;
                                    }
                                    while (woIdx < wout - 7)
                                    {
                                        const int wiIdx = woIdx*stride - padding;
                                        #ifdef __DEBUG_PTMM_OFF
                                            pthread_mutex_lock(&ptmm::printMutex);
                                            printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                                id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                            pthread_mutex_unlock(&ptmm::printMutex);
                                        #endif
                                        if (woIdx + 8 <= wout - 7)
                                        {
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_5_8_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 8;
                                        }
                                        else if (woIdx + 7 <= wout - 7)
                                        {
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_5_7_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 7;
                                        }
                                        else if (woIdx + 6 <= wout - 7)
                                        {
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_5_6_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 6;
                                        }
                                        else if (woIdx + 5 <= wout - 7)
                                        {
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_5_5_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 5;
                                        }
                                        else if (woIdx + 4 <= wout - 7)
                                        {
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_5_4_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 4;
                                        }
                                        else if (woIdx + 3 <= wout - 7)
                                        {
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_5_3_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 3;
                                        }
                                        else if (woIdx + 2 <= wout - 7)
                                        {
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_5_2_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 2;
                                        }
                                        else
                                        {
                                            for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                            {
                                                kernel_5_1_8_1_1_0_0(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                            }
                                            woIdx += 1;
                                        }
                                    }
                                    if (padding == 2)
                                    {
                                        const int wiIdx = woIdx*stride - padding;
                                        #ifdef __DEBUG_PTMM_OFF
                                            pthread_mutex_lock(&ptmm::printMutex);
                                            printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                                id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                            pthread_mutex_unlock(&ptmm::printMutex);
                                        #endif
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_5_7_8_1_1_0_2(in + wiIdx*__COUTB1, fil + coIdx * hfil * cin * wfil,  
                                                        output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 7;
                                    }
                                }
                            }    
                        }
                    }
                }
                else if (hfil == 7)
                {
                    for (; hoIdx < hoEnd; hoIdx++)
                    {
                        for (int hfIdx = 0; hfIdx < hfil; hfIdx++)
                        {
                            const int hiIdx = hoIdx*stride - padding + hfIdx * dilation;     
                            if (0 <= hiIdx && hiIdx < hin)
                            {
                                int woIdx = 0;
                                // CoutB2 - H - Cin - W - __COUTB1
                                float* fil = filter + hfIdx * cin * wfil * __COUTB1;
                                float* in = input + hiIdx*win*3;
                                if (padding == 3)
                                {
                                    const int wiIdx = woIdx*stride - padding;
                                    #ifdef __DEBUG_PTMM_OFF
                                        pthread_mutex_lock(&ptmm::printMutex);
                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                        pthread_mutex_unlock(&ptmm::printMutex);
                                    #endif
                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                    {
                                        kernel_7_5_8_2_1_3_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                    }
                                    woIdx += 5;
                                }
                                while (woIdx < wout - 5)
                                {
                                    const int wiIdx = woIdx*stride - padding;
                                    #ifdef __DEBUG_PTMM_OFF
                                        pthread_mutex_lock(&ptmm::printMutex);
                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d\n", 
                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                        pthread_mutex_unlock(&ptmm::printMutex);
                                    #endif
                                    if (woIdx + 5 <= wout - 5)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_7_5_8_2_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 5;
                                    }
                                    else if (woIdx + 4 <= wout - 5)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_7_4_8_2_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 4;
                                    }
                                    else if (woIdx + 3 <= wout - 5)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_7_3_8_2_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 3;
                                    }
                                    else if (woIdx + 2 <= wout - 5)
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_7_2_8_2_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 2;
                                    }
                                    else
                                    {
                                        for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                        {
                                            kernel_7_1_8_2_1_0_0_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                        }
                                        woIdx += 1;
                                    }
                                }
                                if (padding == 3)
                                {
                                    const int wiIdx = woIdx*stride - padding;
                                    #ifdef __DEBUG_PTMM_OFF
                                        pthread_mutex_lock(&ptmm::printMutex);
                                        printf("Thread %d calculating wout %d, hout %d, cout %d ~ %d, hin %d, hfil %d, wfil %d ~ %d, Skip\n", 
                                            id, woIdx, hoIdx, coStart, coEnd, hiIdx, hfIdx, 0, wfil-1);
                                        pthread_mutex_unlock(&ptmm::printMutex);
                                    #endif
                                    for (int coIdx = coStart; coIdx < coEnd; coIdx += __COUTB1)
                                    {
                                        kernel_7_5_8_2_1_0_2_ci3(in + wiIdx*3, fil + coIdx * hfil * cin * wfil,  
                                                    output + hoIdx*wout*__COUTB1 + woIdx*__COUTB1 + hout*wout*coIdx, cin, hin*win*__COUTB1);
                                    }
                                    woIdx += 5;
                                }
                            }
                        }
                    }
                }
            }   
            else
            {
                int b = idx / ((hout/houtB1) * (cout/(__COUTB1*coutB2)));
                idx = idx % ((hout/houtB1) * (cout/(__COUTB1*coutB2)));
                const int coStart = (idx / (hout/houtB1))*__COUTB1*coutB2;
                int hoIdx = (idx % (hout/houtB1))*houtB1;
                const int hoEnd = hoIdx + houtB1;
                const int coEnd = coStart + __COUTB1*coutB2;
                float* input = inputPtr + b*hin*win*cin;
                float* output = outputPtr + b*hout*wout*cout;
                float* biasInput;
                float zeroBias[__COUTB1] = {0};
                if (bias == nullptr)
                {
                    biasInput = &(zeroBias[0]);
                }
                else
                {
                    biasInput = bias + coStart;
                }
                // Computation
                #ifdef __DEBUG_PTMM_OFF
                    pthread_mutex_lock(&ptmm::printMutex);
                    printf("Thread %d calculating workload %d, cout: %d ~ %d, Hout: %d ~ %d\n", 
                        id, idx, coStart, coEnd, hoIdx, hoEnd);
                    pthread_mutex_unlock(&ptmm::printMutex);
                #endif
                if (wfil == 3)
                {
                    if (stride == 1)
                    {
                        if (wout == 7)
                        {
                            kernel_depth_3_7_8_1_1_1_1_1
                            (input + coStart*hin*win, filter + coStart*hfil*wfil, output + coStart*hout*wout, biasInput);
                        }
                        else if (wout == 14)
                        {
                            kernel_depth_3_14_8_1_1_1_1_1
                            (input + coStart*hin*win, filter + coStart*hfil*wfil, output + coStart*hout*wout, biasInput);
                        }
                        else if (wout == 28)
                        {
                            int hiIdx = hoIdx*stride - padding;
                            if (hiIdx < 0)
                            {
                                kernel_depth_3_28_8_1_1_1_1_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, 1);
                                hiIdx += stride;
                                hoIdx += 1;
                                kernel_depth_3_28_8_1_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1-1);
                            }
                            else if (hoEnd == hout)
                            {
                                kernel_depth_3_28_8_1_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1-1);
                                hiIdx += stride * (houtB1-1);
                                hoIdx += houtB1-1;
                                kernel_depth_3_28_8_1_1_1_0_1
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, 1);
                            }
                            else
                            {
                                kernel_depth_3_28_8_1_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1);
                            }
                        }
                        else if (wout == 56)
                        {
                            int hiIdx = hoIdx*stride - padding;
                            if (hiIdx < 0)
                            {
                                kernel_depth_3_56_8_1_1_1_1_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, 1);
                                hiIdx += stride;
                                hoIdx += 1;
                                kernel_depth_3_56_8_1_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1-1);
                            }
                            else if (hoEnd == hout)
                            {
                                kernel_depth_3_56_8_1_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1-1);
                                hiIdx += stride * (houtB1-1);
                                hoIdx += houtB1-1;
                                kernel_depth_3_56_8_1_1_1_0_1
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, 1);
                            }
                            else
                            {
                                kernel_depth_3_56_8_1_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1);
                            }
                        }
                        else if (wout == 112)
                        {
                            // kernel_depth_3_112_8_1_1_1_1_1
                            // (input + coStart*hin*win, filter + coStart*hfil*wfil, output + coStart*hout*wout, biasInput);
                            
                            int hiIdx = hoIdx*stride - padding;
                            if (hiIdx < 0)
                            {
                                kernel_depth_3_112_8_1_1_1_1_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, 1);
                                hiIdx += stride;
                                hoIdx += 1;
                                kernel_depth_3_112_8_1_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1-1);
                            }
                            else if (hoEnd == hout)
                            {
                                kernel_depth_3_112_8_1_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1-1);
                                hiIdx += stride * (houtB1-1);
                                hoIdx += houtB1-1;
                                kernel_depth_3_112_8_1_1_1_0_1
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, 1);
                            }
                            else
                            {
                                kernel_depth_3_112_8_1_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1);
                            }
                        }
                    }
                    else if (stride == 2)
                    {
                        if (wout == 7)
                        {
                            kernel_depth_3_7_8_2_1_1_1_1
                            (input + coStart*hin*win, filter + coStart*hfil*wfil, output + coStart*hout*wout, biasInput);
                        }
                        else if (wout == 14)
                        {
                            kernel_depth_3_14_8_2_1_1_1_1
                            (input + coStart*hin*win, filter + coStart*hfil*wfil, output + coStart*hout*wout, biasInput);
                        }
                        else if (wout == 28)
                        {
                            int hiIdx = hoIdx*stride - padding;
                            if (hiIdx < 0)
                            {
                                kernel_depth_3_28_8_2_1_1_1_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, 1);
                                hiIdx += stride;
                                hoIdx += 1;
                                kernel_depth_3_28_8_2_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1-1);
                            }
                            else
                            {
                                kernel_depth_3_28_8_2_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1);
                            }
                        }
                        else if (wout == 56)
                        {
                            int hiIdx = hoIdx*stride - padding;
                            if (hiIdx < 0)
                            {
                                kernel_depth_3_56_8_2_1_1_1_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, 1);
                                hiIdx += stride;
                                hoIdx += 1;
                                kernel_depth_3_56_8_2_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1-1);
                            }
                            else
                            {
                                kernel_depth_3_56_8_2_1_1_0_0
                                (input + coStart*hin*win + hiIdx*win*__COUTB1, filter + coStart*hfil*wfil, 
                                output + coStart*hout*wout + hoIdx*wout*__COUTB1, biasInput, houtB1);
                            }
                        }
                        
                    }
                }
            }
        }
    }
}

void ptmm::conv (float* input, float* output, float* bias, int batch, int hin, int win, int padding, int stride, int dilation)
{
    if(!isConvThreadInitialized)
    {
        for (int i = 0; i < (ptmm_num_threads); i++)
        {
            convThreads[i] = pthread_self();
            convThreadConditions[i] = PTHREAD_COND_INITIALIZER; 
            convThreadMutexes[i] = PTHREAD_MUTEX_INITIALIZER;
        }
        convThreadMutexes[ptmm_num_threads] = PTHREAD_MUTEX_INITIALIZER;
        isConvThreadInitialized = true;
    }

    if (padding < 0 || stride < 1 || dilation < 1)
    {
        printf("Wrong input parameters - Stride %d, Padding %d, Dilation %d.\n", stride, padding, dilation);
        return;
    }

    const int& cout = this->blocks;
    const int& hfil = this->height;
    const int& wfil = this->width;
    const int& cin = this->channels;

    const int hout = (hin + padding*2 - dilation*(hfil -1) -1)/stride + 1;
    const int wout = (win + padding*2 - dilation*(wfil -1) -1)/stride + 1;
    const int sout = hout*wout;

    int cinB1 = 8;
    int woutB1 = 14;
    int coutB2 = 1;
    int houtB1 = 1;
    if (!this->depthwise)
    {
        while (cinB1*woutB1*sizeof(float) < __PTMM_D1_CACHE_SIZE)
        {
            cinB1++;
            if (cinB1 >= cin)
            {
                cinB1 = cin;
                break;
            }
        }
        while (cin%cinB1)
        {
            cinB1--;
        }

        // while (cinB1*woutB1*sizeof(float) < __PTMM_D1_CACHE_SIZE)
        // {
        //     woutB1++;
        //     if (woutB1 >= wout)
        //     {
        //         woutB1 = wout;
        //         break;
        //     }
        // }
        // while (wout%woutB1)
        // {
        //     woutB1--;
        // }

        while ((cinB1*woutB1*hfil*ptmm_num_threads + cinB1*hfil*wfil*__COUTB1*coutB2)*sizeof(float) < __PTMM_LL_CACHE_SIZE)
        {
            coutB2++;
            if (__COUTB1*coutB2 >= cout)
            {
                coutB2 = cout/__COUTB1;
                break;
            }
        }
        while (cout%(__COUTB1*coutB2))
        {
            coutB2--;
        }
        
    }
    else
    {
        woutB1 = wout;
        houtB1 = hout > 14? 14 : hout;
        coutB2 = 1;
        cinB1 = __COUTB1;
    }
    #ifdef __DEBUG_PTMM_OFF
        if (!this->depthwise)
        {
            int l1CacheUsage = cinB1*woutB1*sizeof(float);
            int llCacheUsage = (cinB1*woutB1*hfil*ptmm_num_threads + cinB1*hfil*wfil*__COUTB1*coutB2)*sizeof(float);
            printf ("Conv - Batch: %d, Cout: %d, Cin: %d, hfil: %d, wfil: %d, Hin: %d, Win: %d, Ho: %d, Wo: %d, P: %d, S: %d, D: %d.\n",
                batch, cout, cin, hfil, wfil, hin, win, hout, wout, padding, stride, dilation);
            printf ("Loop dimension report.\n");
            printf ("Cin broken into %d pieces, each %d.\n", cin/cinB1, cinB1);
            printf ("Wout broken into %d pieces, each %d.\n", wout/woutB1, woutB1);
            printf ("Cout broken into %d pieces, each %d.\n", cout/(__COUTB1*coutB2), __COUTB1*coutB2);
            printf ("L1 Cache usage - %d, %d remaining.\n", l1CacheUsage, __PTMM_D1_CACHE_SIZE - l1CacheUsage);
            printf ("LL Cache usage - %d, %d remaining.\n", llCacheUsage, __PTMM_LL_CACHE_SIZE - llCacheUsage);
            printf ("Total number of workloads: %d, KFLOPs per workload: %d\n", 
                batch * (cout/(__COUTB1*coutB2))*hout, (cout*cin*hfil*wfil*hout*wout/1000)/((cout/(__COUTB1*coutB2))*hout));
        }
        else
        {
            int l1CacheUsage = (cinB1*wout*hout*(1 + stride) + cinB1*hfil*wfil)*sizeof(float);
            int llCacheUsage = l1CacheUsage*ptmm_num_threads;
            printf ("Conv(Depth) - Batch: %d, Cout: %d, Cin: %d, hfil: %d, wfil: %d, Hin: %d, Win: %d, Ho: %d, Wo: %d, P: %d, S: %d, D: %d.\n",
                batch, cout, cin, hfil, wfil, hin, win, hout, wout, padding, stride, dilation);
            printf ("Loop dimension report.\n");
            printf ("Cin broken into %d pieces, each %d.\n", cin/cinB1, cinB1);
            printf ("Wout broken into %d pieces, each %d.\n", wout/woutB1, woutB1);
            printf ("Cout broken into %d pieces, each %d.\n", cout/(__COUTB1*coutB2), __COUTB1*coutB2);
            printf ("L1 Cache usage - %d, %d remaining.\n", l1CacheUsage, __PTMM_D1_CACHE_SIZE - l1CacheUsage);
            printf ("LL Cache usage - %d, %d remaining.\n", llCacheUsage, __PTMM_LL_CACHE_SIZE - llCacheUsage);
            printf ("Total number of workloads: %d, KFLOPs per workload: %d\n", 
                (cout/(__COUTB1*coutB2)), (cin*hfil*wfil*hout*wout/1000)/((cout/(__COUTB1*coutB2))));
        }
    #endif

    int workLoadIdx = 0;
    convThreadDataObj.input = input;
    convThreadDataObj.filter = this->filterPtr;
    convThreadDataObj.bias = bias;
    convThreadDataObj.output = output;
    convThreadDataObj.cout = this->blocks;
    convThreadDataObj.hfil = this->height;
    convThreadDataObj.wfil = this->width;
    convThreadDataObj.cin = this->channels;
    convThreadDataObj.batch = batch;
    convThreadDataObj.hin = hin;
    convThreadDataObj.win = win;
    convThreadDataObj.cinB1 = cinB1;
    convThreadDataObj.coutB2 = coutB2;
    convThreadDataObj.houtB1 = houtB1;
    convThreadDataObj.padding = padding;
    convThreadDataObj.stride = stride;
    convThreadDataObj.dilation = dilation;
    convThreadDataObj.workLoadIndexPtr = &workLoadIdx;
    convThreadDataObj.workload = batch * (cout/(__COUTB1*coutB2))*(hout/houtB1);
    convThreadDataObj.depthwise = this->depthwise;

    //Custom threadpool implementation.
    pthread_mutex_lock(&ptmm::runningMutex);
    runningThreads = ptmm_num_threads;
    for (int i = 0; i < (ptmm_num_threads); i++)
    {       
        threadArgArr[i].threadDataPtr = (void*)&convThreadDataObj;
        if (pthread_equal(convThreads[i], pthread_self()))
        {
            threadArgArr[i].id = i;
            pthread_create (&convThreads[i], NULL, ptmmConvThreadRoutine, (void* )&threadArgArr[i]);
        }
        else
        {
            pthread_mutex_unlock(&ptmm::convThreadMutexes[i]);
            pthread_cond_signal(&convThreadConditions[i]);
        }                
    }
    pthread_cond_wait(&runningCondition, &runningMutex);
    for (int i = 0; i < (ptmm_num_threads); i++)
    {
        pthread_mutex_lock(&ptmm::convThreadMutexes[i]);
    }
    pthread_mutex_unlock(&ptmm::runningMutex);
}


// void *batchNormThreadRoutine(void *arg)
// {
//     struct threadArg *threadArg = (struct threadArg *)arg;
//     const int id = threadArg->id;
//     pthread_mutex_lock(&batchNormThreadMutexes[id]);

// BATCHNORM_RESTART:;
//     struct batchNormThreadData *dataPtr = (struct batchNormThreadData *)(threadArg->tDataPtr);
//     Tensor inputTensor = dataPtr->input;
//     const int cIn = inputTensor.shape[0], hIn = inputTensor.shape[1], wIn = inputTensor.shape[2];
//     float* precalc = dataPtr->precalc.buf, *bias = dataPtr->bias.buf, *mean = dataPtr->mean.buf;

//     int *workloadPtr = dataPtr->workloadPtr;
//     const int idxNum = hIn * wIn;
//     while (1)
//     {
//         // Worker thread fetches new workloads
//         const int idx = __atomic_fetch_add(workloadPtr, __CPU_BATCHNORM_BATCH, __ATOMIC_RELAXED);
//         if (idx >= idxNum)
//         {
//             pthread_mutex_lock(&runningMutex);
//             runningThreads--;
//             if (runningThreads == 0)
//             {
//                 pthread_cond_signal(&runningCondition); // Signals change of running condition to the main thread.
//             }
//             pthread_mutex_unlock(&runningMutex);
//             pthread_cond_wait(&batchNormThreadConditions[id], &batchNormThreadMutexes[id]); // Thread waits here until the main thread signals batchNormThreadConditions[id].
//             goto BATCHNORM_RESTART;                                                 // Worker batchNormThreads start with new thread arguments.
//         }
//         else
//         {
//             #ifdef __DEBUG_CPU_OFF
//             printf("CPU Batchnorm thread %d, idx %d, h: %d, w: %d ~ %d\n", id, idx, (idx/wIn), (idx%wIn), (idx%wIn) + __CPU_BATCHNORM_BATCH);
//             #endif
//             for (int c = 0; c < cIn; c += __CPU_BATCHNORM_CVEC)
//             {
//                 const __m256 calcVec = _mm256_load_ps(precalc + c);
//                 const __m256 biasVec = _mm256_loadu_ps(bias + c);
//                 const __m256 meanVec = _mm256_loadu_ps(mean + c);
//                 for (int i = idx; i < idx + __CPU_BATCHNORM_BATCH; i++)
//                 {
//                     __m256 inVec = _mm256_load_ps(inputTensor.buf + i * cIn + c);
//                     inVec = _mm256_sub_ps(inVec, meanVec);
//                     inVec = _mm256_fmadd_ps(inVec, calcVec, biasVec);
//                     _mm256_store_ps(inputTensor.buf + i * cIn + c, inVec);
//                 }
                
//             }
//         }
//     }
//     return NULL;
// }

// void batchNorm2d(float* input, float* precalc, float* bias, float* mean, int batch, int hin, int win, int cin)
// {
//     if (_num_threads == 0)
//     {
//         _num_threads = std::thread::hardware_concurrency();
//     }
//     int C = input.shape[0], H = input.shape[1], W = input.shape[2];
//     CHECK_ERROR(precalc.shape[0] == C && bias.shape[0] == C && mean.shape[0] == C, "Channel Shape mismatch");
//     CHECK_ERROR(H%__CPU_BATCHNORM_BATCH == 0 && W%__CPU_BATCHNORM_BATCH == 0, "Input Shape mismatch");

//     batchNormTData.input = input;
//     batchNormTData.precalc = precalc;
//     batchNormTData.bias = bias;
//     batchNormTData.mean = mean;
//     int workload = 0;
//     batchNormTData.workloadPtr = &workload;

//     if (!isBatchNormThreadInitialized)
//     {
//         for (int i = 0; i < (_num_threads); i++)
//         {
//             batchNormThreads[i] = pthread_self();
//             batchNormThreadConditions[i] = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
//             batchNormThreadMutexes[i] = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
//         }
//         isBatchNormThreadInitialized = 1;
//     }
//     pthread_mutex_lock(&runningMutex);
//     for (int i = 0; i < _num_threads; i++)
//     {
//         tArg[i].tDataPtr = (void *)&batchNormTData;
//         if (pthread_equal(batchNormThreads[i], pthread_self())) // First time starting the worker batchNormThreads.
//         {
//             tArg[i].id = i;
//             pthread_create(&(batchNormThreads[i]), NULL, batchNormThreadRoutine, (void *)(&(tArg[i])));
//         }
//         else // Restarting worker batchNormThreads
//         {
//             pthread_mutex_unlock(&batchNormThreadMutexes[i]);
//             pthread_cond_signal(&batchNormThreadConditions[i]);
//         }
//         runningThreads++;
//     }
//     if (runningThreads > 0) // Main thread sleeps here.
//     {
//         pthread_cond_wait(&runningCondition, &runningMutex);
//     }
//     else // The last worker thread will wake the main thread.
//     {
//         pthread_mutex_unlock(&runningMutex);
//     }
//     // Main thread collects the mutexes of all worker batchNormThreads. (Makes sure no worker batchNormThreads are left running.)
//     for (int i = 0; i < (_num_threads); i++)
//     {
//         pthread_mutex_lock(&batchNormThreadMutexes[i]);
//     }
//     pthread_mutex_unlock(&runningMutex); 
// }


float* ptmm::vectorize (float* input, int blocks, int channels, int height, int width, int vecSize, bool isNHWC)
{
    float* out;
    int vecChannel = getVectorNum(channels, vecSize)*vecSize;
    if(posix_memalign((void**)(&out), 128, blocks * vecChannel * height * width*sizeof(float)))
    {
        printf ("Vectorize - POSIX memalign failed.");
    }
    for (int b = 0; b < blocks; b++)
    {
        bzero (out + b*vecChannel*height*width + (vecChannel-vecSize)*height*width, vecSize*height*width*sizeof(float));
    }
    for (int b = 0; b < blocks; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0 ; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float val;
                    if (isNHWC)
                    {
                        val = *(input + b*(height*width*channels) + h*(width*channels) + w*channels + c);
                    }
                    else
                    {
                        val = *(input + b*(channels*height*width) + c*(height*width) + h*width + w);
                    }
                    int vecIdx = c%vecSize;
                    int vecNum = c/vecSize;
                    *(out + b*vecChannel*height*width + vecNum*height*width*vecSize + h*width*vecSize + w*vecSize + vecIdx) = val;
                }
            }
        }
    }
    return out;
}

float* ptmm::deVectorize (float* input, int blocks, int channels, int height, int width, int vecSize)
{
    float* out;
    int vecChannel = getVectorNum(channels, vecSize)*vecSize;
    if(posix_memalign((void**)(&out), sizeof(float32x4_t), blocks * channels * height * width*sizeof(float)))
    {
        printf ("Vectorize - POSIX memalign failed.");
    }
    for (int b = 0; b < blocks; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0 ; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float val;
                    int vecIdx = c%vecSize;
                    int vecNum = c/vecSize;
                    val = *(input + b*vecChannel*height*width + vecNum*height*width*vecSize + h*width*vecSize + w*vecSize + vecIdx);
                    *(out + b*(height*width*channels) + h*(width*channels) + w*channels + c) = val;
                }
            }
        }
    }
    return out;
}

void ptmm::printFloat (float* input, int blocks, int channels, int height, int width, int newlineNum, int format, int vecSize)
{
    printf("Printing Float\n");
    if (format == 1)
    {
        int idx = 0;
        int vecChannel = getVectorNum(channels, vecSize)*vecSize;
        for (int b = 0; b < blocks; b++)
        {
            for (int h = 0 ; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        int vecIdx = c%vecSize;
                        int vecNum = c/vecSize;
                        float val = *(input + b*vecChannel*height*width + vecNum*height*width*vecSize + h*width*vecSize + w*vecSize + vecIdx);
                        printf("%6.4f\t", val);
                        if (idx%newlineNum == (newlineNum-1))
                        {
                            printf("- (B:%d, H:%d, W:%d, C:%d)\n", b, h, w, c);
                        }
                        idx++;
                    }
                }
            }
        }
    }
    else if (format == 2)
    {
        int idx = 0;
        int vecChannel = getVectorNum(channels, vecSize)*vecSize;
        for (int b = 0; b < blocks; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0 ; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int vecIdx = c%vecSize;
                        int vecNum = c/vecSize;
                        float val = *(input + b*vecChannel*height*width + vecNum*height*width*vecSize + h*width*vecSize + w*vecSize + vecIdx);
                        printf("%6.4f\t", val);
                        if (idx%newlineNum == (newlineNum-1))
                        {
                            printf("- (B:%d, C:%d, H:%d, W:%d)\n", b, c, h, w);
                        }
                        idx++;
                    }
                }
            }
        }
    }
    else
    {
        int vecChannel = getVectorNum(channels, vecSize)*vecSize;
        for (int idx = 0; idx < blocks*vecChannel*height*width; idx++)
        {
            printf("%6.4f\t", *(input  + idx));
            if (idx%newlineNum == (newlineNum-1))
            {
                printf("- (%d)\n", idx);
            }
        }
        printf("\nEnd of print.\n");
    }
}

void ptmm::printSize()
{
    printf ("PTMM Filter tensor with COUTB1 %d, Blocks %d, Channels %d, Height %d, Width %d.\n", __COUTB1, blocks, channels, height, width);
}

void ptmm::print(int newlineNum, int format)
{
    printf("Printing ");
    printSize();
    if (format == 1)
    {
        int idx = 0;
        for (int b = 0; b < blocks; b++)
        {
            for (int h = 0 ; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        int bb = b / __COUTB1;
                        int bi = b % __COUTB1;
                        float val = *(filterPtr + bb*(height*width*channels*__COUTB1) + h*(width*channels*__COUTB1) 
                                + w*(channels*__COUTB1) + c*__COUTB1 + bi);
                        printf("%6.4f\t", val);
                        if (idx%newlineNum == (newlineNum-1))
                        {
                            printf("- (B:%d, H:%d, W:%d, C:%d)\n", b, h, w, c);
                        }
                        idx++;
                    }
                }
            }
        }
    }
    else if (format == 2)
    {
        int idx = 0;
        for (int b = 0; b < blocks; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0 ; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int bb = b / __COUTB1;
                        int bi = b % __COUTB1;
                        float val = *(filterPtr + bb*(height*width*channels*__COUTB1) + h*(width*channels*__COUTB1) 
                                + w*(channels*__COUTB1) + c*__COUTB1 + bi);
                        printf("%6.4f\t", val);
                        if (idx%newlineNum == (newlineNum-1))
                        {
                            printf("- (B:%d, C:%d, H:%d, W:%d)\n", b, c, h, w);
                        }
                        idx++;
                    }
                }
            }
        }
    }
    else
    {
        for (int idx = 0; idx < blocks*channels*height*width; idx++)
        {
            printf("%6.4f\t", *(filterPtr + idx));
            if (idx%newlineNum == (newlineNum-1))
            {
                printf("- (%d)\n", idx);
            }
        }
        printf("\nEnd of print.\n");
    }
}

void* ptmmCompareThreadRoutine(void* threadArg)
{
    struct ptmmCompareThreadData* dataPtr = (struct ptmmCompareThreadData*) threadArg;
    int id = dataPtr->id;
    float* input1 = dataPtr->input1;
    float* input2 = dataPtr->input2;
    int blocks = dataPtr->blocks;
    int channels = dataPtr->channels;
    int height = dataPtr->height;
    int width = dataPtr->width;
    bool isNHWC = dataPtr->isNHWC;
    double epsilon = dataPtr->epsilon;
    double* totalPtr = dataPtr->totalPtr;
    int* errNumPtr = dataPtr->errNumPtr;
    double* varPtr = dataPtr->varPtr;

    int size = dataPtr->workloads;
    int errNum = 0;
    int num = 0;
    double total = 0;
    double var = 0.0;

    int end = ((id+2)*size > blocks*channels*height*width) ? blocks*channels*height*width : (id+1)*size;
    for (int idx = id*size; idx < end; idx++)
    {
        float val1 = *(input1 + idx);
        float val2 = *(input2 + idx);
        total += val1;
        double error = (double)val1 - (double)val2;
        var += error*error;
        num++;
        if(fabs(error) > epsilon)
        {
            errNum++;
            int b, c, h, w;
            if (isNHWC)
            {
                b = idx/(channels*height*width);
                h = (idx%(channels*height*width))/(channels*width);
                w = (idx%(channels*width))/(channels);
                c = idx%channels;
            }
            else
            {
                b = idx/(channels*height*width);
                c = (idx%(channels*height*width))/(height*width);
                h = (idx%(height*width))/(width);
                w = idx%width;
            }
            pthread_mutex_lock(&ptmm::printMutex);
            std::cout << "//////////// Warning!! - Float Comapare check failed!! Under epsilon " << std::fixed << std::setprecision(8) << epsilon << " - At block, channel, height, width index of " << b << ", " << c << ", " << h << ", " << w <<  " ////////////" << std::endl;
            std::cout << "\tValue1: " << std::fixed << std::setprecision(5) << val1 << ", value2: " << val2 << ", Error: " << std::setprecision(8) << error << std::endl;
            pthread_mutex_unlock(&ptmm::printMutex);
        }

    }
    pthread_mutex_lock(&ptmm::printMutex);
    *(dataPtr->totalPtr) += total; 
    *(dataPtr->numPtr) += num; 
    *(dataPtr->errNumPtr) += errNum; 
    *(dataPtr->varPtr) += var; 
    pthread_mutex_unlock(&ptmm::printMutex);
    return nullptr;
}

bool ptmm::floatCompare(float* input1, float* input2, int blocks, int channels, int height, int width, double epsilon, bool isNHWC)
{
    bool out = true;
    
    if(!out)
    {
        return out;
    }
    int num = 0;
    double total = 0;
    int errNum = 0;
    double var = 0;

    pthread_t threads[ptmm_num_threads];
    struct ptmmCompareThreadData  threadDataArr[ptmm_num_threads];

    // Starting threads.
    for (int i = 0; i < ptmm_num_threads; i++)
    {
        threadDataArr[i].id = i;
        threadDataArr[i].input1 = input1;
        threadDataArr[i].input2 = input2;
        threadDataArr[i].blocks = blocks;
        threadDataArr[i].channels = channels;
        threadDataArr[i].height = height;
        threadDataArr[i].width = width;
        threadDataArr[i].workloads = blocks*channels*height*width/ptmm_num_threads;
        threadDataArr[i].isNHWC = isNHWC;
        threadDataArr[i].epsilon = epsilon;
        threadDataArr[i].totalPtr = &total;
        threadDataArr[i].errNumPtr = &errNum;
        threadDataArr[i].varPtr = &var;
        threadDataArr[i].numPtr = &num;
        pthread_create (&threads[i], NULL, ptmmCompareThreadRoutine, (void* )&(threadDataArr[i]));
    }
    for (int i = 0; i < ptmm_num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    if(errNum == 0)
    {
        std::cout << "ptmm to Float Arr Compare passed under epsilon " << std::fixed << std::setprecision(6) << epsilon << std::endl;
        std::cout << "Total Number of elements: " << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(10) << sqrt(var/num) << std::endl;
    }
    else
    {
        std::cout << "//////////// Warning!! - ptmm compare to Float Arr failed. ////////////" << std::endl;
        std::cout << "Total Number of elements: " << std::fixed << std::setprecision(6) << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(10) << sqrt(var/num) << std::endl;
    }
    std::cout << std::endl;
    return out;
}
