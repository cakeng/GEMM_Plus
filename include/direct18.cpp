/*
*	direct18.cpp
*	Created: 2021-4-29
*	Author: JongSeok Park (cakeng@naver.com)
*/
#include <direct18.h>

#define __COUTB1_DIRECT18 8

direct18ThreadArg direct18::threadArgArr[__MAX_NUM_THREADS];

pthread_mutex_t direct18::runningMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t direct18::runningCondition = PTHREAD_COND_INITIALIZER; 
int direct18::runningThreads = 0;

pthread_t direct18::convThreads[__MAX_NUM_THREADS];
pthread_cond_t direct18::convThreadConditions[__MAX_NUM_THREADS];
pthread_mutex_t direct18::convThreadMutexes[__MAX_NUM_THREADS];
direct18ConvThreadData direct18::convThreadDataObj;
bool direct18::isConvThreadInitialized = false;

pthread_mutex_t direct18::printMutex = PTHREAD_MUTEX_INITIALIZER;
int direct18::direct18_num_threads = std::thread::hardware_concurrency();

direct18::direct18()
{
    freeWhendel = 1;
    blocks = 0;
    channels = 0;
    height = 0;
    width = 0;
    kernelHeight = 0;
    filterPtr = nullptr;
}

direct18::direct18(int blocksIn, int channelsIn, int heightIn, int widthIn)
{
    freeWhendel = 1;
    blocks = blocksIn;
    channels = channelsIn;
    height = heightIn;
    width = widthIn;
    kernelHeight = 1;
    if(posix_memalign((void**)(&filterPtr), sizeof(float32x4_t), (blocks*channels*height*width)*sizeof(float)))
    {
        printf ("DIRECT18 constructer - POSIX memalign failed.");
    }
    bzero(filterPtr, blocks*channels*height*width*sizeof(float));
}

direct18::direct18(float* filterIn, int blocksIn, int channelsIn, int heightIn, int widthIn, bool isNHWC)
{
    freeWhendel = 1;
    blocks = ((blocksIn/__COUTB1_DIRECT18) + ((blocksIn%__COUTB1_DIRECT18) > 0))*__COUTB1_DIRECT18;
    channels = channelsIn;
    height = heightIn;
    width = widthIn;
    kernelHeight = __COUTB1_DIRECT18;
    if (posix_memalign((void**)(&filterPtr), sizeof(float32x4_t), (blocks*channels*height*width)*sizeof(float))) 
    {
        printf ("DIRECT18 constructer - POSIX memalign failed.");
    }
    bzero (filterPtr, blocks*height*width*channels*sizeof(float));

    cinB1 = __COUTB1_DIRECT18;
    while ((height*width*(cinB1 + __COUTB1_DIRECT18)*__COUTB1_DIRECT18 + 2*(cinB1 + __COUTB1_DIRECT18)*10 + 10*__COUTB1_DIRECT18)*sizeof(float) < __DIRECT18_D1_CACHE_SIZE)
    {
        
        cinB1 += __COUTB1_DIRECT18;
        if (cinB1 >= channels)
        {
            cinB1 = channels;
            break;
        }
    }
    while (channels % cinB1)
    {
        cinB1 -= __COUTB1_DIRECT18;
    }

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
                    int bb = b/kernelHeight;
                    int bi = b%kernelHeight;
                    int cb = c/cinB1;
                    int ci = c%cinB1;
                    // Cout - Cin - H - W - CiB - CoB (8)
                    *(filterPtr + bb*(height*width*channels*kernelHeight) + cb*(height*width*cinB1*kernelHeight)
                        + h*(width*cinB1*kernelHeight) + w*cinB1*kernelHeight + ci*kernelHeight + bi) = val;
                }
            }
        }
    }
}

direct18::direct18(int dummy, float* filterIn, int blocksIn, int channelsIn, int heightIn, int widthIn, bool isNHWC)
{
    freeWhendel = 0;
    blocks = ((blocksIn/__COUTB1_DIRECT18) + ((blocksIn%__COUTB1_DIRECT18) > 0))*__COUTB1_DIRECT18;
    channels = channelsIn;
    height = heightIn;
    width = widthIn;
    kernelHeight = __COUTB1_DIRECT18;
    filterPtr = filterIn;
    cinB1 = __COUTB1_DIRECT18;
    while ((height*width*(cinB1 + __COUTB1_DIRECT18)*__COUTB1_DIRECT18 + 2*(cinB1 + __COUTB1_DIRECT18)*10 + 10*__COUTB1_DIRECT18)*sizeof(float) < __DIRECT18_D1_CACHE_SIZE)
    {
        
        cinB1 += __COUTB1_DIRECT18;
        if (cinB1 >= channels)
        {
            cinB1 = channels;
            break;
        }
    }
    while (channels % cinB1)
    {
        cinB1 -= __COUTB1_DIRECT18;
    }
}

direct18::direct18(direct18& old)
{
    freeWhendel = 1;
    blocks = old.blocks;
    channels = old.channels;
    height = old.height;
    width = old.width;
    kernelHeight = old.kernelHeight;
    if (posix_memalign((void**)(&filterPtr), sizeof(float32x4_t), (blocks*channels*height*width)*sizeof(float)))
    {
        printf ("DIRECT18 constructer - POSIX memalign failed.");
    }
    memcpy(filterPtr, old.filterPtr, (blocks*channels*height*width)*sizeof(float));
}

direct18 &direct18::operator=(const direct18 &other)
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
            kernelHeight = other.kernelHeight;
            if (posix_memalign((void**)(&filterPtr), sizeof(float32x4_t), (blocks*channels*height*width)*sizeof(float)))
            {
                printf ("DIRECT18 constructer - POSIX memalign failed.");
            }
            memcpy(filterPtr, other.filterPtr, (blocks*channels*height*width)*sizeof(float));
        }
        else
        {
            blocks = 0;
            channels = 0;
            height = 0;
            width = 0;
            kernelHeight = 0;
        }
    }
    return *this;
}

direct18::~direct18()
{
    if (filterPtr != nullptr && freeWhendel != 0)
    {
        free(filterPtr);
    }
}

void* direct18ConvThreadRoutine(void* threadArg)
{
    struct direct18ThreadArg* threadData = (struct direct18ThreadArg*) threadArg;
    int &id = threadData->id;
    struct direct18ConvThreadData* dataPtr = (struct direct18ConvThreadData*)(threadData->threadDataPtr);
    pthread_mutex_lock(&direct18::convThreadMutexes[id]);
    DIRECT18_THREAD_ROUTINE_START:
    float* input = dataPtr->input;
    float* filter = dataPtr->filter;
    float* bias = dataPtr->bias;
    float* output = dataPtr->output;
    const int cout = dataPtr->cout;
    const int hfil = dataPtr->hfil;
    const int wfil = dataPtr->wfil;
    const int cin = dataPtr->cin;
    const int hin = dataPtr->hin;
    const int win = dataPtr->win;
    const int cinB1 = dataPtr->cinB1;
    const int padding = dataPtr->padding;
    const int stride = dataPtr->stride;
    const int dilation = dataPtr->dilation;
    int* workLoadIndexPtr = dataPtr->workLoadIndexPtr;
    const int workload = dataPtr->workload;
    const int hout = (hin + padding*2 - dilation*(hfil -1) -1)/stride + 1;
    const int wout = (win + padding*2 - dilation*(wfil -1) -1)/stride + 1;
    
    // Main loop.
    while (true)
    {
        const int idx = __atomic_fetch_add(workLoadIndexPtr, 1, __ATOMIC_RELAXED);
        if (idx >= workload)
        {
            #ifdef __DEBUG_DIRECT18_OFF
                pthread_mutex_lock(&direct18::printMutex);
                printf("Calculation done. Thread %d exiting.\n\n", id);
                pthread_mutex_unlock(&direct18::printMutex);
            #endif
            pthread_mutex_lock(&direct18::runningMutex);
            direct18::runningThreads--;
            if (direct18::runningThreads == 0)
            {
                pthread_cond_signal(&direct18::runningCondition); // Signals change of running condition to the main thread.
            }
            pthread_mutex_unlock(&direct18::runningMutex);
            pthread_cond_wait(&direct18::convThreadConditions[id], &direct18::convThreadMutexes[id]); // Thread waits here until the main thread signals convThreadConditions.
            goto DIRECT18_THREAD_ROUTINE_START; // Child threads start with new thread arguments.
        }
        else
        {
            const int coStart = idx * __COUTB1_DIRECT18;
            float* outBase = output + coStart*hout*wout;
            // Computation
            #ifdef __DEBUG_DIRECT18_OFF
                pthread_mutex_lock(&direct18::printMutex);
                printf("Thread %d calculating workload %d, cout: %d ~ %d, hout %d ~ %d (hin %d ~ %d)\n", 
                    id, idx, coStart, coEnd, hoStart, hoEnd-1, hoStart*stride - padding, (hoEnd - 1)*stride - padding + (hfil - 1)*dilation);
                pthread_mutex_unlock(&direct18::printMutex);
            #endif
            if (bias != nullptr)
            {
                const float32x4_t o00 = vld1q_f32(bias + coStart);
                const float32x4_t o01 = vld1q_f32(bias + coStart + 4);
                for (int s = 0; s < hout*wout; s++)
                {
                    vst1q_f32(outBase + s * __COUTB1_DIRECT18, o00);
                    vst1q_f32(outBase + s * __COUTB1_DIRECT18 + 4, o01);
                }
            }  
            for (int cinIdx = 0; cinIdx < cin; cinIdx += cinB1)
            {
                const int k = (cinIdx + cinB1 > cin) ? (cin - cinIdx) : cinB1;
                for (int ho = 0; ho < hout; ho++)
                {
                    if (wout%7==0)
                    {
                        for (int wo = 0; wo < wout; wo += 7)
                        {
                            float32x4_t o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13;
                            o0 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*0);  // W 0, Co 0
                            o1 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*1);  // W 0, Co 4
                            o2 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*2);  // W 1, Co 0
                            o3 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*3);  // W 1, Co 4
                            o4 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*4);  // W 2, Co 0
                            o5 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*5);  // W 2, Co 4
                            o6 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*6);  // W 3, Co 0
                            o7 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*7);  // W 3, Co 4
                            o8 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*8);  // W 4, Co 0
                            o9 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*9);  // W 4, Co 4
                            o10 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*10);// W 5, Co 0
                            o11 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*11);// W 5, Co 4
                            o12 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*12);// W 6, Co 0
                            o13 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*13);// W 6, Co 4
                            for (int hf = 0; hf < hfil; hf++)
                            {
                                const int hi = ho * stride + hf * dilation - padding;
                                if (0 <= hi && hi < hin)
                                {
                                    for (int wf = 0; wf < wfil; wf++)
                                    {
                                        const int wi = wo * stride + wf * dilation - padding;
                                        float* filBase = filter + 
                                            coStart*cin*hfil*wfil + cinIdx*hfil*wfil*__COUTB1_DIRECT18 + hf*wfil*cinB1*__COUTB1_DIRECT18 + wf*cinB1*__COUTB1_DIRECT18;
                                        float* inBase = input +
                                            cinIdx*hin*win + hi*win*__COUTB1_DIRECT18 + wi*__COUTB1_DIRECT18;
                                        #ifdef __DEBUG_DIRECT18_OFF
                                        printf ("Input:\n");
                                        float* in = inBase;
                                        for (int i = 0; i < k; i += __COUTB1_DIRECT18)
                                        {
                                            for (int j = 0; j < __COUTB1_DIRECT18; j++)
                                            {
                                                printf("Row %d:\t", i + j);
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*0));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*1));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*2));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*3));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*4));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*5));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*6));
                                                printf ("\n");  
                                            }
                                            in += hin*win*__COUTB1_DIRECT18;
                                        }
                                        printf ("Filter:\n");
                                        for (int i = 0; i < __COUTB1_DIRECT18; i++)
                                        {
                                            printf("Row %d:\t", i);
                                            for (int j = 0; j < k; j++)
                                            {
                                                printf("%6.3f\t", *(filBase + j*__COUTB1_DIRECT18 + i));
                                            }
                                            printf("\n");
                                        }
                                        printf ("Output:\n");
                                        float* out = outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*0;
                                        for (int i = 0; i < __COUTB1_DIRECT18; i++)
                                        {
                                            printf("Row %d:\t", i);
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*0));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*1));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*2));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*3));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*4));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*5));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*6));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*7));
                                            printf ("\n");
                                        }
                                        printf ("\n");
                                        #endif
                                        for (int kidx = 0; kidx < k; kidx += __COUTB1_DIRECT18)
                                        {
                                            float32x4_t f0 = vld1q_f32(filBase + 4*0); // K 0 Co 0
                                            float32x4_t f1 = vld1q_f32(filBase + 4*1); // K 0 Co 4
                                            float32x4_t f2 = vld1q_f32(filBase + 4*2); // K 1 Co 0
                                            float32x4_t f3 = vld1q_f32(filBase + 4*3); // K 1 Co 4

                                            float32x4_t i0; // K 0 wi 0
                                            float32x4_t i1; // K 4 wi 0
                                            if (wi < 0)
                                            {
                                                i0 = vdupq_n_f32 (0.0);
                                                i1 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i0 = vld1q_f32(inBase + 4*(stride*2*0 + 0)); // K 0 wi 0
                                                i1 = vld1q_f32(inBase + 4*(stride*2*0 + 1)); // K 4 wi 0
                                            }
                                            float32x4_t i2; // K 0 wi 0
                                            float32x4_t i3; // K 4 wi 0
                                            if (wi + stride < 0)
                                            {
                                                i2 = vdupq_n_f32 (0.0);
                                                i3 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i2 = vld1q_f32(inBase + 4*(stride*2*1 + 0)); // K 0 wi 0
                                                i3 = vld1q_f32(inBase + 4*(stride*2*1 + 1)); // K 4 wi 0
                                            }
                                            float32x4_t i4 = vld1q_f32(inBase + 4*(stride*2*2 + 0)); // K 0 wi 2
                                            float32x4_t i5 = vld1q_f32(inBase + 4*(stride*2*2 + 1)); // K 4 wi 2
                                            float32x4_t i6 = vld1q_f32(inBase + 4*(stride*2*3 + 0)); // K 0 wi 3
                                            float32x4_t i7 = vld1q_f32(inBase + 4*(stride*2*3 + 1)); // K 4 wi 3
                                            float32x4_t i8 = vld1q_f32(inBase + 4*(stride*2*4 + 0)); // K 0 wi 4
                                            float32x4_t i9 = vld1q_f32(inBase + 4*(stride*2*4 + 1)); // K 4 wi 4
                                            float32x4_t i10; // K 0 wi 5
                                            float32x4_t i11; // K 0 wi 5
                                            if (wi + 5*stride >= win)
                                            {
                                                i10 = vdupq_n_f32 (0.0);
                                                i11 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i10 = vld1q_f32(inBase + 4*(stride*2*5 + 0)); // K 0 wi 0
                                                i11 = vld1q_f32(inBase + 4*(stride*2*5 + 1)); // K 4 wi 0
                                            }
                                            float32x4_t i12; // K 0 wi 6
                                            float32x4_t i13; // K 0 wi 6
                                            if (wi + 6*stride >= win)
                                            {
                                                i12 = vdupq_n_f32 (0.0);
                                                i13 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i12 = vld1q_f32(inBase + 4*(stride*2*6 + 0)); // K 0 wi 6
                                                i13 = vld1q_f32(inBase + 4*(stride*2*6 + 1)); // K 0 wi 6
                                            }

                                            o0 = vfmaq_laneq_f32 (o0, f0, i0, 0); // Co 0 wi 0 K 0 
                                            o1 = vfmaq_laneq_f32 (o1, f1, i0, 0); // Co 4 wi 0 K 0 
                                            o2 = vfmaq_laneq_f32 (o2, f0, i2, 0); // Co 0 wi 1 K 0 
                                            o3 = vfmaq_laneq_f32 (o3, f1, i2, 0); // Co 4 wi 1 K 0 
                                            o4 = vfmaq_laneq_f32 (o4, f0, i4, 0); // Co 0 wi 2 K 0 
                                            o5 = vfmaq_laneq_f32 (o5, f1, i4, 0); // Co 4 wi 2 K 0 
                                            o6 = vfmaq_laneq_f32 (o6, f0, i6, 0); // Co 0 wi 3 K 0 
                                            o7 = vfmaq_laneq_f32 (o7, f1, i6, 0); // Co 4 wi 3 K 0 
                                            o8 = vfmaq_laneq_f32 (o8, f0, i8, 0); // Co 0 wi 4 K 0 
                                            o9 = vfmaq_laneq_f32 (o9, f1, i8, 0); // Co 4 wi 4 K 0 
                                            o10 = vfmaq_laneq_f32 (o10, f0, i10, 0); // Co 0 wi 5 K 0 
                                            o11 = vfmaq_laneq_f32 (o11, f1, i10, 0); // Co 4 wi 5 K 0 
                                            o12 = vfmaq_laneq_f32 (o12, f0, i12, 0); // Co 0 wi 6 K 0 
                                            o13 = vfmaq_laneq_f32 (o13, f1, i12, 0); // Co 4 wi 6 K 0 

                                            f0 = vld1q_f32(filBase + 4*4); // K 2 Co 0
                                            f1 = vld1q_f32(filBase + 4*5); // K 2 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f2, i0, 1); // Co 0 wi 0 K 1 
                                            o1 = vfmaq_laneq_f32 (o1, f3, i0, 1); // Co 4 wi 0 K 1 
                                            o2 = vfmaq_laneq_f32 (o2, f2, i2, 1); // Co 0 wi 1 K 1 
                                            o3 = vfmaq_laneq_f32 (o3, f3, i2, 1); // Co 4 wi 1 K 1 
                                            o4 = vfmaq_laneq_f32 (o4, f2, i4, 1); // Co 0 wi 2 K 1 
                                            o5 = vfmaq_laneq_f32 (o5, f3, i4, 1); // Co 4 wi 2 K 1 
                                            o6 = vfmaq_laneq_f32 (o6, f2, i6, 1); // Co 0 wi 3 K 1 
                                            o7 = vfmaq_laneq_f32 (o7, f3, i6, 1); // Co 4 wi 3 K 1 
                                            o8 = vfmaq_laneq_f32 (o8, f2, i8, 1); // Co 0 wi 4 K 1 
                                            o9 = vfmaq_laneq_f32 (o9, f3, i8, 1); // Co 4 wi 4 K 1 
                                            o10 = vfmaq_laneq_f32 (o10, f2, i10, 1); // Co 0 wi 5 K 1 
                                            o11 = vfmaq_laneq_f32 (o11, f3, i10, 1); // Co 4 wi 5 K 1 
                                            o12 = vfmaq_laneq_f32 (o12, f2, i12, 1); // Co 0 wi 6 K 1 
                                            o13 = vfmaq_laneq_f32 (o13, f3, i12, 1); // Co 4 wi 6 K 1 

                                            f2 = vld1q_f32(filBase + 4*6); // K 3 Co 0
                                            f3 = vld1q_f32(filBase + 4*7); // K 3 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f0, i0, 2); // Co 0 wi 0 K 2 
                                            o1 = vfmaq_laneq_f32 (o1, f1, i0, 2); // Co 4 wi 0 K 2 
                                            o2 = vfmaq_laneq_f32 (o2, f0, i2, 2); // Co 0 wi 1 K 2 
                                            o3 = vfmaq_laneq_f32 (o3, f1, i2, 2); // Co 4 wi 1 K 2 
                                            o4 = vfmaq_laneq_f32 (o4, f0, i4, 2); // Co 0 wi 2 K 2 
                                            o5 = vfmaq_laneq_f32 (o5, f1, i4, 2); // Co 4 wi 2 K 2 
                                            o6 = vfmaq_laneq_f32 (o6, f0, i6, 2); // Co 0 wi 3 K 2 
                                            o7 = vfmaq_laneq_f32 (o7, f1, i6, 2); // Co 4 wi 3 K 2 
                                            o8 = vfmaq_laneq_f32 (o8, f0, i8, 2); // Co 0 wi 4 K 2 
                                            o9 = vfmaq_laneq_f32 (o9, f1, i8, 2); // Co 4 wi 4 K 2 
                                            o10 = vfmaq_laneq_f32 (o10, f0, i10, 2); // Co 0 wi 5 K 2 
                                            o11 = vfmaq_laneq_f32 (o11, f1, i10, 2); // Co 4 wi 5 K 2 
                                            o12 = vfmaq_laneq_f32 (o12, f0, i12, 2); // Co 0 wi 6 K 2 
                                            o13 = vfmaq_laneq_f32 (o13, f1, i12, 2); // Co 4 wi 6 K 2 

                                            f0 = vld1q_f32(filBase + 4*8); // K 4 Co 0
                                            f1 = vld1q_f32(filBase + 4*9); // K 4 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f2, i0, 3); // Co 0 wi 0 K 3 
                                            o1 = vfmaq_laneq_f32 (o1, f3, i0, 3); // Co 4 wi 0 K 3 
                                            o2 = vfmaq_laneq_f32 (o2, f2, i2, 3); // Co 0 wi 1 K 3 
                                            o3 = vfmaq_laneq_f32 (o3, f3, i2, 3); // Co 4 wi 1 K 3 
                                            o4 = vfmaq_laneq_f32 (o4, f2, i4, 3); // Co 0 wi 2 K 3 
                                            o5 = vfmaq_laneq_f32 (o5, f3, i4, 3); // Co 4 wi 2 K 3 
                                            o6 = vfmaq_laneq_f32 (o6, f2, i6, 3); // Co 0 wi 3 K 3 
                                            o7 = vfmaq_laneq_f32 (o7, f3, i6, 3); // Co 4 wi 3 K 3 
                                            o8 = vfmaq_laneq_f32 (o8, f2, i8, 3); // Co 0 wi 4 K 3 
                                            o9 = vfmaq_laneq_f32 (o9, f3, i8, 3); // Co 4 wi 4 K 3 
                                            o10 = vfmaq_laneq_f32 (o10, f2, i10, 3); // Co 0 wi 5 K 3 
                                            o11 = vfmaq_laneq_f32 (o11, f3, i10, 3); // Co 4 wi 5 K 3 
                                            o12 = vfmaq_laneq_f32 (o12, f2, i12, 3); // Co 0 wi 6 K 3 
                                            o13 = vfmaq_laneq_f32 (o13, f3, i12, 3); // Co 4 wi 6 K 3 

                                            f2 = vld1q_f32(filBase + 4*10); // K 5 Co 0
                                            f3 = vld1q_f32(filBase + 4*11); // K 5 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f0, i1, 0); // Co 0 wi 0 K 0 
                                            o1 = vfmaq_laneq_f32 (o1, f1, i1, 0); // Co 4 wi 0 K 0 
                                            o2 = vfmaq_laneq_f32 (o2, f0, i3, 0); // Co 0 wi 1 K 0 
                                            o3 = vfmaq_laneq_f32 (o3, f1, i3, 0); // Co 4 wi 1 K 0 
                                            o4 = vfmaq_laneq_f32 (o4, f0, i5, 0); // Co 0 wi 2 K 0 
                                            o5 = vfmaq_laneq_f32 (o5, f1, i5, 0); // Co 4 wi 2 K 0 
                                            o6 = vfmaq_laneq_f32 (o6, f0, i7, 0); // Co 0 wi 3 K 0 
                                            o7 = vfmaq_laneq_f32 (o7, f1, i7, 0); // Co 4 wi 3 K 0 
                                            o8 = vfmaq_laneq_f32 (o8, f0, i9, 0); // Co 0 wi 4 K 0 
                                            o9 = vfmaq_laneq_f32 (o9, f1, i9, 0); // Co 4 wi 4 K 0 
                                            o10 = vfmaq_laneq_f32 (o10, f0, i11, 0); // Co 0 wi 5 K 0 
                                            o11 = vfmaq_laneq_f32 (o11, f1, i11, 0); // Co 4 wi 5 K 0 
                                            o12 = vfmaq_laneq_f32 (o12, f0, i13, 0); // Co 0 wi 6 K 0 
                                            o13 = vfmaq_laneq_f32 (o13, f1, i13, 0); // Co 4 wi 6 K 0 

                                            f0 = vld1q_f32(filBase + 4*12); // K 6 Co 0
                                            f1 = vld1q_f32(filBase + 4*13); // K 6 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f2, i1, 1); // Co 0 wi 0 K 1 
                                            o1 = vfmaq_laneq_f32 (o1, f3, i1, 1); // Co 4 wi 0 K 1 
                                            o2 = vfmaq_laneq_f32 (o2, f2, i3, 1); // Co 0 wi 1 K 1 
                                            o3 = vfmaq_laneq_f32 (o3, f3, i3, 1); // Co 4 wi 1 K 1 
                                            o4 = vfmaq_laneq_f32 (o4, f2, i5, 1); // Co 0 wi 2 K 1 
                                            o5 = vfmaq_laneq_f32 (o5, f3, i5, 1); // Co 4 wi 2 K 1 
                                            o6 = vfmaq_laneq_f32 (o6, f2, i7, 1); // Co 0 wi 3 K 1 
                                            o7 = vfmaq_laneq_f32 (o7, f3, i7, 1); // Co 4 wi 3 K 1 
                                            o8 = vfmaq_laneq_f32 (o8, f2, i9, 1); // Co 0 wi 4 K 1 
                                            o9 = vfmaq_laneq_f32 (o9, f3, i9, 1); // Co 4 wi 4 K 1 
                                            o10 = vfmaq_laneq_f32 (o10, f2, i11, 1); // Co 0 wi 5 K 1 
                                            o11 = vfmaq_laneq_f32 (o11, f3, i11, 1); // Co 4 wi 5 K 1 
                                            o12 = vfmaq_laneq_f32 (o12, f2, i13, 1); // Co 0 wi 6 K 1 
                                            o13 = vfmaq_laneq_f32 (o13, f3, i13, 1); // Co 4 wi 6 K 1 

                                            f2 = vld1q_f32(filBase + 4*14); // K 7 Co 0
                                            f3 = vld1q_f32(filBase + 4*15); // K 7 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f0, i1, 2); // Co 0 wi 0 K 2 
                                            o1 = vfmaq_laneq_f32 (o1, f1, i1, 2); // Co 4 wi 0 K 2 
                                            o2 = vfmaq_laneq_f32 (o2, f0, i3, 2); // Co 0 wi 1 K 2 
                                            o3 = vfmaq_laneq_f32 (o3, f1, i3, 2); // Co 4 wi 1 K 2 
                                            o4 = vfmaq_laneq_f32 (o4, f0, i5, 2); // Co 0 wi 2 K 2 
                                            o5 = vfmaq_laneq_f32 (o5, f1, i5, 2); // Co 4 wi 2 K 2 
                                            o6 = vfmaq_laneq_f32 (o6, f0, i7, 2); // Co 0 wi 3 K 2 
                                            o7 = vfmaq_laneq_f32 (o7, f1, i7, 2); // Co 4 wi 3 K 2 
                                            o8 = vfmaq_laneq_f32 (o8, f0, i9, 2); // Co 0 wi 4 K 2 
                                            o9 = vfmaq_laneq_f32 (o9, f1, i9, 2); // Co 4 wi 4 K 2 
                                            o10 = vfmaq_laneq_f32 (o10, f0, i11, 2); // Co 0 wi 5 K 2 
                                            o11 = vfmaq_laneq_f32 (o11, f1, i11, 2); // Co 4 wi 5 K 2 
                                            o12 = vfmaq_laneq_f32 (o12, f0, i13, 2); // Co 0 wi 6 K 2 
                                            o13 = vfmaq_laneq_f32 (o13, f1, i13, 2); // Co 4 wi 6 K 2 

                                            o0 = vfmaq_laneq_f32 (o0, f2, i1, 3); // Co 0 wi 0 K 3 
                                            o1 = vfmaq_laneq_f32 (o1, f3, i1, 3); // Co 4 wi 0 K 3 
                                            o2 = vfmaq_laneq_f32 (o2, f2, i3, 3); // Co 0 wi 1 K 3 
                                            o3 = vfmaq_laneq_f32 (o3, f3, i3, 3); // Co 4 wi 1 K 3 
                                            o4 = vfmaq_laneq_f32 (o4, f2, i5, 3); // Co 0 wi 2 K 3 
                                            o5 = vfmaq_laneq_f32 (o5, f3, i5, 3); // Co 4 wi 2 K 3 
                                            o6 = vfmaq_laneq_f32 (o6, f2, i7, 3); // Co 0 wi 3 K 3 
                                            o7 = vfmaq_laneq_f32 (o7, f3, i7, 3); // Co 4 wi 3 K 3 
                                            o8 = vfmaq_laneq_f32 (o8, f2, i9, 3); // Co 0 wi 4 K 3 
                                            o9 = vfmaq_laneq_f32 (o9, f3, i9, 3); // Co 4 wi 4 K 3 
                                            o10 = vfmaq_laneq_f32 (o10, f2, i11, 3); // Co 0 wi 5 K 3 
                                            o11 = vfmaq_laneq_f32 (o11, f3, i11, 3); // Co 4 wi 5 K 3 
                                            o12 = vfmaq_laneq_f32 (o12, f2, i13, 3); // Co 0 wi 6 K 3 
                                            o13 = vfmaq_laneq_f32 (o13, f3, i13, 3); // Co 4 wi 6 K 3 

                                            filBase += 8*8;
                                            inBase += hin*win*__COUTB1_DIRECT18;
                                        }
                                    }
                                }
                            }
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*0, o0);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*1, o1);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*2, o2);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*3, o3);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*4, o4);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*5, o5);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*6, o6);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*7, o7);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*8, o8);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*9, o9);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*10, o10);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*11, o11);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*12, o12);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*13, o13);
                        }
                    }
                    else
                    {
                        for (int wo = 0; wo < wout; wo += 8)
                        {
                            float32x4_t o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15;
                            o0 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*0);  // W 0, Co 0
                            o1 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*1);  // W 0, Co 4
                            o2 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*2);  // W 1, Co 0
                            o3 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*3);  // W 1, Co 4
                            o4 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*4);  // W 2, Co 0
                            o5 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*5);  // W 2, Co 4
                            o6 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*6);  // W 3, Co 0
                            o7 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*7);  // W 3, Co 4
                            o8 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*8);  // W 4, Co 0
                            o9 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*9);  // W 4, Co 4
                            o10 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*10);// W 5, Co 0
                            o11 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*11);// W 5, Co 4
                            o12 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*12);// W 6, Co 0
                            o13 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*13);// W 6, Co 4
                            o14 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*14);// W 7, Co 0
                            o15 = vld1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*15);// W 7, Co 4
                            for (int hf = 0; hf < hfil; hf++)
                            {
                                const int hi = ho * stride + hf * dilation - padding;
                                if (0 <= hi && hi < hin)
                                {
                                    for (int wf = 0; wf < wfil; wf++)
                                    {
                                        const int wi = wo * stride + wf * dilation - padding;
                                        float* filBase = filter + 
                                            coStart*cin*hfil*wfil + cinIdx*hfil*wfil*__COUTB1_DIRECT18 + hf*wfil*cinB1*__COUTB1_DIRECT18 + wf*cinB1*__COUTB1_DIRECT18;
                                        float* inBase = input +
                                            cinIdx*hin*win + hi*win*__COUTB1_DIRECT18 + wi*__COUTB1_DIRECT18;
                                        #ifdef __DEBUG_DIRECT18_OFF
                                        printf ("Input:\n");
                                        float* in = inBase;
                                        for (int i = 0; i < k; i += __COUTB1_DIRECT18)
                                        {
                                            for (int j = 0; j < __COUTB1_DIRECT18; j++)
                                            {
                                                printf("Row %d:\t", i + j);
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*0));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*1));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*2));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*3));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*4));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*5));
                                                printf("%6.3f\t", *(in + j + __COUTB1_DIRECT18*6));
                                                printf ("\n");  
                                            }
                                            in += hin*win*__COUTB1_DIRECT18;
                                        }
                                        printf ("Filter:\n");
                                        for (int i = 0; i < __COUTB1_DIRECT18; i++)
                                        {
                                            printf("Row %d:\t", i);
                                            for (int j = 0; j < k; j++)
                                            {
                                                printf("%6.3f\t", *(filBase + j*__COUTB1_DIRECT18 + i));
                                            }
                                            printf("\n");
                                        }
                                        printf ("Output:\n");
                                        float* out = outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*0;
                                        for (int i = 0; i < __COUTB1_DIRECT18; i++)
                                        {
                                            printf("Row %d:\t", i);
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*0));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*1));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*2));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*3));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*4));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*5));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*6));
                                            printf("%6.3f\t", *(out + i + __COUTB1_DIRECT18*7));
                                            printf ("\n");
                                        }
                                        printf ("\n");
                                        #endif
                                        for (int kidx = 0; kidx < k; kidx += __COUTB1_DIRECT18)
                                        {
                                            float32x4_t f0 = vld1q_f32(filBase + 4*0); // K 0 Co 0
                                            float32x4_t f1 = vld1q_f32(filBase + 4*1); // K 0 Co 4
                                            float32x4_t f2 = vld1q_f32(filBase + 4*2); // K 1 Co 0
                                            float32x4_t f3 = vld1q_f32(filBase + 4*3); // K 1 Co 4

                                            float32x4_t i0; // K 0 wi 0
                                            float32x4_t i1; // K 4 wi 0
                                            if (wi < 0)
                                            {
                                                i0 = vdupq_n_f32 (0.0);
                                                i1 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i0 = vld1q_f32(inBase + 4*(stride*2*0 + 0)); // K 0 wi 0
                                                i1 = vld1q_f32(inBase + 4*(stride*2*0 + 1)); // K 4 wi 0
                                            }
                                            float32x4_t i2; // K 0 wi 0
                                            float32x4_t i3; // K 4 wi 0
                                            if (wi + stride < 0)
                                            {
                                                i2 = vdupq_n_f32 (0.0);
                                                i3 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i2 = vld1q_f32(inBase + 4*(stride*2*1 + 0)); // K 0 wi 0
                                                i3 = vld1q_f32(inBase + 4*(stride*2*1 + 1)); // K 4 wi 0
                                            }
                                            float32x4_t i4; // K 0 wi 2
                                            float32x4_t i5; // K 4 wi 2
                                            if (wi + stride < 0)
                                            {
                                                i4 = vdupq_n_f32 (0.0);
                                                i5 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i4 = vld1q_f32(inBase + 4*(stride*2*2 + 0)); // K 0 wi 0
                                                i5 = vld1q_f32(inBase + 4*(stride*2*2 + 1)); // K 4 wi 0
                                            }
                                            float32x4_t i6 = vld1q_f32(inBase + 4*(stride*2*3 + 0)); // K 0 wi 3
                                            float32x4_t i7 = vld1q_f32(inBase + 4*(stride*2*3 + 1)); // K 4 wi 3
                                            float32x4_t i8 = vld1q_f32(inBase + 4*(stride*2*4 + 0)); // K 0 wi 4
                                            float32x4_t i9 = vld1q_f32(inBase + 4*(stride*2*4 + 1)); // K 4 wi 4
                                            float32x4_t i10; // K 0 wi 5
                                            float32x4_t i11; // K 0 wi 5
                                            if (wi + 5*stride >= win)
                                            {
                                                i10 = vdupq_n_f32 (0.0);
                                                i11 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i10 = vld1q_f32(inBase + 4*(stride*2*5 + 0)); // K 0 wi 0
                                                i11 = vld1q_f32(inBase + 4*(stride*2*5 + 1)); // K 4 wi 0
                                            }
                                            float32x4_t i12; // K 0 wi 6
                                            float32x4_t i13; // K 0 wi 6
                                            if (wi + 6*stride >= win)
                                            {
                                                i12 = vdupq_n_f32 (0.0);
                                                i13 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i12 = vld1q_f32(inBase + 4*(stride*2*6 + 0)); // K 0 wi 6
                                                i13 = vld1q_f32(inBase + 4*(stride*2*6 + 1)); // K 0 wi 6
                                            }
                                            float32x4_t i14; // K 0 wi 6
                                            float32x4_t i15; // K 0 wi 6
                                            if (wi + 7*stride >= win)
                                            {
                                                i14 = vdupq_n_f32 (0.0);
                                                i15 = vdupq_n_f32 (0.0);
                                            }
                                            else
                                            {
                                                i14 = vld1q_f32(inBase + 4*(stride*2*7 + 0)); // K 0 wi 6
                                                i15 = vld1q_f32(inBase + 4*(stride*2*7 + 1)); // K 0 wi 6
                                            }

                                            o0 = vfmaq_laneq_f32 (o0, f0, i0, 0); // Co 0 wi 0 K 0 
                                            o1 = vfmaq_laneq_f32 (o1, f1, i0, 0); // Co 4 wi 0 K 0 
                                            o2 = vfmaq_laneq_f32 (o2, f0, i2, 0); // Co 0 wi 1 K 0 
                                            o3 = vfmaq_laneq_f32 (o3, f1, i2, 0); // Co 4 wi 1 K 0 
                                            o4 = vfmaq_laneq_f32 (o4, f0, i4, 0); // Co 0 wi 2 K 0 
                                            o5 = vfmaq_laneq_f32 (o5, f1, i4, 0); // Co 4 wi 2 K 0 
                                            o6 = vfmaq_laneq_f32 (o6, f0, i6, 0); // Co 0 wi 3 K 0 
                                            o7 = vfmaq_laneq_f32 (o7, f1, i6, 0); // Co 4 wi 3 K 0 
                                            o8 = vfmaq_laneq_f32 (o8, f0, i8, 0); // Co 0 wi 4 K 0 
                                            o9 = vfmaq_laneq_f32 (o9, f1, i8, 0); // Co 4 wi 4 K 0 
                                            o10 = vfmaq_laneq_f32 (o10, f0, i10, 0); // Co 0 wi 5 K 0 
                                            o11 = vfmaq_laneq_f32 (o11, f1, i10, 0); // Co 4 wi 5 K 0 
                                            o12 = vfmaq_laneq_f32 (o12, f0, i12, 0); // Co 0 wi 6 K 0 
                                            o13 = vfmaq_laneq_f32 (o13, f1, i12, 0); // Co 4 wi 6 K 0 
                                            o14 = vfmaq_laneq_f32 (o14, f0, i14, 0); // Co 0 wi 7 K 0 
                                            o15 = vfmaq_laneq_f32 (o15, f1, i14, 0); // Co 4 wi 7 K 0 

                                            f0 = vld1q_f32(filBase + 4*4); // K 2 Co 0
                                            f1 = vld1q_f32(filBase + 4*5); // K 2 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f2, i0, 1); // Co 0 wi 0 K 1 
                                            o1 = vfmaq_laneq_f32 (o1, f3, i0, 1); // Co 4 wi 0 K 1 
                                            o2 = vfmaq_laneq_f32 (o2, f2, i2, 1); // Co 0 wi 1 K 1 
                                            o3 = vfmaq_laneq_f32 (o3, f3, i2, 1); // Co 4 wi 1 K 1 
                                            o4 = vfmaq_laneq_f32 (o4, f2, i4, 1); // Co 0 wi 2 K 1 
                                            o5 = vfmaq_laneq_f32 (o5, f3, i4, 1); // Co 4 wi 2 K 1 
                                            o6 = vfmaq_laneq_f32 (o6, f2, i6, 1); // Co 0 wi 3 K 1 
                                            o7 = vfmaq_laneq_f32 (o7, f3, i6, 1); // Co 4 wi 3 K 1 
                                            o8 = vfmaq_laneq_f32 (o8, f2, i8, 1); // Co 0 wi 4 K 1 
                                            o9 = vfmaq_laneq_f32 (o9, f3, i8, 1); // Co 4 wi 4 K 1 
                                            o10 = vfmaq_laneq_f32 (o10, f2, i10, 1); // Co 0 wi 5 K 1 
                                            o11 = vfmaq_laneq_f32 (o11, f3, i10, 1); // Co 4 wi 5 K 1 
                                            o12 = vfmaq_laneq_f32 (o12, f2, i12, 1); // Co 0 wi 6 K 1 
                                            o13 = vfmaq_laneq_f32 (o13, f3, i12, 1); // Co 4 wi 6 K 1 
                                            o14 = vfmaq_laneq_f32 (o14, f2, i14, 1); // Co 0 wi 7 K 1
                                            o15 = vfmaq_laneq_f32 (o15, f3, i14, 1); // Co 4 wi 7 K 1 

                                            f2 = vld1q_f32(filBase + 4*6); // K 3 Co 0
                                            f3 = vld1q_f32(filBase + 4*7); // K 3 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f0, i0, 2); // Co 0 wi 0 K 2 
                                            o1 = vfmaq_laneq_f32 (o1, f1, i0, 2); // Co 4 wi 0 K 2 
                                            o2 = vfmaq_laneq_f32 (o2, f0, i2, 2); // Co 0 wi 1 K 2 
                                            o3 = vfmaq_laneq_f32 (o3, f1, i2, 2); // Co 4 wi 1 K 2 
                                            o4 = vfmaq_laneq_f32 (o4, f0, i4, 2); // Co 0 wi 2 K 2 
                                            o5 = vfmaq_laneq_f32 (o5, f1, i4, 2); // Co 4 wi 2 K 2 
                                            o6 = vfmaq_laneq_f32 (o6, f0, i6, 2); // Co 0 wi 3 K 2 
                                            o7 = vfmaq_laneq_f32 (o7, f1, i6, 2); // Co 4 wi 3 K 2 
                                            o8 = vfmaq_laneq_f32 (o8, f0, i8, 2); // Co 0 wi 4 K 2 
                                            o9 = vfmaq_laneq_f32 (o9, f1, i8, 2); // Co 4 wi 4 K 2 
                                            o10 = vfmaq_laneq_f32 (o10, f0, i10, 2); // Co 0 wi 5 K 2 
                                            o11 = vfmaq_laneq_f32 (o11, f1, i10, 2); // Co 4 wi 5 K 2 
                                            o12 = vfmaq_laneq_f32 (o12, f0, i12, 2); // Co 0 wi 6 K 2 
                                            o13 = vfmaq_laneq_f32 (o13, f1, i12, 2); // Co 4 wi 6 K 2 
                                            o14 = vfmaq_laneq_f32 (o14, f0, i14, 2); // Co 0 wi 7 K 2 
                                            o15 = vfmaq_laneq_f32 (o15, f1, i14, 2); // Co 4 wi 7 K 2 

                                            f0 = vld1q_f32(filBase + 4*8); // K 4 Co 0
                                            f1 = vld1q_f32(filBase + 4*9); // K 4 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f2, i0, 3); // Co 0 wi 0 K 3 
                                            o1 = vfmaq_laneq_f32 (o1, f3, i0, 3); // Co 4 wi 0 K 3 
                                            o2 = vfmaq_laneq_f32 (o2, f2, i2, 3); // Co 0 wi 1 K 3 
                                            o3 = vfmaq_laneq_f32 (o3, f3, i2, 3); // Co 4 wi 1 K 3 
                                            o4 = vfmaq_laneq_f32 (o4, f2, i4, 3); // Co 0 wi 2 K 3 
                                            o5 = vfmaq_laneq_f32 (o5, f3, i4, 3); // Co 4 wi 2 K 3 
                                            o6 = vfmaq_laneq_f32 (o6, f2, i6, 3); // Co 0 wi 3 K 3 
                                            o7 = vfmaq_laneq_f32 (o7, f3, i6, 3); // Co 4 wi 3 K 3 
                                            o8 = vfmaq_laneq_f32 (o8, f2, i8, 3); // Co 0 wi 4 K 3 
                                            o9 = vfmaq_laneq_f32 (o9, f3, i8, 3); // Co 4 wi 4 K 3 
                                            o10 = vfmaq_laneq_f32 (o10, f2, i10, 3); // Co 0 wi 5 K 3 
                                            o11 = vfmaq_laneq_f32 (o11, f3, i10, 3); // Co 4 wi 5 K 3 
                                            o12 = vfmaq_laneq_f32 (o12, f2, i12, 3); // Co 0 wi 6 K 3 
                                            o13 = vfmaq_laneq_f32 (o13, f3, i12, 3); // Co 4 wi 6 K 3 
                                            o14 = vfmaq_laneq_f32 (o14, f2, i14, 3); // Co 0 wi 7 K 3
                                            o15 = vfmaq_laneq_f32 (o15, f3, i14, 3); // Co 4 wi 7 K 3 

                                            f2 = vld1q_f32(filBase + 4*10); // K 5 Co 0
                                            f3 = vld1q_f32(filBase + 4*11); // K 5 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f0, i1, 0); // Co 0 wi 0 K 0 
                                            o1 = vfmaq_laneq_f32 (o1, f1, i1, 0); // Co 4 wi 0 K 0 
                                            o2 = vfmaq_laneq_f32 (o2, f0, i3, 0); // Co 0 wi 1 K 0 
                                            o3 = vfmaq_laneq_f32 (o3, f1, i3, 0); // Co 4 wi 1 K 0 
                                            o4 = vfmaq_laneq_f32 (o4, f0, i5, 0); // Co 0 wi 2 K 0 
                                            o5 = vfmaq_laneq_f32 (o5, f1, i5, 0); // Co 4 wi 2 K 0 
                                            o6 = vfmaq_laneq_f32 (o6, f0, i7, 0); // Co 0 wi 3 K 0 
                                            o7 = vfmaq_laneq_f32 (o7, f1, i7, 0); // Co 4 wi 3 K 0 
                                            o8 = vfmaq_laneq_f32 (o8, f0, i9, 0); // Co 0 wi 4 K 0 
                                            o9 = vfmaq_laneq_f32 (o9, f1, i9, 0); // Co 4 wi 4 K 0 
                                            o10 = vfmaq_laneq_f32 (o10, f0, i11, 0); // Co 0 wi 5 K 0 
                                            o11 = vfmaq_laneq_f32 (o11, f1, i11, 0); // Co 4 wi 5 K 0 
                                            o12 = vfmaq_laneq_f32 (o12, f0, i13, 0); // Co 0 wi 6 K 0 
                                            o13 = vfmaq_laneq_f32 (o13, f1, i13, 0); // Co 4 wi 6 K 0 
                                            o14 = vfmaq_laneq_f32 (o14, f0, i15, 0); // Co 0 wi 7 K 0 
                                            o15 = vfmaq_laneq_f32 (o15, f1, i15, 0); // Co 4 wi 7 K 0 

                                            f0 = vld1q_f32(filBase + 4*12); // K 6 Co 0
                                            f1 = vld1q_f32(filBase + 4*13); // K 6 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f2, i1, 1); // Co 0 wi 0 K 1 
                                            o1 = vfmaq_laneq_f32 (o1, f3, i1, 1); // Co 4 wi 0 K 1 
                                            o2 = vfmaq_laneq_f32 (o2, f2, i3, 1); // Co 0 wi 1 K 1 
                                            o3 = vfmaq_laneq_f32 (o3, f3, i3, 1); // Co 4 wi 1 K 1 
                                            o4 = vfmaq_laneq_f32 (o4, f2, i5, 1); // Co 0 wi 2 K 1 
                                            o5 = vfmaq_laneq_f32 (o5, f3, i5, 1); // Co 4 wi 2 K 1 
                                            o6 = vfmaq_laneq_f32 (o6, f2, i7, 1); // Co 0 wi 3 K 1 
                                            o7 = vfmaq_laneq_f32 (o7, f3, i7, 1); // Co 4 wi 3 K 1 
                                            o8 = vfmaq_laneq_f32 (o8, f2, i9, 1); // Co 0 wi 4 K 1 
                                            o9 = vfmaq_laneq_f32 (o9, f3, i9, 1); // Co 4 wi 4 K 1 
                                            o10 = vfmaq_laneq_f32 (o10, f2, i11, 1); // Co 0 wi 5 K 1 
                                            o11 = vfmaq_laneq_f32 (o11, f3, i11, 1); // Co 4 wi 5 K 1 
                                            o12 = vfmaq_laneq_f32 (o12, f2, i13, 1); // Co 0 wi 6 K 1 
                                            o13 = vfmaq_laneq_f32 (o13, f3, i13, 1); // Co 4 wi 6 K 1 
                                            o14 = vfmaq_laneq_f32 (o14, f2, i15, 1); // Co 0 wi 7 K 1
                                            o15 = vfmaq_laneq_f32 (o15, f3, i15, 1); // Co 4 wi 7 K 1 

                                            f2 = vld1q_f32(filBase + 4*14); // K 7 Co 0
                                            f3 = vld1q_f32(filBase + 4*15); // K 7 Co 4

                                            o0 = vfmaq_laneq_f32 (o0, f0, i1, 2); // Co 0 wi 0 K 2 
                                            o1 = vfmaq_laneq_f32 (o1, f1, i1, 2); // Co 4 wi 0 K 2 
                                            o2 = vfmaq_laneq_f32 (o2, f0, i3, 2); // Co 0 wi 1 K 2 
                                            o3 = vfmaq_laneq_f32 (o3, f1, i3, 2); // Co 4 wi 1 K 2 
                                            o4 = vfmaq_laneq_f32 (o4, f0, i5, 2); // Co 0 wi 2 K 2 
                                            o5 = vfmaq_laneq_f32 (o5, f1, i5, 2); // Co 4 wi 2 K 2 
                                            o6 = vfmaq_laneq_f32 (o6, f0, i7, 2); // Co 0 wi 3 K 2 
                                            o7 = vfmaq_laneq_f32 (o7, f1, i7, 2); // Co 4 wi 3 K 2 
                                            o8 = vfmaq_laneq_f32 (o8, f0, i9, 2); // Co 0 wi 4 K 2 
                                            o9 = vfmaq_laneq_f32 (o9, f1, i9, 2); // Co 4 wi 4 K 2 
                                            o10 = vfmaq_laneq_f32 (o10, f0, i11, 2); // Co 0 wi 5 K 2 
                                            o11 = vfmaq_laneq_f32 (o11, f1, i11, 2); // Co 4 wi 5 K 2 
                                            o12 = vfmaq_laneq_f32 (o12, f0, i13, 2); // Co 0 wi 6 K 2 
                                            o13 = vfmaq_laneq_f32 (o13, f1, i13, 2); // Co 4 wi 6 K 2 
                                            o14 = vfmaq_laneq_f32 (o14, f0, i15, 2); // Co 0 wi 7 K 0 
                                            o15 = vfmaq_laneq_f32 (o15, f1, i15, 2); // Co 4 wi 7 K 0 

                                            o0 = vfmaq_laneq_f32 (o0, f2, i1, 3); // Co 0 wi 0 K 3 
                                            o1 = vfmaq_laneq_f32 (o1, f3, i1, 3); // Co 4 wi 0 K 3 
                                            o2 = vfmaq_laneq_f32 (o2, f2, i3, 3); // Co 0 wi 1 K 3 
                                            o3 = vfmaq_laneq_f32 (o3, f3, i3, 3); // Co 4 wi 1 K 3 
                                            o4 = vfmaq_laneq_f32 (o4, f2, i5, 3); // Co 0 wi 2 K 3 
                                            o5 = vfmaq_laneq_f32 (o5, f3, i5, 3); // Co 4 wi 2 K 3 
                                            o6 = vfmaq_laneq_f32 (o6, f2, i7, 3); // Co 0 wi 3 K 3 
                                            o7 = vfmaq_laneq_f32 (o7, f3, i7, 3); // Co 4 wi 3 K 3 
                                            o8 = vfmaq_laneq_f32 (o8, f2, i9, 3); // Co 0 wi 4 K 3 
                                            o9 = vfmaq_laneq_f32 (o9, f3, i9, 3); // Co 4 wi 4 K 3 
                                            o10 = vfmaq_laneq_f32 (o10, f2, i11, 3); // Co 0 wi 5 K 3 
                                            o11 = vfmaq_laneq_f32 (o11, f3, i11, 3); // Co 4 wi 5 K 3 
                                            o12 = vfmaq_laneq_f32 (o12, f2, i13, 3); // Co 0 wi 6 K 3 
                                            o13 = vfmaq_laneq_f32 (o13, f3, i13, 3); // Co 4 wi 6 K 3 
                                            o14 = vfmaq_laneq_f32 (o14, f2, i15, 3); // Co 0 wi 7 K 1
                                            o15 = vfmaq_laneq_f32 (o15, f3, i15, 3); // Co 4 wi 7 K 1 

                                            filBase += 8*8;
                                            inBase += hin*win*__COUTB1_DIRECT18;
                                        }
                                    }
                                }
                            }
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*0, o0);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*1, o1);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*2, o2);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*3, o3);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*4, o4);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*5, o5);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*6, o6);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*7, o7);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*8, o8);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*9, o9);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*10, o10);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*11, o11);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*12, o12);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*13, o13);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*14, o14);
                            vst1q_f32(outBase + (ho*wout + wo)*__COUTB1_DIRECT18 + 4*15, o15);
                        }
                    }
                }
            }   
        }
    }
}

void direct18::conv (float* input, float* output, float* bias, int hin, int win, int padding, int stride, int dilation)
{
    if(!isConvThreadInitialized)
    {
        for (int i = 0; i < (direct18_num_threads); i++)
        {
            convThreads[i] = pthread_self();
            convThreadConditions[i] = PTHREAD_COND_INITIALIZER; 
            convThreadMutexes[i] = PTHREAD_MUTEX_INITIALIZER;
        }
        convThreadMutexes[direct18_num_threads] = PTHREAD_MUTEX_INITIALIZER;
        isConvThreadInitialized = true;
    }

    if (padding < 0 || stride < 1 || dilation < 1)
    {
        printf("Wrong input parameters - Stride %d, Padding %d, Dilation %d.\n", stride, padding, dilation);
        return;
    }
    if (dilation > 1)
    {
        printf("Direct 18 cannot compute dilation %d.\n", dilation);
        return;
    }

    const int& cout = this->blocks;
    const int& hfil = this->height;
    const int& wfil = this->width;
    const int& cin = this->channels;

    const int hout = (hin + padding*2 - dilation*(hfil -1) -1)/stride + 1;
    const int wout = (win + padding*2 - dilation*(wfil -1) -1)/stride + 1;
    const int sout = hout*wout;

    #ifdef __DEBUG_DIRECT18_OFF
        printf ("Conv - Cout: %d, Cin: %d, hfil: %d, wfil: %d, Hin: %d, Win: %d, Ho: %d, Wo: %d, P: %d, S: %d, D: %d.\n",
            cout, cin, hfil, wfil, hin, win, hout, wout, padding, stride, dilation);
        printf ("Loop dimension report.\n");
        printf ("Cin broken into %d pieces, each %d.\n", (cin/cinB1), cinB1);
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
    convThreadDataObj.hin = hin;
    convThreadDataObj.win = win;
    convThreadDataObj.cinB1 = this->cinB1;
    convThreadDataObj.padding = padding;
    convThreadDataObj.stride = stride;
    convThreadDataObj.dilation = dilation;
    convThreadDataObj.workLoadIndexPtr = &workLoadIdx;
    convThreadDataObj.workload = cout/__COUTB1_DIRECT18;

    //Custom threadpool implementation.
    pthread_mutex_lock(&direct18::runningMutex);
    for (int i = 0; i < (direct18_num_threads); i++)
    {       
        threadArgArr[i].threadDataPtr = (void*)&convThreadDataObj;
        if (pthread_equal(convThreads[i], pthread_self()))
        {
            threadArgArr[i].id = i;
            pthread_create (&convThreads[i], NULL, direct18ConvThreadRoutine, (void* )&threadArgArr[i]);
        }
        else
        {
            pthread_mutex_unlock(&direct18::convThreadMutexes[i]);
            pthread_cond_signal(&convThreadConditions[i]);
        }                
        runningThreads++;
    }
    if (runningThreads > 0)
    {
        pthread_cond_wait(&runningCondition, &runningMutex);
    }
    else
    {
        pthread_mutex_unlock(&direct18::runningMutex);
    }
    for (int i = 0; i < (direct18_num_threads); i++)
    {
        pthread_mutex_lock(&direct18::convThreadMutexes[i]);
    }
    pthread_mutex_unlock(&direct18::runningMutex);
}

float* direct18::vectorize (float* input, int blocks, int channels, int height, int width, bool isNHWC)
{
    float* out;
    int vecChannel = getVectorNum(channels, __COUTB1_DIRECT18)*__COUTB1_DIRECT18;
    if(posix_memalign((void**)(&out), sizeof(float32x4_t), blocks * vecChannel * height * width*sizeof(float)))
    {
        printf ("Vectorize - POSIX memalign failed.");
    }
    for (int b = 0; b < blocks; b++)
    {
        bzero (out + b*vecChannel*height*width + (vecChannel-__COUTB1_DIRECT18)*height*width, __COUTB1_DIRECT18*height*width*sizeof(float));
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
                    int vecIdx = c%__COUTB1_DIRECT18;
                    int vecNum = c/__COUTB1_DIRECT18;
                    *(out + b*vecChannel*height*width + vecNum*height*width*__COUTB1_DIRECT18 + h*width*__COUTB1_DIRECT18 + w*__COUTB1_DIRECT18 + vecIdx) = val;
                }
            }
        }
    }
    return out;
}

float* direct18::deVectorize (float* input, int blocks, int channels, int height, int width)
{
    float* out;
    int vecChannel = getVectorNum(channels, __COUTB1_DIRECT18)*__COUTB1_DIRECT18;
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
                    int vecIdx = c%__COUTB1_DIRECT18;
                    int vecNum = c/__COUTB1_DIRECT18;
                    val = *(input + b*vecChannel*height*width + vecNum*height*width*__COUTB1_DIRECT18 + h*width*__COUTB1_DIRECT18 + w*__COUTB1_DIRECT18 + vecIdx);
                    *(out + b*(height*width*channels) + h*(width*channels) + w*channels + c) = val;
                }
            }
        }
    }
    return out;
}

void direct18::printSize()
{
    printf ("DIRECT18 Filter tensor with Kernel Height %d, Blocks %d, Channels %d, Height %d, Width %d.\n", kernelHeight, blocks, channels, height, width);
}

void direct18::print(int newlineNum, int format)
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
                        int bb = b / kernelHeight;
                        int bi = b % kernelHeight;
                        float val = *(filterPtr + bb*(height*width*channels*kernelHeight) + h*(width*channels*kernelHeight) 
                                + w*(channels*kernelHeight) + c*kernelHeight + bi);
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
                        int bb = b / kernelHeight;
                        int bi = b % kernelHeight;
                        float val = *(filterPtr + bb*(height*width*channels*kernelHeight) + h*(width*channels*kernelHeight) 
                                + w*(channels*kernelHeight) + c*kernelHeight + bi);
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

void* direct18CompareThreadRoutine(void* threadArg)
{
    struct direct18CompareThreadData* dataPtr = (struct direct18CompareThreadData*) threadArg;
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
            pthread_mutex_lock(&direct18::printMutex);
            std::cout << "//////////// Warning!! - Float Comapare check failed!! Under epsilon " << std::fixed << std::setprecision(8) << epsilon << " - At block, channel, height, width index of " << b << ", " << c << ", " << h << ", " << w <<  " ////////////" << std::endl;
            std::cout << "\tValue1: " << std::fixed << std::setprecision(5) << val1 << ", value2: " << val2 << ", Error: " << std::setprecision(8) << error << std::endl;
            pthread_mutex_unlock(&direct18::printMutex);
        }

    }
    pthread_mutex_lock(&direct18::printMutex);
    *(dataPtr->totalPtr) += total; 
    *(dataPtr->numPtr) += num; 
    *(dataPtr->errNumPtr) += errNum; 
    *(dataPtr->varPtr) += var; 
    pthread_mutex_unlock(&direct18::printMutex);
    return nullptr;
}

bool direct18::floatCompare(float* input1, float* input2, int blocks, int channels, int height, int width, double epsilon, bool isNHWC)
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

    pthread_t threads[direct18_num_threads];
    struct direct18CompareThreadData  threadDataArr[direct18_num_threads];

    // Starting threads.
    for (int i = 0; i < direct18_num_threads; i++)
    {
        threadDataArr[i].id = i;
        threadDataArr[i].input1 = input1;
        threadDataArr[i].input2 = input2;
        threadDataArr[i].blocks = blocks;
        threadDataArr[i].channels = channels;
        threadDataArr[i].height = height;
        threadDataArr[i].width = width;
        threadDataArr[i].workloads = blocks*channels*height*width/direct18_num_threads;
        threadDataArr[i].isNHWC = isNHWC;
        threadDataArr[i].epsilon = epsilon;
        threadDataArr[i].totalPtr = &total;
        threadDataArr[i].errNumPtr = &errNum;
        threadDataArr[i].varPtr = &var;
        threadDataArr[i].numPtr = &num;
        pthread_create (&threads[i], NULL, direct18CompareThreadRoutine, (void* )&(threadDataArr[i]));
    }
    for (int i = 0; i < direct18_num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    if(errNum == 0)
    {
        std::cout << "direct18 to Float Arr Compare passed under epsilon " << std::fixed << std::setprecision(6) << epsilon << std::endl;
        std::cout << "Total Number of elements: " << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(10) << sqrt(var/num) << std::endl;
    }
    else
    {
        std::cout << "//////////// Warning!! - direct18 compare to Float Arr failed. ////////////" << std::endl;
        std::cout << "Total Number of elements: " << std::fixed << std::setprecision(6) << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(10) << sqrt(var/num) << std::endl;
    }
    std::cout << std::endl;
    return out;
}
