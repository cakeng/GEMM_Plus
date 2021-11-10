/*
*	comp.cpp
*	Created: 2021-4-29
*	Author: JongSeok Park (cakeng@naver.com)
*/
#include <comp.h>

#define COUTB1 8

compThreadArg comp::threadArgArr[__MAX_NUM_THREADS];

pthread_mutex_t comp::runningMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t comp::runningCondition = PTHREAD_COND_INITIALIZER; 
int comp::runningThreads = 0;

pthread_t comp::convThreads[__MAX_NUM_THREADS];
pthread_cond_t comp::convThreadConditions[__MAX_NUM_THREADS];
pthread_mutex_t comp::convThreadMutexes[__MAX_NUM_THREADS];
compConvThreadData comp::convThreadDataObj;
bool comp::isConvThreadInitialized = false;

pthread_mutex_t comp::printMutex = PTHREAD_MUTEX_INITIALIZER;
int comp::comp_num_threads = std::thread::hardware_concurrency();

comp::comp()
{
    blocks = 0;
    channels = 0;
    height = 0;
    width = 0;
    kernelHeight = 0;
    filterPtr = nullptr;
}

comp::comp(int blocksIn, int channelsIn, int heightIn, int widthIn)
{
    blocks = blocksIn;
    channels = channelsIn;
    height = heightIn;
    width = widthIn;
    kernelHeight = 1;
    if(posix_memalign((void**)(&filterPtr), sizeof(float32x4_t), (blocks*channels*height*width)*sizeof(float)))
    {
        printf ("COMP constructer - POSIX memalign failed.");
    }
    bzero(filterPtr, blocks*channels*height*width*sizeof(float));
}

comp::comp(float* filterIn, int blocksIn, int channelsIn, int heightIn, int widthIn, bool isNHWC)
{
    blocks = ((blocksIn/COUTB1) + ((blocksIn%COUTB1) > 0))*COUTB1;
    channels = channelsIn;
    height = heightIn;
    width = widthIn;
    kernelHeight = COUTB1;
    if (posix_memalign((void**)(&filterPtr), sizeof(float32x4_t), (blocks*channels*height*width)*sizeof(float))) 
    {
        printf ("COMP constructer - POSIX memalign failed.");
    }
    bzero (filterPtr, blocks*height*width*channels*sizeof(float));

    cinB1 = COUTB1;
    while ((height*width*(cinB1 + COUTB1)*COUTB1 + 2*(cinB1 + COUTB1)*10 + 10*COUTB1)*sizeof(float) < __COMP_D1_CACHE_SIZE)
    {
        
        cinB1 += COUTB1;
        if (cinB1 >= channels)
        {
            cinB1 = channels;
            break;
        }
    }
    while (channels % cinB1)
    {
        cinB1 -= COUTB1;
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

comp::comp(comp& old)
{
    blocks = old.blocks;
    channels = old.channels;
    height = old.height;
    width = old.width;
    kernelHeight = old.kernelHeight;
    if (posix_memalign((void**)(&filterPtr), sizeof(float32x4_t), (blocks*channels*height*width)*sizeof(float)))
    {
        printf ("COMP constructer - POSIX memalign failed.");
    }
    memcpy(filterPtr, old.filterPtr, (blocks*channels*height*width)*sizeof(float));
}

comp &comp::operator=(const comp &other)
{
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
                printf ("COMP constructer - POSIX memalign failed.");
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

comp::~comp()
{
    if (filterPtr != nullptr)
    {
        free(filterPtr);
    }
}

void* compConvThreadRoutine(void* threadArg)
{
    struct compThreadArg* threadData = (struct compThreadArg*) threadArg;
    int &id = threadData->id;
    struct compConvThreadData* dataPtr = (struct compConvThreadData*)(threadData->threadDataPtr);
    pthread_mutex_lock(&comp::convThreadMutexes[id]);
    COMP_THREAD_ROUTINE_START:
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
            #ifdef __DEBUG_COMP_OFF
                pthread_mutex_lock(&comp::printMutex);
                printf("Calculation done. Thread %d exiting.\n\n", id);
                pthread_mutex_unlock(&comp::printMutex);
            #endif
            pthread_mutex_lock(&comp::runningMutex);
            comp::runningThreads--;
            if (comp::runningThreads == 0)
            {
                pthread_cond_signal(&comp::runningCondition); // Signals change of running condition to the main thread.
            }
            pthread_mutex_unlock(&comp::runningMutex);
            pthread_cond_wait(&comp::convThreadConditions[id], &comp::convThreadMutexes[id]); // Thread waits here until the main thread signals convThreadConditions.
            goto COMP_THREAD_ROUTINE_START; // Child threads start with new thread arguments.
        }
        else
        {
            const int coStart = idx * COUTB1;
            float* outBase = output + coStart*hout*wout;
            // Computation
            #ifdef __DEBUG_COMP_OFF
                pthread_mutex_lock(&comp::printMutex);
                printf("Thread %d calculating workload %d, cout: %d ~ %d, hout: %d, wout: %d\n", 
                    id, idx, coStart,  coStart + COUTB1 - 1, hout, wout);
                pthread_mutex_unlock(&comp::printMutex);
            #endif
            const float32x4_t o00 = vld1q_f32(bias + coStart);
            const float32x4_t o01 = vld1q_f32(bias + coStart + 4);
            for (int s = 0; s < hout*wout; s++)
            {
                vst1q_f32(outBase + s * COUTB1, o00);
                vst1q_f32(outBase + s * COUTB1 + 4, o01);
            }
            for (int cinIdx = 0; cinIdx < cin; cinIdx += cinB1)
            {
                const int k = (cinIdx + cinB1 > cin) ? (cin - cinIdx) : cinB1;
                for (int ho = 0; ho < hout; ho++)
                {
                    for (int wo = 0; wo < wout; wo += 7)
                    {
                        float32x4_t o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13;
                        for (int hf = 0; hf < hfil; hf++)
                        {
                            const int hi = ho * stride + hf * dilation - padding;
                            if (0 <= hi && hi < hin)
                            {
                                for (int wf = 0; wf < wfil; wf++)
                                {
                                    const int wi = wo * stride + wf * dilation - padding;
                                    float* filBase = filter + 
                                        coStart*cin*hfil*wfil + cinIdx*hfil*wfil*COUTB1 + hf*wfil*cinB1*COUTB1 + wf*cinB1*COUTB1;
                                    float* inBase = input +
                                        cinIdx*hin*win + hi*win*COUTB1 + wi*COUTB1;
                                    for (int kidx = 0; kidx < k; kidx += COUTB1)
                                    {
                                        __asm __volatile (
                                            "fmla v4.4s, v0.4s, v22.s[0]\n"
                                            "fmla v5.4s, v1.4s, v22.s[0]\n"
                                            "fmla v6.4s, v0.4s, v23.s[0]\n"
                                            "fmla v7.4s, v1.4s, v23.s[0]\n"
                                            "fmla v8.4s, v0.4s, v24.s[0]\n"
                                            "fmla v9.4s, v1.4s, v24.s[0]\n"
                                            "fmla v10.4s, v0.4s, v25.s[0]\n"
                                            "fmla v11.4s, v1.4s, v25.s[0]\n"
                                            "fmla v12.4s, v0.4s, v26.s[0]\n"
                                            "fmla v13.4s, v1.4s, v26.s[0]\n"
                                            "fmla v14.4s, v0.4s, v27.s[0]\n"
                                            "fmla v15.4s, v0.4s, v26.s[0]\n"
                                            "fmla v16.4s, v1.4s, v26.s[0]\n"
                                            "fmla v17.4s, v0.4s, v27.s[0]\n"
                                            
                                            "fmla v4.4s, v0.4s, v22.s[0]\n"
                                            "fmla v5.4s, v1.4s, v22.s[0]\n"
                                            "fmla v6.4s, v0.4s, v23.s[0]\n"
                                            "fmla v7.4s, v1.4s, v23.s[0]\n"
                                            "fmla v8.4s, v0.4s, v24.s[0]\n"
                                            "fmla v9.4s, v1.4s, v24.s[0]\n"
                                            "fmla v10.4s, v0.4s, v25.s[0]\n"
                                            "fmla v11.4s, v1.4s, v25.s[0]\n"
                                            "fmla v12.4s, v0.4s, v26.s[0]\n"
                                            "fmla v13.4s, v1.4s, v26.s[0]\n"
                                            "fmla v14.4s, v0.4s, v27.s[0]\n"
                                            "fmla v15.4s, v0.4s, v26.s[0]\n"
                                            "fmla v16.4s, v1.4s, v26.s[0]\n"
                                            "fmla v17.4s, v0.4s, v27.s[0]\n"
                                            
                                            "fmla v4.4s, v0.4s, v22.s[0]\n"
                                            "fmla v5.4s, v1.4s, v22.s[0]\n"
                                            "fmla v6.4s, v0.4s, v23.s[0]\n"
                                            "fmla v7.4s, v1.4s, v23.s[0]\n"
                                            "fmla v8.4s, v0.4s, v24.s[0]\n"
                                            "fmla v9.4s, v1.4s, v24.s[0]\n"
                                            "fmla v10.4s, v0.4s, v25.s[0]\n"
                                            "fmla v11.4s, v1.4s, v25.s[0]\n"
                                            "fmla v12.4s, v0.4s, v26.s[0]\n"
                                            "fmla v13.4s, v1.4s, v26.s[0]\n"
                                            "fmla v14.4s, v0.4s, v27.s[0]\n"
                                            "fmla v15.4s, v0.4s, v26.s[0]\n"
                                            "fmla v16.4s, v1.4s, v26.s[0]\n"
                                            "fmla v17.4s, v0.4s, v27.s[0]\n"

                                            "fmla v4.4s, v0.4s, v22.s[0]\n"
                                            "fmla v5.4s, v1.4s, v22.s[0]\n"
                                            "fmla v6.4s, v0.4s, v23.s[0]\n"
                                            "fmla v7.4s, v1.4s, v23.s[0]\n"
                                            "fmla v8.4s, v0.4s, v24.s[0]\n"
                                            "fmla v9.4s, v1.4s, v24.s[0]\n"
                                            "fmla v10.4s, v0.4s, v25.s[0]\n"
                                            "fmla v11.4s, v1.4s, v25.s[0]\n"
                                            "fmla v12.4s, v0.4s, v26.s[0]\n"
                                            "fmla v13.4s, v1.4s, v26.s[0]\n"
                                            "fmla v14.4s, v0.4s, v27.s[0]\n"
                                            "fmla v15.4s, v0.4s, v26.s[0]\n"
                                            "fmla v16.4s, v1.4s, v26.s[0]\n"
                                            "fmla v17.4s, v0.4s, v27.s[0]\n"

                                            "fmla v4.4s, v0.4s, v22.s[0]\n"
                                            "fmla v5.4s, v1.4s, v22.s[0]\n"
                                            "fmla v6.4s, v0.4s, v23.s[0]\n"
                                            "fmla v7.4s, v1.4s, v23.s[0]\n"
                                            "fmla v8.4s, v0.4s, v24.s[0]\n"
                                            "fmla v9.4s, v1.4s, v24.s[0]\n"
                                            "fmla v10.4s, v0.4s, v25.s[0]\n"
                                            "fmla v11.4s, v1.4s, v25.s[0]\n"
                                            "fmla v12.4s, v0.4s, v26.s[0]\n"
                                            "fmla v13.4s, v1.4s, v26.s[0]\n"
                                            "fmla v14.4s, v0.4s, v27.s[0]\n"
                                            "fmla v15.4s, v0.4s, v26.s[0]\n"
                                            "fmla v16.4s, v1.4s, v26.s[0]\n"
                                            "fmla v17.4s, v0.4s, v27.s[0]\n"

                                            "fmla v4.4s, v0.4s, v22.s[0]\n"
                                            "fmla v5.4s, v1.4s, v22.s[0]\n"
                                            "fmla v6.4s, v0.4s, v23.s[0]\n"
                                            "fmla v7.4s, v1.4s, v23.s[0]\n"
                                            "fmla v8.4s, v0.4s, v24.s[0]\n"
                                            "fmla v9.4s, v1.4s, v24.s[0]\n"
                                            "fmla v10.4s, v0.4s, v25.s[0]\n"
                                            "fmla v11.4s, v1.4s, v25.s[0]\n"
                                            "fmla v12.4s, v0.4s, v26.s[0]\n"
                                            "fmla v13.4s, v1.4s, v26.s[0]\n"
                                            "fmla v14.4s, v0.4s, v27.s[0]\n"
                                            "fmla v15.4s, v0.4s, v26.s[0]\n"
                                            "fmla v16.4s, v1.4s, v26.s[0]\n"
                                            "fmla v17.4s, v0.4s, v27.s[0]\n"

                                            "fmla v4.4s, v0.4s, v22.s[0]\n"
                                            "fmla v5.4s, v1.4s, v22.s[0]\n"
                                            "fmla v6.4s, v0.4s, v23.s[0]\n"
                                            "fmla v7.4s, v1.4s, v23.s[0]\n"
                                            "fmla v8.4s, v0.4s, v24.s[0]\n"
                                            "fmla v9.4s, v1.4s, v24.s[0]\n"
                                            "fmla v10.4s, v0.4s, v25.s[0]\n"
                                            "fmla v11.4s, v1.4s, v25.s[0]\n"
                                            "fmla v12.4s, v0.4s, v26.s[0]\n"
                                            "fmla v13.4s, v1.4s, v26.s[0]\n"
                                            "fmla v14.4s, v0.4s, v27.s[0]\n"
                                            "fmla v15.4s, v0.4s, v26.s[0]\n"
                                            "fmla v16.4s, v1.4s, v26.s[0]\n"
                                            "fmla v17.4s, v0.4s, v27.s[0]\n"

                                            "fmla v4.4s, v0.4s, v22.s[0]\n"
                                            "fmla v5.4s, v1.4s, v22.s[0]\n"
                                            "fmla v6.4s, v0.4s, v23.s[0]\n"
                                            "fmla v7.4s, v1.4s, v23.s[0]\n"
                                            "fmla v8.4s, v0.4s, v24.s[0]\n"
                                            "fmla v9.4s, v1.4s, v24.s[0]\n"
                                            "fmla v10.4s, v0.4s, v25.s[0]\n"
                                            "fmla v11.4s, v1.4s, v25.s[0]\n"
                                            "fmla v12.4s, v0.4s, v26.s[0]\n"
                                            "fmla v13.4s, v1.4s, v26.s[0]\n"
                                            "fmla v14.4s, v0.4s, v27.s[0]\n"
                                            "fmla v15.4s, v0.4s, v26.s[0]\n"
                                            "fmla v16.4s, v1.4s, v26.s[0]\n"
                                            "fmla v17.4s, v0.4s, v27.s[0]\n"
                                            :
                                            :   
                                            :   "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                                                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
                                            );

                                        filBase += 8*8;
                                        inBase += hin*win*COUTB1;
                                    }
                                }
                            }
                        }
                    }
                }
            }   
        }
    }
}

void comp::conv (float* input, float* output, float* bias, int hin, int win, int padding, int stride, int dilation)
{
    if(!isConvThreadInitialized)
    {
        for (int i = 0; i < (comp_num_threads); i++)
        {
            convThreads[i] = pthread_self();
            convThreadConditions[i] = PTHREAD_COND_INITIALIZER; 
            convThreadMutexes[i] = PTHREAD_MUTEX_INITIALIZER;
        }
        convThreadMutexes[comp_num_threads] = PTHREAD_MUTEX_INITIALIZER;
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

    #ifdef __DEBUG_COMP_OFF
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
    convThreadDataObj.workload = cout/COUTB1;

    //Custom threadpool implementation.
    pthread_mutex_lock(&comp::runningMutex);
    for (int i = 0; i < (comp_num_threads); i++)
    {       
        threadArgArr[i].threadDataPtr = (void*)&convThreadDataObj;
        if (pthread_equal(convThreads[i], pthread_self()))
        {
            threadArgArr[i].id = i;
            pthread_create (&convThreads[i], NULL, compConvThreadRoutine, (void* )&threadArgArr[i]);
        }
        else
        {
            pthread_mutex_unlock(&comp::convThreadMutexes[i]);
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
        pthread_mutex_unlock(&comp::runningMutex);
    }
    for (int i = 0; i < (comp_num_threads); i++)
    {
        pthread_mutex_lock(&comp::convThreadMutexes[i]);
    }
    pthread_mutex_unlock(&comp::runningMutex);
}

float* comp::vectorize (float* input, int blocks, int channels, int height, int width, bool isNHWC)
{
    float* out;
    int vecChannel = getVectorNum(channels, COUTB1)*COUTB1;
    if(posix_memalign((void**)(&out), sizeof(float32x4_t), blocks * vecChannel * height * width*sizeof(float)))
    {
        printf ("Vectorize - POSIX memalign failed.");
    }
    for (int b = 0; b < blocks; b++)
    {
        bzero (out + b*vecChannel*height*width + (vecChannel-COUTB1)*height*width, COUTB1*height*width*sizeof(float));
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
                    int vecIdx = c%COUTB1;
                    int vecNum = c/COUTB1;
                    *(out + b*vecChannel*height*width + vecNum*height*width*COUTB1 + h*width*COUTB1 + w*COUTB1 + vecIdx) = val;
                }
            }
        }
    }
    return out;
}

float* comp::deVectorize (float* input, int blocks, int channels, int height, int width)
{
    float* out;
    int vecChannel = getVectorNum(channels, COUTB1)*COUTB1;
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
                    int vecIdx = c%COUTB1;
                    int vecNum = c/COUTB1;
                    val = *(input + b*vecChannel*height*width + vecNum*height*width*COUTB1 + h*width*COUTB1 + w*COUTB1 + vecIdx);
                    *(out + b*(height*width*channels) + h*(width*channels) + w*channels + c) = val;
                }
            }
        }
    }
    return out;
}

void comp::printSize()
{
    printf ("COMP Filter tensor with Kernel Height %d, Blocks %d, Channels %d, Height %d, Width %d.\n", kernelHeight, blocks, channels, height, width);
}

void comp::print(int newlineNum, int format)
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

void* compCompareThreadRoutine(void* threadArg)
{
    struct compCompareThreadData* dataPtr = (struct compCompareThreadData*) threadArg;
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
            pthread_mutex_lock(&comp::printMutex);
            std::cout << "//////////// Warning!! - Float Comapare check failed!! Under epsilon " << std::fixed << std::setprecision(8) << epsilon << " - At block, channel, height, width index of " << b << ", " << c << ", " << h << ", " << w <<  " ////////////" << std::endl;
            std::cout << "\tValue1: " << std::fixed << std::setprecision(5) << val1 << ", value2: " << val2 << ", Error: " << std::setprecision(8) << error << std::endl;
            pthread_mutex_unlock(&comp::printMutex);
        }

    }
    pthread_mutex_lock(&comp::printMutex);
    *(dataPtr->totalPtr) += total; 
    *(dataPtr->numPtr) += num; 
    *(dataPtr->errNumPtr) += errNum; 
    *(dataPtr->varPtr) += var; 
    pthread_mutex_unlock(&comp::printMutex);
    return nullptr;
}

bool comp::floatCompare(float* input1, float* input2, int blocks, int channels, int height, int width, double epsilon, bool isNHWC)
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

    pthread_t threads[comp_num_threads];
    struct compCompareThreadData  threadDataArr[comp_num_threads];

    // Starting threads.
    for (int i = 0; i < comp_num_threads; i++)
    {
        threadDataArr[i].id = i;
        threadDataArr[i].input1 = input1;
        threadDataArr[i].input2 = input2;
        threadDataArr[i].blocks = blocks;
        threadDataArr[i].channels = channels;
        threadDataArr[i].height = height;
        threadDataArr[i].width = width;
        threadDataArr[i].workloads = blocks*channels*height*width/comp_num_threads;
        threadDataArr[i].isNHWC = isNHWC;
        threadDataArr[i].epsilon = epsilon;
        threadDataArr[i].totalPtr = &total;
        threadDataArr[i].errNumPtr = &errNum;
        threadDataArr[i].varPtr = &var;
        threadDataArr[i].numPtr = &num;
        pthread_create (&threads[i], NULL, compCompareThreadRoutine, (void* )&(threadDataArr[i]));
    }
    for (int i = 0; i < comp_num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    if(errNum == 0)
    {
        std::cout << "comp to Float Arr Compare passed under epsilon " << std::fixed << std::setprecision(6) << epsilon << std::endl;
        std::cout << "Total Number of elements: " << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(10) << sqrt(var/num) << std::endl;
    }
    else
    {
        std::cout << "//////////// Warning!! - comp compare to Float Arr failed. ////////////" << std::endl;
        std::cout << "Total Number of elements: " << std::fixed << std::setprecision(6) << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(10) << sqrt(var/num) << std::endl;
    }
    std::cout << std::endl;
    return out;
}
