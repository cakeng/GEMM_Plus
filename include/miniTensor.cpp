/*
*	miniTensor.cpp
*	Created: 2021-1-9
*	Author: JongSeok Park (cakeng@naver.com)
*/
#include <miniTensor.h>

int mTen::mTen_num_threads = std::thread::hardware_concurrency();

mTenThreadArg mTen::threadArgArr[__MAX_NUM_THREADS];

pthread_mutex_t mTen::runningMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t mTen::runningCondition = PTHREAD_COND_INITIALIZER; 
int mTen::runningThreads = 0;

pthread_t mTen::convThreads[__MAX_NUM_THREADS];
pthread_cond_t mTen::convThreadConditions[__MAX_NUM_THREADS];
pthread_mutex_t mTen::convThreadMutexes[__MAX_NUM_THREADS];
mTenConvThreadData mTen::convThreadDataObj;
bool mTen::isConvThreadInitialized = false;

pthread_mutex_t mTen::printMutex = PTHREAD_MUTEX_INITIALIZER;
#ifdef __DEBUG_MTEN_OFF
      pthread_mutex_t mTen::threadLockMutex = PTHREAD_MUTEX_INITIALIZER;
#endif

mTen::mTen()
{
    blocks = 0;
    channels = 0;
    height = 0;
    width = 0;
    tensorPtr = nullptr;
    isNHWC = false;
    vecSize = 0;
    vecNum = 0;
}

mTen::mTen(int blocksIn, int channelsIn, int heightIn, int widthIn)
{
    blocks = blocksIn;
    channels = channelsIn;
    height = heightIn;
    width = widthIn;
    tensorPtr = new float[blocks*channels*height*width];
    bzero(tensorPtr, blocks*channels*height*width*sizeof(float));
    isNHWC = false;
    vecSize = 0;
    vecNum = 0;
}

mTen::mTen(float* input, int blocksIn, int channelsIn, int heightIn, int widthIn, bool NHWC, int vecSizeIn)
{
    blocks = blocksIn;
    channels = channelsIn;
    height = heightIn;
    width = widthIn;
    if (vecSizeIn == 0)
    {
        if (NHWC)
        {
            isNHWC = true;
            vecSize = 0;
            vecNum = 0;
            tensorPtr = new float[blocks*height*width*channels];
            for (int bIdx = 0; bIdx < blocksIn; bIdx++)
            {
                for (int cIdx = 0; cIdx < channelsIn; cIdx++)
                {
                    float* saveTarget = tensorPtr + bIdx*channels*height*width + cIdx;
                    float* loadTarget = input + bIdx*channels*height*width + cIdx*height*width;
                    for (int hIdx = 0; hIdx < heightIn; hIdx++)
                    {
                        for (int wIdx = 0; wIdx < widthIn; wIdx++)
                        {
                            *(saveTarget) = *(loadTarget);
                            saveTarget += channels;
                            loadTarget += 1;
                        }
                    }
                }
            }
        }
        else
        {
            isNHWC = false;
            vecSize = 0;
            vecNum = 0;
            tensorPtr = new float[blocks*channels*height*width];
            memcpy(tensorPtr, input, blocks*channels*height*width*sizeof(float));
        }
    }
    else
    {
        isNHWC = true;
        vecSize = vecSizeIn;
        vecNum = getVectorNum(channelsIn, vecSizeIn);
        tensorPtr = new float[blocks*vecNum*height*width*vecSize];
        for (int bIdx = 0; bIdx < blocksIn; bIdx++)
        {
            bzero(tensorPtr + ((bIdx+1)*vecNum - 1)*height*width*vecSize, height*width*vecSize*sizeof(float));
            for (int cIdx = 0; cIdx < channelsIn; cIdx++)
            {
                float* saveTarget = tensorPtr + bIdx*vecNum*height*width*vecSize + (cIdx/vecSize)*height*width*vecSize + (cIdx%vecSize);
                float* loadTarget = input + bIdx*channels*height*width + cIdx*height*width;
                for (int hIdx = 0; hIdx < heightIn; hIdx++)
                {
                    for (int wIdx = 0; wIdx < widthIn; wIdx++)
                    {
                        *(saveTarget) = *(loadTarget);
                        saveTarget += vecSize;
                        loadTarget += 1;
                    }
                }
            }
        }
    }
}

mTen::mTen(mTen& old)
{
    blocks = old.blocks;
    channels = old.channels;
    height = old.height;
    width = old.width;
    if (old.vecSize == 0)
    {
        isNHWC = old.isNHWC;
        vecSize = 0;
        vecNum = 0;
        tensorPtr = new float[blocks*channels*height*width];
        memcpy(tensorPtr, old.tensorPtr, blocks*channels*height*width*sizeof(float));
    }
    else
    {
        isNHWC = old.isNHWC;
        vecSize = old.vecSize;
        vecNum = old.vecNum;
        tensorPtr = new float[blocks*vecNum*height*width*vecSize];
        memcpy(tensorPtr, old.tensorPtr, blocks*vecNum*height*width*vecSize*sizeof(float));
    }
}

mTen &mTen::operator=(const mTen &other)
{
    if (this != &other)
    {
        if (tensorPtr != nullptr)
        {
            delete[] tensorPtr;
        }
        blocks = other.blocks;
        channels = other.channels;
        height = other.height;
        width = other.width;
        isNHWC = other.isNHWC;
        vecSize = other.vecSize;
        vecNum = other.vecNum;

        if (other.tensorPtr != nullptr)
        {
            if (other.vecSize == 0)
            {
                tensorPtr = new float[blocks*channels*height*width];
                memcpy(tensorPtr, other.tensorPtr, blocks*channels*height*width*sizeof(float));
            }
            else
            {
                tensorPtr = new float[blocks*vecNum*height*width*vecSize];
                memcpy(tensorPtr, other.tensorPtr, blocks*vecNum*height*width*vecSize*sizeof(float));
            }
        }
        else
        {
            tensorPtr = nullptr;
        }
    }
    return *this;
}

mTen::~mTen()
{
    if (tensorPtr != nullptr)
    {
        delete[] tensorPtr;
    }
}

void mTen::NCHWtoNHWC()
{
    if (!isNHWC)
    {
        isNHWC = true;
        vecSize = 0;
        vecNum = 0;
        if (tensorPtr != nullptr)
        {
            float* newTensorPtr = new float[blocks*height*width*channels];
            for (int bIdx = 0; bIdx < blocks; bIdx++)
            {
                for (int cIdx = 0; cIdx < channels; cIdx++)
                {
                    float* saveTarget = newTensorPtr + bIdx*channels*height*width + cIdx;
                    float* loadTarget = tensorPtr + bIdx*channels*height*width + cIdx*height*width;
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
            delete[] tensorPtr;
            tensorPtr = newTensorPtr;
        }
        else
        {
            printf ("NCHWtoNHWC - WARNING - EMPTY TENSOR!\n");
        }
    }
}

void mTen::NHWCtoNCHW()
{
    if (isNHWC)
    {
        isNHWC = false;
        if (vecSize == 0)
        {
            if (tensorPtr != nullptr)
            {
                float* newTensorPtr = new float[blocks*channels*height*width];
                for (int bIdx = 0; bIdx < blocks; bIdx++)
                {
                    for (int cIdx = 0; cIdx < channels; cIdx++)
                    {
                        float* loadTarget = tensorPtr + bIdx*channels*height*width + cIdx;
                        float* saveTarget = newTensorPtr + bIdx*channels*height*width + cIdx*height*width;
                        for (int hIdx = 0; hIdx < height; hIdx++)
                        {
                            for (int wIdx = 0; wIdx < width; wIdx++)
                            {
                                *(saveTarget) = *(loadTarget);
                                loadTarget += channels;
                                saveTarget += 1;
                            }
                        }
                    }
                }
                delete[] tensorPtr;
                tensorPtr = newTensorPtr;
            }
            else
            {
                printf ("NHWCtoNCHW - WARNING - EMPTY TENSOR!\n");
            }
        }
        else
        {
            if (tensorPtr != nullptr)
            {
                float* newTensorPtr = new float[blocks*channels*height*width];
                for (int bIdx = 0; bIdx < blocks; bIdx++)
                {
                    for (int cIdx = 0; cIdx < channels; cIdx++)
                    {
                        float* loadTarget = tensorPtr + bIdx*vecNum*height*width*vecSize + (cIdx/vecSize)*height*width*vecSize + cIdx%vecSize;
                        float* saveTarget = newTensorPtr + bIdx*channels*height*width + cIdx*height*width;
                        for (int hIdx = 0; hIdx < height; hIdx++)
                        {
                            for (int wIdx = 0; wIdx < width; wIdx++)
                            {
                                *(saveTarget) = *(loadTarget);
                                loadTarget += vecSize;
                                saveTarget += 1;
                            }
                        }
                    }
                }
                delete[] tensorPtr;
                tensorPtr = newTensorPtr;
            }
            else
            {
                printf ("NHWCtoNCHW - WARNING - EMPTY TENSOR!\n");
            }
            vecSize = 0;
            vecNum = 0;
        }
    }
}

void* mTenVectorizeThreadRoutine(void *threadArg)
{
    struct mTenVectorizeThreadData* data = (struct mTenVectorizeThreadData*)threadArg;
    mTen* input = data->input;
    int* workIdxPtr = data->workLoadIndexPtr;
    const int channels = input->channels;
    const int blocks = input->blocks;
    const int height = input->height;
    const int width = input->width;
    const int vecSize = input->vecSize;
    const int vecNum = input->vecNum;
    const int vecSizeIn = data->vecSizeIn;
    const int vecNumIn = input->getVectorNum(channels, vecSizeIn);
    const int vecInBlockSize = height*width*vecSizeIn;
    const bool isNHWC = input->isNHWC;
    float* tensorPtr = input->tensorPtr;
    float* newTensorPtr = data->newTensorPtr;
    if (isNHWC)
    {
        if (vecSize == 0)
        {   
            while (true)
            {
                const int hwIdx = __atomic_sub_fetch(workIdxPtr, 1, __ATOMIC_RELAXED);
                if (hwIdx < 0)
                {
                    return nullptr;
                }
                else
                {
                    for (int bIdx = 0; bIdx < blocks; bIdx++)
                    {
                        float* loadPtr = tensorPtr + bIdx*channels*height*width + hwIdx*channels;
                        float* savePtr = newTensorPtr + bIdx*vecNumIn*vecInBlockSize + hwIdx*vecSizeIn;
                        for (int cIdx = 0; cIdx < channels;)
                        {
                            if (vecSizeIn == 32 && ((cIdx + 32) < channels))
                            {
                                float32x4_t vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
                                vec0 = vld1q_f32(loadPtr);
                                vec1 = vld1q_f32(loadPtr + 4);
                                vec2 = vld1q_f32(loadPtr + 8);
                                vec3 = vld1q_f32(loadPtr + 12);
                                vec4 = vld1q_f32(loadPtr + 16);
                                vec5 = vld1q_f32(loadPtr + 20);
                                vec6 = vld1q_f32(loadPtr + 24);
                                vec7 = vld1q_f32(loadPtr + 28);
                                vst1q_f32(savePtr, vec0);
                                vst1q_f32(savePtr + 4, vec1);
                                vst1q_f32(savePtr + 8, vec2);
                                vst1q_f32(savePtr + 12, vec3);
                                vst1q_f32(savePtr + 16, vec4);
                                vst1q_f32(savePtr + 20, vec5);
                                vst1q_f32(savePtr + 24, vec6);
                                vst1q_f32(savePtr + 28, vec7);
                                loadPtr += 32;
                                savePtr += vecInBlockSize;
                                cIdx += 32;
                            }
                            else if (vecSizeIn == 4 && ((cIdx + 4) < channels))
                            {
                                float32x4_t vec0;
                                vec0 = vld1q_f32(loadPtr);
                                vst1q_f32(savePtr, vec0);
                                loadPtr += 4;
                                savePtr += vecInBlockSize;
                                cIdx += 4;
                            }
                            else
                            {
                                for (int vIdx = 0; vIdx < vecSizeIn; vIdx++)
                                {
                                    if (cIdx + vIdx < channels)
                                    {
                                        *savePtr = *loadPtr;
                                        savePtr++;
                                        loadPtr++;
                                    }
                                }
                                savePtr += (vecInBlockSize - vecSizeIn);
                                cIdx += vecSizeIn;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            const int vecBlockSize = height*width*vecSize;
            int counter;
            while (true)
            {
                const int hwIdx = __atomic_sub_fetch(workIdxPtr, 1, __ATOMIC_RELAXED);
                if (hwIdx < 0)
                {
                    return nullptr;
                }
                else
                {
                    for (int bIdx = 0; bIdx < blocks; bIdx++)
                    {
                        float* loadPtr = tensorPtr + bIdx*vecNum*vecBlockSize + hwIdx*vecSize;
                        float* savePtr = newTensorPtr + bIdx*vecNumIn*vecInBlockSize + hwIdx*vecSizeIn;
                        counter = 0;
                        for (int vIdx = 0; vIdx < vecNum;)
                        {
                            if (vecSizeIn == 32 && vecSize == 32)
                            {
                                float32x4_t vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
                                vec0 = vld1q_f32(loadPtr);
                                vec1 = vld1q_f32(loadPtr + 4);
                                vec2 = vld1q_f32(loadPtr + 8);
                                vec3 = vld1q_f32(loadPtr + 12);
                                vec4 = vld1q_f32(loadPtr + 16);
                                vec5 = vld1q_f32(loadPtr + 20);
                                vec6 = vld1q_f32(loadPtr + 24);
                                vec7 = vld1q_f32(loadPtr + 28);
                                vst1q_f32(savePtr, vec0);
                                vst1q_f32(savePtr + 4, vec1);
                                vst1q_f32(savePtr + 8, vec2);
                                vst1q_f32(savePtr + 12, vec3);
                                vst1q_f32(savePtr + 16, vec4);
                                vst1q_f32(savePtr + 20, vec5);
                                vst1q_f32(savePtr + 24, vec6);
                                vst1q_f32(savePtr + 28, vec7);
                                loadPtr += vecBlockSize;
                                savePtr += vecInBlockSize;
                                vIdx++;
                            }
                            else if (vecSizeIn == 32 && vecSize == 4)
                            {
                                float32x4_t vec0;
                                vec0 = vld1q_f32(loadPtr);
                                vst1q_f32(savePtr, vec0);
                                loadPtr += vecBlockSize;
                                savePtr += 4;
                                if (vIdx%8 == 7)
                                {
                                    savePtr += (vecInBlockSize - 32);
                                }
                                vIdx++;
                            }
                            else if (vecSizeIn == 4 && vecSize == 4)
                            {
                                float32x4_t vec0;
                                vec0 = vld1q_f32(loadPtr);
                                vst1q_f32(savePtr, vec0);
                                loadPtr += vecBlockSize;
                                savePtr += vecInBlockSize;
                                vIdx++;
                            }
                            else if (vecSizeIn == 4 && vecSize == 32)
                            {
                                if (vIdx*8 + counter < vecNumIn)
                                {
                                    float32x4_t vec0;
                                    vec0 = vld1q_f32(loadPtr);
                                    vst1q_f32(savePtr, vec0);
                                    savePtr += vecInBlockSize;
                                }
                                loadPtr += 4;
                                if (counter%8 == 7)
                                {
                                    vIdx++;
                                    loadPtr += (vecBlockSize - 32);
                                }
                                counter++;
                            }
                            else
                            {
                                int cIdx = vIdx*vecSize;
                                for (int vVIdx = 0; vVIdx < vecSize; vVIdx++)
                                {
                                    *(savePtr + vecInBlockSize*(cIdx/vecSizeIn) + (cIdx%vecSizeIn)) = *loadPtr;
                                    cIdx++;
                                    loadPtr++;
                                }
                                loadPtr += (vecBlockSize - vecSize);
                                vIdx++;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        while (true)
        {
            const int cIdx = __atomic_sub_fetch(workIdxPtr, 1, __ATOMIC_RELAXED);
            if (cIdx < 0)
            {
                return nullptr;
            }
            else
            {
                for (int bIdx = 0; bIdx < blocks; bIdx++)
                {
                    float* loadPtr = tensorPtr + (bIdx*channels + cIdx)*height*width; 
                    float* savePtr = newTensorPtr + bIdx*vecNumIn*vecInBlockSize + (cIdx/vecSizeIn)*height*width*vecSizeIn + (cIdx%vecSizeIn);
                    for (int whIdx = 0; whIdx < width*height; whIdx++)
                    {
                        *savePtr = *loadPtr;
                        loadPtr++;
                        savePtr += vecSizeIn;
                    }
                }
            }
        }
    }
}

void mTen::vectorize(int vecSizeIn)
{
    if (tensorPtr != nullptr)
    {
        int workLoadIndex;
        if (isNHWC)
        {
            workLoadIndex = width*height;
        }
        else
        {
            workLoadIndex = channels;
        }
        struct mTenVectorizeThreadData threadData;
        int vecNumIn = getVectorNum(channels, vecSizeIn);
        float* newTensorPtr = new float[blocks*vecNumIn*height*width*vecSizeIn];
        for (int bIdx = 0; bIdx < blocks; bIdx++)
        {
            bzero(newTensorPtr + ((bIdx+1)*vecNumIn - 1)*height*width*vecSizeIn, height*width*vecSizeIn*sizeof(float));
        }
        pthread_t threads[mTen_num_threads];
        threadData.input = this;
        threadData.vecSizeIn = vecSizeIn;
        threadData.workLoadIndexPtr = &workLoadIndex;
        threadData.newTensorPtr = newTensorPtr;
        for (int i = 0; i < mTen_num_threads; i++)
        {
            pthread_create (&threads[i], NULL, mTenVectorizeThreadRoutine, (void*)&threadData);
        }
        for (int i = 0; i < mTen_num_threads; i++)
        {
            pthread_join (threads[i], NULL);
        }
        delete[] tensorPtr;
        vecSize = vecSizeIn;
        vecNum = vecNumIn;
        tensorPtr = newTensorPtr;
        isNHWC = true;
    }
    else
    {
        printf ("vectorize - WARNING - EMPTY TENSOR!\n");
        vecSize = 0;
        vecNum = 0;
    }
}

void mTen::deVectorize()
{
    if (isNHWC && vecSize)
    {
        if (tensorPtr != nullptr)
        {
            float* newTensorPtr = new float[blocks*channels*height*width];
            for (int bIdx = 0; bIdx < blocks; bIdx++)
            {
                for (int cIdx = 0; cIdx < channels; cIdx++)
                {
                    float* loadTarget = tensorPtr + bIdx*vecNum*height*width*vecSize + (cIdx/vecSize)*height*width*vecSize + (cIdx%vecSize);
                    float* saveTarget = newTensorPtr + bIdx*channels*height*width + cIdx;
                    for (int hIdx = 0; hIdx < height; hIdx++)
                    {
                        for (int wIdx = 0; wIdx < width; wIdx++)
                        {
                            *(saveTarget) = *(loadTarget);
                            saveTarget += channels;
                            loadTarget += vecSize;
                        }
                    }
                }
            }
            delete[] tensorPtr;
            tensorPtr = newTensorPtr;
            vecSize = 0;
            vecNum = 0;
        }
        else
        {
            printf ("deVectorize - WARNING - EMPTY TENSOR!\n");
            vecSize = 0;
            vecNum = 0;
        }
    }
}


void aarch64_convKernel3x3_vec8_vec8_iterRow_ASM(float* input, float* filter, float* output, int inputWidth, int inputHeight, int inputHStart, int outWidth, int iterHeight, int padding, int filterBlockSize)
{
    // 11 general purpose registers for input, 4 genneral purpose registers for vIdx, fIdx, hIdx, wIdx (x9, x10, x11, x12), 2 genneral purpose registers for calculation (x13, x14).
    // Total of 17 general purpose registers.
    // 24 NEON vector registers for filter values (v0 ~ v23), 2 NEON vector registers for input values (v24, v25), 6 NEON vector registers for output values (v26 ~ v31)
    // Total of 32 NEON vector registers.
    const int moduloVal = inputWidth - inputWidth%14;

    // float testArr[128] = {0};
    // int testArr2[128] = {0};
    __asm __volatile (
        "mov x10, #0\n" // x10 = fIdx
        "FIDX_LOOP_8TO8:\n"
        "mov x9, #0\n" // x9 = vInIdx
        "VIDX_LOOP_8TO8:\n"
        "add x12, x10, x10, lsl #1\n" // x12 = 3*fIdx
        "mov x11, #32\n"
        "add x12, %[filP], x12, lsl #5\n" // x12 = %[filP] + 96*fIdx
        "sub x14, %[filS], #16\n"
        "add x12, x12, x9, lsl #2\n"
        "add x13, %[startH], x10\n"
        "lsl x14, x14, #2\n"
        "ld4 {v16.s-v19.s}[0], [x12], x11\n"
        "ld4 {v8.s-v11.s}[0], [x12], x11\n"
        "ld4 {v0.s-v3.s}[0], [x12], x14\n"
        "ld4 {v16.s-v19.s}[1], [x12], x11\n"
        "ld4 {v8.s-v11.s}[1], [x12], x11\n"
        "ld4 {v0.s-v3.s}[1], [x12], x14\n"
        "ld4 {v16.s-v19.s}[2], [x12], x11\n"
        "ld4 {v8.s-v11.s}[2], [x12], x11\n"
        "ld4 {v0.s-v3.s}[2], [x12], x14\n"
        "ld4 {v16.s-v19.s}[3], [x12], x11\n"
        "ld4 {v8.s-v11.s}[3], [x12], x11\n"
        "ld4 {v0.s-v3.s}[3], [x12], x14\n"

        "ld4 {v20.s-v23.s}[0], [x12], x11\n"
        "ld4 {v12.s-v15.s}[0], [x12], x11\n"
        "ld4 {v4.s-v7.s}[0], [x12], x14\n"
        "ld4 {v20.s-v23.s}[1], [x12], x11\n"
        "ld4 {v12.s-v15.s}[1], [x12], x11\n"
        "ld4 {v4.s-v7.s}[1], [x12], x14\n"
        "ld4 {v20.s-v23.s}[2], [x12], x11\n"
        "ld4 {v12.s-v15.s}[2], [x12], x11\n"
        "ld4 {v4.s-v7.s}[2], [x12], x14\n"
        "ld4 {v20.s-v23.s}[3], [x12], x11\n"
        "ld4 {v12.s-v15.s}[3], [x12], x11\n"
        "ld4 {v4.s-v7.s}[3], [x12], x14\n"
        
        "mov x11, #0\n" // x11 = hIdx (Referenced to Output)
        "cmp x13, #0\n"
        "b.ge NO_PAD_SKIPPING_8TO8\n"
        "sub x11, x11, x13\n"
        "NO_PAD_SKIPPING_8TO8:"
        "cmp x11, %[outH]\n"
        "mul x14, x11, %[outW]\n" // x14 = Number of output height to skip.
        "b.ge HIDX_LOOP_8TO8_EXIT\n" // HIDX_LOOP_8TO8 EXIT
        "add x13, x11, x10\n" // x13 = Number of input height to skip.
        "mul x13, x13, %[inW]\n"
        "add x14, %[outP], x14, lsl# 5\n" // x14 = Output Ptr
        "add x13, %[inP], x13, lsl #5\n" // x13 = Input Ptr
        "add x13, x13, x9, lsl #2\n"
        
        "HIDX_LOOP_8TO8:\n"
        "add x12, x10, %[startH]\n"
        "add x12, x12, x11\n"
        "cmp x12, %[inH]\n"
        "b.ge HIDX_LOOP_8TO8_EXIT\n" // HIDX_LOOP_8TO8 EXIT
        "mov x12, #0\n" // x12 = wIdx

        "prfm pldl1strm, [x13]\n"
        "cmp %[inW], #14\n"
        "b.lt WIDX_LOOP_8TO8_EXIT\n"

        "cmp %[pad], #1\n"
        "b.gt START_WIDX_8TO8_PAD_NOT_SKIP\n"
        "b.eq START_WIDX_8TO8_PAD1_SKIP\n"
        "b WIDX_LOOP_8TO8\n"
        "START_WIDX_8TO8_PAD_NOT_SKIP:"
        "ldp q26, q27, [x14], #32\n"
        "START_WIDX_8TO8_PAD1_SKIP:\n"
        "ldp q28, q29, [x14], #32\n"
        "WIDX_LOOP_8TO8:\n"
        // C: filvecW0 - v0 ~ v7,  B: filvecW1 - v8 ~ v15, A: filvecW2 - v16 ~ v23, 
        // inputVec0 - v24, inputVec1 - v25, outVec00 - v26, outVec01 - v27, outVec10 - v28, outVec11 - v29, outVec20 - v30, outVec21 - v31
        "ldp q30, q31, [x14], #32\n"
        "ldr q24, [x13], #32\n" // Input 0
        "ldr q25, [x13], #32\n" // Input 1

        "b.gt WIDX_8TO8_PAD_NOT_SKIP\n"
        "b.eq WIDX_8TO8_PAD1_SKIP\n"
        "fmla v30.4s, v16.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v31.4s, v20.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v30.4s, v17.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v31.4s, v21.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v30.4s, v18.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v31.4s, v22.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v30.4s, v19.4s, v24.s[3]\n"   // 0A to idx 2
        "fmla v31.4s, v23.4s, v24.s[3]\n"   // 0A to idx 2

        "ldp q26, q27, [x14], #32\n"
        "ldr q24, [x13], #32\n" // Input 2

        "fmla v30.4s, v8.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v31.4s, v12.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v26.4s, v16.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v27.4s, v20.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v30.4s, v9.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v31.4s, v13.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v17.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v27.4s, v21.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v30.4s, v10.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v31.4s, v14.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v26.4s, v18.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v27.4s, v22.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v30.4s, v11.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v31.4s, v15.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v26.4s, v19.4s, v25.s[3]\n"   // 1A to idx 3
        "fmla v27.4s, v23.4s, v25.s[3]\n"   // 1A to idx 3

        "ldp q28, q29, [x14], #32\n"
        "ldr q25, [x13], #32\n" // Input 3
        "b WIDX_8TO8_PAD_SKIP_END\n"
        "WIDX_8TO8_PAD1_SKIP:\n"
        "fmla v28.4s, v8.4s, v24.s[0]\n"   // 0B to idx 1
        "fmla v29.4s, v12.4s, v24.s[0]\n"   // 0B to idx 1
        "fmla v30.4s, v16.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v31.4s, v20.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v28.4s, v9.4s, v24.s[1]\n"   // 0B to idx 1
        "fmla v29.4s, v13.4s, v24.s[1]\n"   // 0B to idx 1
        "fmla v30.4s, v17.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v31.4s, v21.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v28.4s, v10.4s, v24.s[2]\n"   // 0B to idx 1
        "fmla v29.4s, v14.4s, v24.s[2]\n"   // 0B to idx 1
        "fmla v30.4s, v18.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v31.4s, v22.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v28.4s, v11.4s, v24.s[3]\n"   // 0B to idx 1
        "fmla v29.4s, v15.4s, v24.s[3]\n"   // 0B to idx 1
        "fmla v30.4s, v19.4s, v24.s[3]\n"   // 0A to idx 2
        "fmla v31.4s, v23.4s, v24.s[3]\n"   // 0A to idx 2

        "ldp q26, q27, [x14], #32\n"
        "ldr q24, [x13], #32\n" // Input 2

        "fmla v28.4s, v0.4s, v25.s[0]\n"   // 1C to idx 1
        "fmla v29.4s, v4.4s, v25.s[0]\n"   // 1C to idx 1
        "fmla v30.4s, v8.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v31.4s, v12.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v26.4s, v16.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v27.4s, v20.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v28.4s, v1.4s, v25.s[1]\n"   // 1C to idx 1
        "fmla v29.4s, v5.4s, v25.s[1]\n"   // 1C to idx 1
        "fmla v30.4s, v9.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v31.4s, v13.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v17.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v27.4s, v21.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v28.4s, v2.4s, v25.s[2]\n"   // 1C to idx 1
        "fmla v29.4s, v6.4s, v25.s[2]\n"   // 1C to idx 1
        "fmla v30.4s, v10.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v31.4s, v14.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v26.4s, v18.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v27.4s, v22.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v28.4s, v3.4s, v25.s[3]\n"   // 1C to idx 1
        "fmla v29.4s, v7.4s, v25.s[3]\n"   // 1C to idx 1
        "fmla v30.4s, v11.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v31.4s, v15.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v26.4s, v19.4s, v25.s[3]\n"   // 1A to idx 3
        "fmla v27.4s, v23.4s, v25.s[3]\n"   // 1A to idx 3
        
        "stp q28, q29, [x14, #-96]\n"
        "ldp q28, q29, [x14], #32\n"
        "ldr q25, [x13], #32\n" // Input 3
        "b WIDX_8TO8_PAD_SKIP_END\n"
        "WIDX_8TO8_PAD_NOT_SKIP:\n"
        "fmla v26.4s, v0.4s, v24.s[0]\n"   // 0C to idx 0
        "fmla v27.4s, v4.4s, v24.s[0]\n"   // 0C to idx 0
        "fmla v28.4s, v8.4s, v24.s[0]\n"   // 0B to idx 1
        "fmla v29.4s, v12.4s, v24.s[0]\n"   // 0B to idx 1
        "fmla v30.4s, v16.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v31.4s, v20.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v26.4s, v1.4s, v24.s[1]\n"   // 0C to idx 0
        "fmla v27.4s, v5.4s, v24.s[1]\n"   // 0C to idx 0
        "fmla v28.4s, v9.4s, v24.s[1]\n"   // 0B to idx 1
        "fmla v29.4s, v13.4s, v24.s[1]\n"   // 0B to idx 1
        "fmla v30.4s, v17.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v31.4s, v21.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v26.4s, v2.4s, v24.s[2]\n"   // 0C to idx 0
        "fmla v27.4s, v6.4s, v24.s[2]\n"   // 0C to idx 0
        "fmla v28.4s, v10.4s, v24.s[2]\n"   // 0B to idx 1
        "fmla v29.4s, v14.4s, v24.s[2]\n"   // 0B to idx 1
        "fmla v30.4s, v18.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v31.4s, v22.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v26.4s, v3.4s, v24.s[3]\n"   // 0C to idx 0
        "fmla v27.4s, v7.4s, v24.s[3]\n"   // 0C to idx 0
        "fmla v28.4s, v11.4s, v24.s[3]\n"   // 0B to idx 1
        "fmla v29.4s, v15.4s, v24.s[3]\n"   // 0B to idx 1
        "fmla v30.4s, v19.4s, v24.s[3]\n"   // 0A to idx 2
        "fmla v31.4s, v23.4s, v24.s[3]\n"   // 0A to idx 2

        "stp q26, q27, [x14, #-96]\n"
        "ldp q26, q27, [x14], #32\n"
        "ldr q24, [x13], #32\n" // Input 2

        "fmla v28.4s, v0.4s, v25.s[0]\n"   // 1C to idx 1
        "fmla v29.4s, v4.4s, v25.s[0]\n"   // 1C to idx 1
        "fmla v30.4s, v8.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v31.4s, v12.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v26.4s, v16.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v27.4s, v20.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v28.4s, v1.4s, v25.s[1]\n"   // 1C to idx 1
        "fmla v29.4s, v5.4s, v25.s[1]\n"   // 1C to idx 1
        "fmla v30.4s, v9.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v31.4s, v13.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v17.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v27.4s, v21.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v28.4s, v2.4s, v25.s[2]\n"   // 1C to idx 1
        "fmla v29.4s, v6.4s, v25.s[2]\n"   // 1C to idx 1
        "fmla v30.4s, v10.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v31.4s, v14.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v26.4s, v18.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v27.4s, v22.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v28.4s, v3.4s, v25.s[3]\n"   // 1C to idx 1
        "fmla v29.4s, v7.4s, v25.s[3]\n"   // 1C to idx 1
        "fmla v30.4s, v11.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v31.4s, v15.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v26.4s, v19.4s, v25.s[3]\n"   // 1A to idx 3
        "fmla v27.4s, v23.4s, v25.s[3]\n"   // 1A to idx 3
        
        "stp q28, q29, [x14, #-96]\n"
        "ldp q28, q29, [x14], #32\n"
        "ldr q25, [x13], #32\n" // Input 3
        "WIDX_8TO8_PAD_SKIP_END:\n"

        "fmla v30.4s, v0.4s, v24.s[0]\n"   // 2C to idx 2
        "fmla v31.4s, v4.4s, v24.s[0]\n"   // 2C to idx 2
        "fmla v26.4s, v8.4s, v24.s[0]\n"   // 2B to idx 0
        "fmla v27.4s, v12.4s, v24.s[0]\n"   // 2B to idx 0
        "fmla v28.4s, v16.4s, v24.s[0]\n"   // 2A to idx 1
        "fmla v29.4s, v20.4s, v24.s[0]\n"   // 2A to idx 1
        "fmla v30.4s, v1.4s, v24.s[1]\n"   // 2C to idx 2
        "fmla v31.4s, v5.4s, v24.s[1]\n"   // 2C to idx 2
        "fmla v26.4s, v9.4s, v24.s[1]\n"   // 2B to idx 0
        "fmla v27.4s, v13.4s, v24.s[1]\n"   // 2B to idx 0
        "fmla v28.4s, v17.4s, v24.s[1]\n"   // 2A to idx 1
        "fmla v29.4s, v21.4s, v24.s[1]\n"   // 2A to idx 1
        "fmla v30.4s, v2.4s, v24.s[2]\n"   // 2C to idx 2
        "fmla v31.4s, v6.4s, v24.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v10.4s, v24.s[2]\n"   // 2B to idx 0
        "fmla v27.4s, v14.4s, v24.s[2]\n"   // 2B to idx 0
        "fmla v28.4s, v18.4s, v24.s[2]\n"   // 2A to idx 1
        "fmla v29.4s, v22.4s, v24.s[2]\n"   // 2A to idx 1
        "fmla v30.4s, v3.4s, v24.s[3]\n"   // 2C to idx 2
        "fmla v31.4s, v7.4s, v24.s[3]\n"   // 2C to idx 2
        "fmla v26.4s, v11.4s, v24.s[3]\n"   // 2B to idx 0
        "fmla v27.4s, v15.4s, v24.s[3]\n"   // 2B to idx 0
        "fmla v28.4s, v19.4s, v24.s[3]\n"   // 2A to idx 1
        "fmla v29.4s, v23.4s, v24.s[3]\n"   // 2A to idx 1

        "stp q30, q31, [x14, #-96]\n"
        "ldp q30, q31, [x14], #32\n"
        "ldr q24, [x13], #32\n" // Input 4

        "fmla v26.4s, v0.4s, v25.s[0]\n"   // 0C to idx 0
        "fmla v27.4s, v4.4s, v25.s[0]\n"   // 0C to idx 0
        "fmla v28.4s, v8.4s, v25.s[0]\n"   // 0B to idx 1
        "fmla v29.4s, v12.4s, v25.s[0]\n"   // 0B to idx 1
        "fmla v30.4s, v16.4s, v25.s[0]\n"   // 0A to idx 2
        "fmla v31.4s, v20.4s, v25.s[0]\n"   // 0A to idx 2
        "fmla v26.4s, v1.4s, v25.s[1]\n"   // 0C to idx 0
        "fmla v27.4s, v5.4s, v25.s[1]\n"   // 0C to idx 0
        "fmla v28.4s, v9.4s, v25.s[1]\n"   // 0B to idx 1
        "fmla v29.4s, v13.4s, v25.s[1]\n"   // 0B to idx 1
        "fmla v30.4s, v17.4s, v25.s[1]\n"   // 0A to idx 2
        "fmla v31.4s, v21.4s, v25.s[1]\n"   // 0A to idx 2
        "fmla v26.4s, v2.4s, v25.s[2]\n"   // 0C to idx 0
        "fmla v27.4s, v6.4s, v25.s[2]\n"   // 0C to idx 0
        "fmla v28.4s, v10.4s, v25.s[2]\n"   // 0B to idx 1
        "fmla v29.4s, v14.4s, v25.s[2]\n"   // 0B to idx 1
        "fmla v30.4s, v18.4s, v25.s[2]\n"   // 0A to idx 2
        "fmla v31.4s, v22.4s, v25.s[2]\n"   // 0A to idx 2
        "fmla v26.4s, v3.4s, v25.s[3]\n"   // 0C to idx 0
        "fmla v27.4s, v7.4s, v25.s[3]\n"   // 0C to idx 0
        "fmla v28.4s, v11.4s, v25.s[3]\n"   // 0B to idx 1
        "fmla v29.4s, v15.4s, v25.s[3]\n"   // 0B to idx 1
        "fmla v30.4s, v19.4s, v25.s[3]\n"   // 0A to idx 2
        "fmla v31.4s, v23.4s, v25.s[3]\n"   // 0A to idx 2

        "stp q26, q27, [x14, #-96]\n"
        "ldp q26, q27, [x14], #32\n"
        "ldr q25, [x13], #32\n" // Input 5

        "fmla v28.4s, v0.4s, v24.s[0]\n"   // 1C to idx 1
        "fmla v29.4s, v4.4s, v24.s[0]\n"   // 1C to idx 1
        "fmla v30.4s, v8.4s, v24.s[0]\n"   // 1B to idx 2
        "fmla v31.4s, v12.4s, v24.s[0]\n"   // 1B to idx 2
        "fmla v26.4s, v16.4s, v24.s[0]\n"   // 1A to idx 0
        "fmla v27.4s, v20.4s, v24.s[0]\n"   // 1A to idx 0
        "fmla v28.4s, v1.4s, v24.s[1]\n"   // 1C to idx 1
        "fmla v29.4s, v5.4s, v24.s[1]\n"   // 1C to idx 1
        "fmla v30.4s, v9.4s, v24.s[1]\n"   // 1B to idx 2
        "fmla v31.4s, v13.4s, v24.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v17.4s, v24.s[1]\n"   // 1A to idx 0
        "fmla v27.4s, v21.4s, v24.s[1]\n"   // 1A to idx 0
        "fmla v28.4s, v2.4s, v24.s[2]\n"   // 1C to idx 1
        "fmla v29.4s, v6.4s, v24.s[2]\n"   // 1C to idx 1
        "fmla v30.4s, v10.4s, v24.s[2]\n"   // 1B to idx 2
        "fmla v31.4s, v14.4s, v24.s[2]\n"   // 1B to idx 2
        "fmla v26.4s, v18.4s, v24.s[2]\n"   // 1A to idx 0
        "fmla v27.4s, v22.4s, v24.s[2]\n"   // 1A to idx 0
        "fmla v28.4s, v3.4s, v24.s[3]\n"   // 1C to idx 1
        "fmla v29.4s, v7.4s, v24.s[3]\n"   // 1C to idx 1
        "fmla v30.4s, v11.4s, v24.s[3]\n"   // 1B to idx 2
        "fmla v31.4s, v15.4s, v24.s[3]\n"   // 1B to idx 2
        "fmla v26.4s, v19.4s, v24.s[3]\n"   // 1A to idx 3
        "fmla v27.4s, v23.4s, v24.s[3]\n"   // 1A to idx 3
        
        "stp q28, q29, [x14, #-96]\n"
        "ldp q28, q29, [x14], #32\n"
        "ldr q24, [x13], #32\n" // Input 6

        "fmla v30.4s, v0.4s, v25.s[0]\n"   // 2C to idx 2
        "fmla v31.4s, v4.4s, v25.s[0]\n"   // 2C to idx 2
        "fmla v26.4s, v8.4s, v25.s[0]\n"   // 2B to idx 0
        "fmla v27.4s, v12.4s, v25.s[0]\n"   // 2B to idx 0
        "fmla v28.4s, v16.4s, v25.s[0]\n"   // 2A to idx 1
        "fmla v29.4s, v20.4s, v25.s[0]\n"   // 2A to idx 1
        "fmla v30.4s, v1.4s, v25.s[1]\n"   // 2C to idx 2
        "fmla v31.4s, v5.4s, v25.s[1]\n"   // 2C to idx 2
        "fmla v26.4s, v9.4s, v25.s[1]\n"   // 2B to idx 0
        "fmla v27.4s, v13.4s, v25.s[1]\n"   // 2B to idx 0
        "fmla v28.4s, v17.4s, v25.s[1]\n"   // 2A to idx 1
        "fmla v29.4s, v21.4s, v25.s[1]\n"   // 2A to idx 1
        "fmla v30.4s, v2.4s, v25.s[2]\n"   // 2C to idx 2
        "fmla v31.4s, v6.4s, v25.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v10.4s, v25.s[2]\n"   // 2B to idx 0
        "fmla v27.4s, v14.4s, v25.s[2]\n"   // 2B to idx 0
        "fmla v28.4s, v18.4s, v25.s[2]\n"   // 2A to idx 1
        "fmla v29.4s, v22.4s, v25.s[2]\n"   // 2A to idx 1
        "fmla v30.4s, v3.4s, v25.s[3]\n"   // 2C to idx 2
        "fmla v31.4s, v7.4s, v25.s[3]\n"   // 2C to idx 2
        "fmla v26.4s, v11.4s, v25.s[3]\n"   // 2B to idx 0
        "fmla v27.4s, v15.4s, v25.s[3]\n"   // 2B to idx 0
        "fmla v28.4s, v19.4s, v25.s[3]\n"   // 2A to idx 1
        "fmla v29.4s, v23.4s, v25.s[3]\n"   // 2A to idx 1

        "stp q30, q31, [x14, #-96]\n"
        "ldp q30, q31, [x14], #32\n"
        "ldr q25, [x13], #32\n" // Input 7

        "fmla v26.4s, v0.4s, v24.s[0]\n"   // 0C to idx 0
        "fmla v27.4s, v4.4s, v24.s[0]\n"   // 0C to idx 0
        "fmla v28.4s, v8.4s, v24.s[0]\n"   // 0B to idx 1
        "fmla v29.4s, v12.4s, v24.s[0]\n"   // 0B to idx 1
        "fmla v30.4s, v16.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v31.4s, v20.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v26.4s, v1.4s, v24.s[1]\n"   // 0C to idx 0
        "fmla v27.4s, v5.4s, v24.s[1]\n"   // 0C to idx 0
        "fmla v28.4s, v9.4s, v24.s[1]\n"   // 0B to idx 1
        "fmla v29.4s, v13.4s, v24.s[1]\n"   // 0B to idx 1
        "fmla v30.4s, v17.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v31.4s, v21.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v26.4s, v2.4s, v24.s[2]\n"   // 0C to idx 0
        "fmla v27.4s, v6.4s, v24.s[2]\n"   // 0C to idx 0
        "fmla v28.4s, v10.4s, v24.s[2]\n"   // 0B to idx 1
        "fmla v29.4s, v14.4s, v24.s[2]\n"   // 0B to idx 1
        "fmla v30.4s, v18.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v31.4s, v22.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v26.4s, v3.4s, v24.s[3]\n"   // 0C to idx 0
        "fmla v27.4s, v7.4s, v24.s[3]\n"   // 0C to idx 0
        "fmla v28.4s, v11.4s, v24.s[3]\n"   // 0B to idx 1
        "fmla v29.4s, v15.4s, v24.s[3]\n"   // 0B to idx 1
        "fmla v30.4s, v19.4s, v24.s[3]\n"   // 0A to idx 2
        "fmla v31.4s, v23.4s, v24.s[3]\n"   // 0A to idx 2

        "stp q26, q27, [x14, #-96]\n"
        "ldp q26, q27, [x14], #32\n"
        "ldr q24, [x13], #32\n" // Input 8

        "fmla v28.4s, v0.4s, v25.s[0]\n"   // 1C to idx 1
        "fmla v29.4s, v4.4s, v25.s[0]\n"   // 1C to idx 1
        "fmla v30.4s, v8.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v31.4s, v12.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v26.4s, v16.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v27.4s, v20.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v28.4s, v1.4s, v25.s[1]\n"   // 1C to idx 1
        "fmla v29.4s, v5.4s, v25.s[1]\n"   // 1C to idx 1
        "fmla v30.4s, v9.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v31.4s, v13.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v17.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v27.4s, v21.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v28.4s, v2.4s, v25.s[2]\n"   // 1C to idx 1
        "fmla v29.4s, v6.4s, v25.s[2]\n"   // 1C to idx 1
        "fmla v30.4s, v10.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v31.4s, v14.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v26.4s, v18.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v27.4s, v22.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v28.4s, v3.4s, v25.s[3]\n"   // 1C to idx 1
        "fmla v29.4s, v7.4s, v25.s[3]\n"   // 1C to idx 1
        "fmla v30.4s, v11.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v31.4s, v15.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v26.4s, v19.4s, v25.s[3]\n"   // 1A to idx 3
        "fmla v27.4s, v23.4s, v25.s[3]\n"   // 1A to idx 3
        
        "stp q28, q29, [x14, #-96]\n"
        "ldp q28, q29, [x14], #32\n"
        "ldr q25, [x13], #32\n" // Input 9

        "fmla v30.4s, v0.4s, v24.s[0]\n"   // 2C to idx 2
        "fmla v31.4s, v4.4s, v24.s[0]\n"   // 2C to idx 2
        "fmla v26.4s, v8.4s, v24.s[0]\n"   // 2B to idx 0
        "fmla v27.4s, v12.4s, v24.s[0]\n"   // 2B to idx 0
        "fmla v28.4s, v16.4s, v24.s[0]\n"   // 2A to idx 1
        "fmla v29.4s, v20.4s, v24.s[0]\n"   // 2A to idx 1
        "fmla v30.4s, v1.4s, v24.s[1]\n"   // 2C to idx 2
        "fmla v31.4s, v5.4s, v24.s[1]\n"   // 2C to idx 2
        "fmla v26.4s, v9.4s, v24.s[1]\n"   // 2B to idx 0
        "fmla v27.4s, v13.4s, v24.s[1]\n"   // 2B to idx 0
        "fmla v28.4s, v17.4s, v24.s[1]\n"   // 2A to idx 1
        "fmla v29.4s, v21.4s, v24.s[1]\n"   // 2A to idx 1
        "fmla v30.4s, v2.4s, v24.s[2]\n"   // 2C to idx 2
        "fmla v31.4s, v6.4s, v24.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v10.4s, v24.s[2]\n"   // 2B to idx 0
        "fmla v27.4s, v14.4s, v24.s[2]\n"   // 2B to idx 0
        "fmla v28.4s, v18.4s, v24.s[2]\n"   // 2A to idx 1
        "fmla v29.4s, v22.4s, v24.s[2]\n"   // 2A to idx 1
        "fmla v30.4s, v3.4s, v24.s[3]\n"   // 2C to idx 2
        "fmla v31.4s, v7.4s, v24.s[3]\n"   // 2C to idx 2
        "fmla v26.4s, v11.4s, v24.s[3]\n"   // 2B to idx 0
        "fmla v27.4s, v15.4s, v24.s[3]\n"   // 2B to idx 0
        "fmla v28.4s, v19.4s, v24.s[3]\n"   // 2A to idx 1
        "fmla v29.4s, v23.4s, v24.s[3]\n"   // 2A to idx 1

        "stp q30, q31, [x14, #-96]\n"
        "ldp q30, q31, [x14], #32\n"
        "ldr q24, [x13], #32\n" // Input 10

        "fmla v26.4s, v0.4s, v25.s[0]\n"   // 0C to idx 0
        "fmla v27.4s, v4.4s, v25.s[0]\n"   // 0C to idx 0
        "fmla v28.4s, v8.4s, v25.s[0]\n"   // 0B to idx 1
        "fmla v29.4s, v12.4s, v25.s[0]\n"   // 0B to idx 1
        "fmla v30.4s, v16.4s, v25.s[0]\n"   // 0A to idx 2
        "fmla v31.4s, v20.4s, v25.s[0]\n"   // 0A to idx 2
        "fmla v26.4s, v1.4s, v25.s[1]\n"   // 0C to idx 0
        "fmla v27.4s, v5.4s, v25.s[1]\n"   // 0C to idx 0
        "fmla v28.4s, v9.4s, v25.s[1]\n"   // 0B to idx 1
        "fmla v29.4s, v13.4s, v25.s[1]\n"   // 0B to idx 1
        "fmla v30.4s, v17.4s, v25.s[1]\n"   // 0A to idx 2
        "fmla v31.4s, v21.4s, v25.s[1]\n"   // 0A to idx 2
        "fmla v26.4s, v2.4s, v25.s[2]\n"   // 0C to idx 0
        "fmla v27.4s, v6.4s, v25.s[2]\n"   // 0C to idx 0
        "fmla v28.4s, v10.4s, v25.s[2]\n"   // 0B to idx 1
        "fmla v29.4s, v14.4s, v25.s[2]\n"   // 0B to idx 1
        "fmla v30.4s, v18.4s, v25.s[2]\n"   // 0A to idx 2
        "fmla v31.4s, v22.4s, v25.s[2]\n"   // 0A to idx 2
        "fmla v26.4s, v3.4s, v25.s[3]\n"   // 0C to idx 0
        "fmla v27.4s, v7.4s, v25.s[3]\n"   // 0C to idx 0
        "fmla v28.4s, v11.4s, v25.s[3]\n"   // 0B to idx 1
        "fmla v29.4s, v15.4s, v25.s[3]\n"   // 0B to idx 1
        "fmla v30.4s, v19.4s, v25.s[3]\n"   // 0A to idx 2
        "fmla v31.4s, v23.4s, v25.s[3]\n"   // 0A to idx 2

        "stp q26, q27, [x14, #-96]\n"
        "ldp q26, q27, [x14], #32\n"
        "ldr q25, [x13], #32\n" // Input 11

        "fmla v28.4s, v0.4s, v24.s[0]\n"   // 1C to idx 1
        "fmla v29.4s, v4.4s, v24.s[0]\n"   // 1C to idx 1
        "fmla v30.4s, v8.4s, v24.s[0]\n"   // 1B to idx 2
        "fmla v31.4s, v12.4s, v24.s[0]\n"   // 1B to idx 2
        "fmla v26.4s, v16.4s, v24.s[0]\n"   // 1A to idx 0
        "fmla v27.4s, v20.4s, v24.s[0]\n"   // 1A to idx 0
        "fmla v28.4s, v1.4s, v24.s[1]\n"   // 1C to idx 1
        "fmla v29.4s, v5.4s, v24.s[1]\n"   // 1C to idx 1
        "fmla v30.4s, v9.4s, v24.s[1]\n"   // 1B to idx 2
        "fmla v31.4s, v13.4s, v24.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v17.4s, v24.s[1]\n"   // 1A to idx 0
        "fmla v27.4s, v21.4s, v24.s[1]\n"   // 1A to idx 0
        "fmla v28.4s, v2.4s, v24.s[2]\n"   // 1C to idx 1
        "fmla v29.4s, v6.4s, v24.s[2]\n"   // 1C to idx 1
        "fmla v30.4s, v10.4s, v24.s[2]\n"   // 1B to idx 2
        "fmla v31.4s, v14.4s, v24.s[2]\n"   // 1B to idx 2
        "fmla v26.4s, v18.4s, v24.s[2]\n"   // 1A to idx 0
        "fmla v27.4s, v22.4s, v24.s[2]\n"   // 1A to idx 0
        "fmla v28.4s, v3.4s, v24.s[3]\n"   // 1C to idx 1
        "fmla v29.4s, v7.4s, v24.s[3]\n"   // 1C to idx 1
        "fmla v30.4s, v11.4s, v24.s[3]\n"   // 1B to idx 2
        "fmla v31.4s, v15.4s, v24.s[3]\n"   // 1B to idx 2
        "fmla v26.4s, v19.4s, v24.s[3]\n"   // 1A to idx 3
        "fmla v27.4s, v23.4s, v24.s[3]\n"   // 1A to idx 3
        
        "stp q28, q29, [x14, #-96]\n"
        "ldp q28, q29, [x14], #32\n"
        "ldr q24, [x13], #32\n" // Input 12

        "fmla v30.4s, v0.4s, v25.s[0]\n"   // 2C to idx 2
        "fmla v31.4s, v4.4s, v25.s[0]\n"   // 2C to idx 2
        "fmla v26.4s, v8.4s, v25.s[0]\n"   // 2B to idx 0
        "fmla v27.4s, v12.4s, v25.s[0]\n"   // 2B to idx 0
        "fmla v28.4s, v16.4s, v25.s[0]\n"   // 2A to idx 1
        "fmla v29.4s, v20.4s, v25.s[0]\n"   // 2A to idx 1
        "fmla v30.4s, v1.4s, v25.s[1]\n"   // 2C to idx 2
        "fmla v31.4s, v5.4s, v25.s[1]\n"   // 2C to idx 2
        "fmla v26.4s, v9.4s, v25.s[1]\n"   // 2B to idx 0
        "fmla v27.4s, v13.4s, v25.s[1]\n"   // 2B to idx 0
        "fmla v28.4s, v17.4s, v25.s[1]\n"   // 2A to idx 1
        "fmla v29.4s, v21.4s, v25.s[1]\n"   // 2A to idx 1
        "fmla v30.4s, v2.4s, v25.s[2]\n"   // 2C to idx 2
        "fmla v31.4s, v6.4s, v25.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v10.4s, v25.s[2]\n"   // 2B to idx 0
        "fmla v27.4s, v14.4s, v25.s[2]\n"   // 2B to idx 0
        "fmla v28.4s, v18.4s, v25.s[2]\n"   // 2A to idx 1
        "fmla v29.4s, v22.4s, v25.s[2]\n"   // 2A to idx 1
        "fmla v30.4s, v3.4s, v25.s[3]\n"   // 2C to idx 2
        "fmla v31.4s, v7.4s, v25.s[3]\n"   // 2C to idx 2
        "fmla v26.4s, v11.4s, v25.s[3]\n"   // 2B to idx 0
        "fmla v27.4s, v15.4s, v25.s[3]\n"   // 2B to idx 0
        "fmla v28.4s, v19.4s, v25.s[3]\n"   // 2A to idx 1
        "fmla v29.4s, v23.4s, v25.s[3]\n"   // 2A to idx 1

        "stp q30, q31, [x14, #-96]\n"
        "ldp q30, q31, [x14], #32\n"
        "ldr q25, [x13], #32\n" // Input 13

        "fmla v26.4s, v0.4s, v24.s[0]\n"   // 0C to idx 0
        "fmla v27.4s, v4.4s, v24.s[0]\n"   // 0C to idx 0
        "fmla v28.4s, v8.4s, v24.s[0]\n"   // 0B to idx 1
        "fmla v29.4s, v12.4s, v24.s[0]\n"   // 0B to idx 1
        "fmla v30.4s, v16.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v31.4s, v20.4s, v24.s[0]\n"   // 0A to idx 2
        "fmla v26.4s, v1.4s, v24.s[1]\n"   // 0C to idx 0
        "fmla v27.4s, v5.4s, v24.s[1]\n"   // 0C to idx 0
        "fmla v28.4s, v9.4s, v24.s[1]\n"   // 0B to idx 1
        "fmla v29.4s, v13.4s, v24.s[1]\n"   // 0B to idx 1
        "fmla v30.4s, v17.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v31.4s, v21.4s, v24.s[1]\n"   // 0A to idx 2
        "fmla v26.4s, v2.4s, v24.s[2]\n"   // 0C to idx 0
        "fmla v27.4s, v6.4s, v24.s[2]\n"   // 0C to idx 0
        "fmla v28.4s, v10.4s, v24.s[2]\n"   // 0B to idx 1
        "fmla v29.4s, v14.4s, v24.s[2]\n"   // 0B to idx 1
        "fmla v30.4s, v18.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v31.4s, v22.4s, v24.s[2]\n"   // 0A to idx 2
        "fmla v26.4s, v3.4s, v24.s[3]\n"   // 0C to idx 0
        "fmla v27.4s, v7.4s, v24.s[3]\n"   // 0C to idx 0
        "fmla v28.4s, v11.4s, v24.s[3]\n"   // 0B to idx 1
        "fmla v29.4s, v15.4s, v24.s[3]\n"   // 0B to idx 1
        "fmla v30.4s, v19.4s, v24.s[3]\n"   // 0A to idx 2
        "fmla v31.4s, v23.4s, v24.s[3]\n"   // 0A to idx 2

        "stp q26, q27, [x14, #-96]\n"
        "ldp q26, q27, [x14], #32\n"

        "fmla v28.4s, v0.4s, v25.s[0]\n"   // 1C to idx 1
        "fmla v29.4s, v4.4s, v25.s[0]\n"   // 1C to idx 1
        "fmla v30.4s, v8.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v31.4s, v12.4s, v25.s[0]\n"   // 1B to idx 2
        "fmla v26.4s, v16.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v27.4s, v20.4s, v25.s[0]\n"   // 1A to idx 0
        "fmla v28.4s, v1.4s, v25.s[1]\n"   // 1C to idx 1
        "fmla v29.4s, v5.4s, v25.s[1]\n"   // 1C to idx 1
        "fmla v30.4s, v9.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v31.4s, v13.4s, v25.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v17.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v27.4s, v21.4s, v25.s[1]\n"   // 1A to idx 0
        "fmla v28.4s, v2.4s, v25.s[2]\n"   // 1C to idx 1
        "fmla v29.4s, v6.4s, v25.s[2]\n"   // 1C to idx 1
        "fmla v30.4s, v10.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v31.4s, v14.4s, v25.s[2]\n"   // 1B to idx 2
        "fmla v26.4s, v18.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v27.4s, v22.4s, v25.s[2]\n"   // 1A to idx 0
        "fmla v28.4s, v3.4s, v25.s[3]\n"   // 1C to idx 1
        "fmla v29.4s, v7.4s, v25.s[3]\n"   // 1C to idx 1
        "fmla v30.4s, v11.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v31.4s, v15.4s, v25.s[3]\n"   // 1B to idx 2
        "fmla v26.4s, v19.4s, v25.s[3]\n"   // 1A to idx 3
        "fmla v27.4s, v23.4s, v25.s[3]\n"   // 1A to idx 3
        
        "stp q28, q29, [x14, #-96]\n"

        "add x12, x12, #14\n"
        "mov v28.16b, v26.16b\n"
        "mov v29.16b, v27.16b\n"
        "cmp %[mVal], x12\n"
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        
        "b.gt WIDX_LOOP_8TO8\n" // WIDX_LOOP_8TO8 EXIT
        "WIDX_LOOP_8TO8_EXIT:\n"

        "cmp %[pad], #1\n"
        "b.gt END_WIDX_8TO8_PAD_NOT_SKIP\n"
        "b.eq END_WIDX_8TO8_PAD1_SKIP\n"
        "sub x14, x14, #96\n"
        "b END_WIDX_8TO8_PAD_EXIT\n"
        "END_WIDX_8TO8_PAD_NOT_SKIP:\n"
        "stp q26, q27, [x14, #-64]\n"
        "stp q28, q29, [x14, #-32]\n"
        "b END_WIDX_8TO8_PAD_EXIT\n"
        "END_WIDX_8TO8_PAD1_SKIP:\n"
        "stp q26, q27, [x14, #-64]\n"
        "sub x14, x14, #32\n"
        "END_WIDX_8TO8_PAD_EXIT:\n"

        "add x11, x11, #1\n"
        "cmp x11, %[outH]\n"
        "b.lt HIDX_LOOP_8TO8\n" // HIDX_LOOP_8TO8 EXIT
        "HIDX_LOOP_8TO8_EXIT:"
        "add x9, x9, #4\n"
        "cmp x9, #8\n"
        "b.lt VIDX_LOOP_8TO8\n" // VIDX_LOOP_8TO8 EXIT
        "add x10, x10, #1\n"
        "cmp x10, #3\n"
        "b.lt FIDX_LOOP_8TO8\n" // FIDX_LOOP_8TO8 EXIT
        "EXIT_8TO8:\n"
    :  
    :   [inP] "r" (input), [filP] "r" (filter), [outP] "r" (output), [inW] "r" (inputWidth), [startH] "r" ((int64_t)inputHStart), [inH] "r" (inputHeight), [outW] "r" (outWidth),
        [outH] "r" (iterHeight), [mVal] "r" (moduloVal), [filS] "r" (filterBlockSize), [pad] "r" (padding)
    :   "x9", "x10", "x11", "x12", "x13", "x14",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "cc"
    );
    #ifdef __DEBUG_MTEN_OFF
        for (int i = 0; i < 128; i++)
        {
            printf("%6.3f\t", testArr[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
        for (int i = 0; i < 128; i++)
        {
            printf("%d\t", testArr2[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
    #endif
}

void aarch64_convKernel3x3_vec3_vec8_iterRow_ASM(float* input, float* filter, float* output, int inputWidth, int inputHeight, int inputHStart, int outWidth, int iterHeight, int padding, int filterBlockSize)
{
    // 11 general purpose registers for input, 3 genneral purpose registers for fIdx, hIdx, wIdx (x9, x10, x11), 2 genneral purpose registers for calculation (x12, x13).
    // Total of 16 general purpose registers.
    // 18 NEON vector registers for filter values (v0 ~ v17), 3 NEON vector registers for input values (v18, 19, v20), 6 NEON vector registers for output values (v21 ~ v26)
    // Total of 27 NEON vector registers.
    const int moduloVal = inputWidth - inputWidth%14;

    // float testArr[128] = {0};
    // int testArr2[128] = {0};
    __asm __volatile (
        "mov x9, #0\n" // x9 = fIdx
        "FIDX_LOOP_3TO8:\n"
        
        "add x11, x9, x9, lsl #3\n" // x11 = 9*fIdx
        "mov x10, #0\n" // x10 = hIdx (Referenced to Output)
        "sub x13, %[filS], #6\n"
        "add x12, %[startH], x9\n"
        "add x11, %[filP], x11, lsl #2\n" // x11 = %[filP] + 36*fIdx
        "lsl x13, x13, #2\n"
        "ld3 {v12.s-v14.s}[0], [x11], #12\n"
        "ld3 {v6.s-v8.s}[0], [x11], #12\n"
        "ld3 {v0.s-v2.s}[0], [x11], x13\n"
        "ld3 {v12.s-v14.s}[1], [x11], #12\n"
        "ld3 {v6.s-v8.s}[1], [x11], #12\n"
        "ld3 {v0.s-v2.s}[1], [x11], x13\n"
        "ld3 {v12.s-v14.s}[2], [x11], #12\n"
        "ld3 {v6.s-v8.s}[2], [x11], #12\n"
        "ld3 {v0.s-v2.s}[2], [x11], x13\n"
        "ld3 {v12.s-v14.s}[3], [x11], #12\n"
        "ld3 {v6.s-v8.s}[3], [x11], #12\n"
        "ld3 {v0.s-v2.s}[3], [x11], x13\n"
        
        "ld3 {v15.s-v17.s}[0], [x11], #12\n"
        "ld3 {v9.s-v11.s}[0], [x11], #12\n"
        "ld3 {v3.s-v5.s}[0], [x11], x13\n"
        "ld3 {v15.s-v17.s}[1], [x11], #12\n"
        "ld3 {v9.s-v11.s}[1], [x11], #12\n"
        "ld3 {v3.s-v5.s}[1], [x11], x13\n"
        "ld3 {v15.s-v17.s}[2], [x11], #12\n"
        "ld3 {v9.s-v11.s}[2], [x11], #12\n"
        "ld3 {v3.s-v5.s}[2], [x11], x13\n"
        "ld3 {v15.s-v17.s}[3], [x11], #12\n"
        "ld3 {v9.s-v11.s}[3], [x11], #12\n"
        "ld3 {v3.s-v5.s}[3], [x11]\n"

        "cmp x12, #0\n"
        "add x11, %[inW], %[inW], lsl #1\n" // x11 = inputWidth * 3
        "b.ge NO_PAD_SKIPPING_3TO8\n"
        "sub x10, x10, x12\n"
        "NO_PAD_SKIPPING_3TO8:"
        "cmp x10, %[outH]\n"
        "mul x13, x10, %[outW]\n"
        "b.ge HIDX_LOOP_3TO8_EXIT\n" // HIDX_LOOP_3TO8 EXIT
        "add x12, x10, x9\n"
        "mul x12, x12, x11\n"
        "add x13, %[outP], x13, lsl# 5\n" // x13 = Output Ptr
        "add x12, %[inP], x12, lsl #2\n" // x12 = Input Ptr
        
        "HIDX_LOOP_3TO8:\n"
        "add x11, x9, %[startH]\n"
        "add x11, x11, x10\n"
        "cmp x11, %[inH]\n"
        "b.ge HIDX_LOOP_3TO8_EXIT\n" // HIDX_LOOP_3TO8 EXIT
        "mov x11, #0\n" // x11 = wIdx

        "prfm pldl1strm, [x12]\n"
        "cmp %[inW], #14\n"
        "b.lt WIDX_LOOP_3TO8_EXIT\n"

        "cmp %[pad], #1\n"
        "b.gt START_WIDX_3TO8_PAD_NOT_SKIP\n"
        "b.eq START_WIDX_3TO8_PAD1_SKIP\n"
        "b WIDX_LOOP_3TO8\n"
        "START_WIDX_3TO8_PAD_NOT_SKIP:"
        "ldp q21, q22, [x13], #32\n"
        "START_WIDX_3TO8_PAD1_SKIP:\n"
        "ldp q23, q24, [x13], #32\n"
        "WIDX_LOOP_3TO8:\n"
        // C: filvecW0 - v0 ~ v5,  B: filvecW1 - v6 ~ v11, A: filvecW2 - v12 ~ v17, 
        // inputVec0 - v18, inputVec1 - v19, inputVec2 - v20, outVec00 - v21, outVec01 - v22, outVec10 - v23, outVec11 - v24, outVec20 - v25, outVec21 - v26
        "ld3 {v18.4s-v20.4s}, [x12], #48\n" // Input 0, 1, 2, 3 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)

        "b.gt WIDX_3TO8_PAD_NOT_SKIP\n"
        "b.eq WIDX_3TO8_PAD1_SKIP\n"
        "ldp q25, q26, [x13], #32\n"
        "fmla v25.4s, v0.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v3.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v25.4s, v1.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v4.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v25.4s, v2.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v5.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        "stp q25, q26, [x13, #-32]\n"
        "b WIDX_3TO8_PAD_SKIP_END\n"
        "WIDX_3TO8_PAD1_SKIP:\n"
        "ldp q25, q26, [x13], #32\n"
        "fmla v25.4s, v0.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v3.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v0.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v3.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v23.4s, v6.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v9.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v25.4s, v1.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v4.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v1.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v4.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v7.4s, v19.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v10.4s, v19.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v25.4s, v2.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v5.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v2.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v5.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v8.4s, v20.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v11.4s, v20.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        "stp q23, q24, [x13, #-64]\n"
        "stp q25, q26, [x13, #-32]\n"
        "b WIDX_3TO8_PAD_SKIP_END\n"
        "WIDX_3TO8_PAD_NOT_SKIP:\n"
        "ldp q25, q26, [x13], #32\n"
        "fmla v21.4s, v0.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v3.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v0.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v3.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v0.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v3.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v23.4s, v6.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v9.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v1.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v4.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v1.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v4.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v1.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v4.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v7.4s, v19.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v10.4s, v19.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v2.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v5.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v2.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v5.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v2.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v5.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v8.4s, v20.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v11.4s, v20.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        "stp q21, q22, [x13, #-96]\n"
        "stp q23, q24, [x13, #-64]\n"
        "stp q25, q26, [x13, #-32]\n"
        "WIDX_3TO8_PAD_SKIP_END:\n"
        "ldp q25, q26, [x13], #32\n"
        "ldp q21, q22, [x13], #32\n"
        "ldp q23, q24, [x13], #32\n"
        
        "fmla v25.4s, v0.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v3.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v21.4s, v6.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v9.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v25.4s, v6.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v9.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v21.4s, v12.4s, v18.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v15.4s, v18.s[2]\n"  // 2A to idx 0
        "fmla v25.4s, v12.4s, v18.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v15.4s, v18.s[1]\n"  // 1A to idx 2  
        "fmla v23.4s, v12.4s, v18.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v15.4s, v18.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v1.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v4.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v21.4s, v7.4s, v19.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v10.4s, v19.s[3]\n"  // 3B to idx 0
        "fmla v25.4s, v7.4s, v19.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v10.4s, v19.s[2]\n"  // 2B to idx 2
        "fmla v21.4s, v13.4s, v19.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v16.4s, v19.s[2]\n"  // 2A to idx 0
        "fmla v25.4s, v13.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v16.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v13.4s, v19.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v16.4s, v19.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v2.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v5.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v21.4s, v8.4s, v20.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v11.4s, v20.s[3]\n"  // 3B to idx 0
        "fmla v25.4s, v8.4s, v20.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v11.4s, v20.s[2]\n"  // 2B to idx 2
        "fmla v21.4s, v14.4s, v20.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v17.4s, v20.s[2]\n"  // 2A to idx 0
        "fmla v25.4s, v14.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v17.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v14.4s, v20.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v17.4s, v20.s[3]\n"  // 3A to idx 1
        "ld3 {v18.4s-v20.4s}, [x12], #48\n" // Input 4, 5, 6, 7 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)
        "stp q25, q26, [x13, #-96]\n"
        "ldp q25, q26, [x13], #32\n"
        "fmla v21.4s, v0.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v3.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v0.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v3.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v0.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v3.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v23.4s, v6.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v9.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v1.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v4.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v1.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v4.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v1.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v4.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v7.4s, v19.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v10.4s, v19.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v2.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v5.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v2.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v5.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v2.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v5.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v8.4s, v20.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v11.4s, v20.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        
        "stp q21, q22, [x13, #-96]\n"
        "stp q23, q24, [x13, #-64]\n"
        "stp q25, q26, [x13, #-32]\n"
        "ldp q25, q26, [x13], #32\n"
        "ldp q21, q22, [x13], #32\n"
        "ldp q23, q24, [x13], #32\n"
        
        "fmla v25.4s, v0.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v3.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v21.4s, v6.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v9.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v25.4s, v6.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v9.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v21.4s, v12.4s, v18.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v15.4s, v18.s[2]\n"  // 2A to idx 0
        "fmla v25.4s, v12.4s, v18.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v15.4s, v18.s[1]\n"  // 1A to idx 2  
        "fmla v23.4s, v12.4s, v18.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v15.4s, v18.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v1.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v4.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v21.4s, v7.4s, v19.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v10.4s, v19.s[3]\n"  // 3B to idx 0
        "fmla v25.4s, v7.4s, v19.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v10.4s, v19.s[2]\n"  // 2B to idx 2
        "fmla v21.4s, v13.4s, v19.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v16.4s, v19.s[2]\n"  // 2A to idx 0
        "fmla v25.4s, v13.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v16.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v13.4s, v19.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v16.4s, v19.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v2.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v5.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v21.4s, v8.4s, v20.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v11.4s, v20.s[3]\n"  // 3B to idx 0
        "fmla v25.4s, v8.4s, v20.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v11.4s, v20.s[2]\n"  // 2B to idx 2
        "fmla v21.4s, v14.4s, v20.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v17.4s, v20.s[2]\n"  // 2A to idx 0
        "fmla v25.4s, v14.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v17.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v14.4s, v20.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v17.4s, v20.s[3]\n"  // 3A to idx 1
        "ld3 {v18.4s-v20.4s}, [x12], #48\n" // Input 8, 9, 10, 11 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)
        "stp q25, q26, [x13, #-96]\n"
        "ldp q25, q26, [x13], #32\n"
        "fmla v21.4s, v0.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v3.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v0.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v3.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v0.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v3.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v23.4s, v6.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v9.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v1.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v4.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v1.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v4.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v1.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v4.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v7.4s, v19.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v10.4s, v19.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v2.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v5.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v2.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v5.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v23.4s, v2.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v5.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v8.4s, v20.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v11.4s, v20.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        
        "stp q21, q22, [x13, #-96]\n"
        "stp q23, q24, [x13, #-64]\n"
        "stp q25, q26, [x13, #-32]\n"
        "ldp q25, q26, [x13], #32\n"
        "ldp q21, q22, [x13], #32\n"
        "ldp q23, q24, [x13], #32\n"
        
        "fmla v25.4s, v0.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v3.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v21.4s, v6.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v9.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v25.4s, v6.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v9.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v21.4s, v12.4s, v18.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v15.4s, v18.s[2]\n"  // 2A to idx 0
        "fmla v25.4s, v12.4s, v18.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v15.4s, v18.s[1]\n"  // 1A to idx 2  
        "fmla v23.4s, v12.4s, v18.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v15.4s, v18.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v1.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v4.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v21.4s, v7.4s, v19.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v10.4s, v19.s[3]\n"  // 3B to idx 0
        "fmla v25.4s, v7.4s, v19.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v10.4s, v19.s[2]\n"  // 2B to idx 2
        "fmla v21.4s, v13.4s, v19.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v16.4s, v19.s[2]\n"  // 2A to idx 0
        "fmla v25.4s, v13.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v16.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v13.4s, v19.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v16.4s, v19.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v2.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v5.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v21.4s, v8.4s, v20.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v11.4s, v20.s[3]\n"  // 3B to idx 0
        "fmla v25.4s, v8.4s, v20.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v11.4s, v20.s[2]\n"  // 2B to idx 2
        "fmla v21.4s, v14.4s, v20.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v17.4s, v20.s[2]\n"  // 2A to idx 0
        "fmla v25.4s, v14.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v17.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v14.4s, v20.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v17.4s, v20.s[3]\n"  // 3A to idx 1
        "ld3 {v18.2s-v20.2s}, [x12], #24\n" // Input 12, 13 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)
        "stp q25, q26, [x13, #-96]\n"
        "ldp q25, q26, [x13], #32\n"

        "fmla v21.4s, v0.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v3.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v23.4s, v0.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v3.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v23.4s, v6.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v9.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v1.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v4.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v23.4s, v1.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v4.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v7.4s, v19.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v10.4s, v19.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v2.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v5.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v23.4s, v2.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v5.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v23.4s, v8.4s, v20.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v11.4s, v20.s[0]\n"  // 0B to idx 1
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        
        "stp q21, q22, [x13, #-96]\n"
        "stp q23, q24, [x13, #-64]\n"
        "ldp q23, q24, [x13], #32\n"
        "mov v21.16b, v25.16b\n"
        "mov v22.16b, v26.16b\n"
        
        "fmla v23.4s, v12.4s, v18.s[1]\n"  // 1A to idx 1
        "fmla v24.4s, v15.4s, v18.s[1]\n"  // 1A to idx 1  

        "fmla v23.4s, v13.4s, v19.s[1]\n"  // 1A to idx 1
        "fmla v24.4s, v16.4s, v19.s[1]\n"  // 1A to idx 1

        "fmla v23.4s, v14.4s, v20.s[1]\n"  // 1A to idx 1
        "fmla v24.4s, v17.4s, v20.s[1]\n"  // 1A to idx 1
        
        "add x11, x11, #14\n"
        "cmp %[mVal], x11\n"
        "b.gt WIDX_LOOP_3TO8\n" // WIDX_LOOP_3TO8 EXIT
        "WIDX_LOOP_3TO8_EXIT:\n"

        "cmp %[pad], #1\n"
        "b.gt END_WIDX_3TO8_PAD_NOT_SKIP\n"
        "b.eq END_WIDX_3TO8_PAD1_SKIP\n"
        "sub x13, x13, #96\n"
        "b END_WIDX_3TO8_PAD_EXIT\n"
        "END_WIDX_3TO8_PAD_NOT_SKIP:\n"
        "stp q21, q22, [x13, #-64]\n"
        "stp q23, q24, [x13, #-32]\n"
        "b END_WIDX_3TO8_PAD_EXIT\n"
        "END_WIDX_3TO8_PAD1_SKIP:\n"
        "stp q21, q22, [x13, #-64]\n"
        "sub x13, x13, #32\n"
        "END_WIDX_3TO8_PAD_EXIT:\n"

        "add x10, x10, #1\n"
        "cmp x10, %[outH]\n"
        "b.lt HIDX_LOOP_3TO8\n" // HIDX_LOOP_3TO8 EXIT
        "HIDX_LOOP_3TO8_EXIT:"
        "add x9, x9, #1\n"
        "cmp x9, #3\n"
        "b.lt FIDX_LOOP_3TO8\n" // FIDX_LOOP_3TO8 EXIT
        "EXIT_3TO8:\n"
    :  
    :   [inP] "r" (input), [filP] "r" (filter), [outP] "r" (output), [inW] "r" (inputWidth), [startH] "r" ((int64_t)inputHStart), [inH] "r" (inputHeight), [outW] "r" (outWidth),
        [outH] "r" (iterHeight), [mVal] "r" (moduloVal), [filS] "r" (filterBlockSize), [pad] "r" (padding)
    :   "x9", "x10", "x11", "x12", "x13",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "cc"
    );
    #ifdef __DEBUG_MTEN_OFF
        for (int i = 0; i < 128; i++)
        {
            printf("%6.3f\t", testArr[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
        for (int i = 0; i < 128; i++)
        {
            printf("%d\t", testArr2[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
    #endif
}

void aarch64_convKernel3x3_vec3_vec8_iterRow_Stride2_ASM(float* input, float* filter, float* output, int inputWidth, int inputHeight, int inputHStart, int outWidth, int iterHeight, int padding, int filterBlockSize)
{
    // 11 general purpose registers for input, 3 genneral purpose registers for fIdx, hIdx, wIdx (x9, x10, x11), 2 genneral purpose registers for calculation (x12, x13).
    // Total of 16 general purpose registers.
    // 18 NEON vector registers for filter values (v0 ~ v17), 3 NEON vector registers for input values (v18, 19, v20), 6 NEON vector registers for output values (v21 ~ v26)
    // Total of 27 NEON vector registers.
    const int moduloVal = inputWidth - inputWidth%14;

    // float testArr[128] = {0};
    // int testArr2[128] = {0};
    __asm __volatile (
        "mov x9, #0\n" // x9 = fIdx
        "FIDX_LOOP_3TO8_STRIDE2:\n"
        
        "add x11, x9, x9, lsl #3\n" // x11 = 9*fIdx
        "mov x10, #0\n" // x10 = hIdx (Referenced to Output)
        "sub x13, %[filS], #6\n"
        "add x12, %[startH], x9\n"
        "add x11, %[filP], x11, lsl #2\n" // x11 = %[filP] + 36*fIdx
        "lsl x13, x13, #2\n"
        "ld3 {v12.s-v14.s}[0], [x11], #12\n"
        "ld3 {v6.s-v8.s}[0], [x11], #12\n"
        "ld3 {v0.s-v2.s}[0], [x11], x13\n"
        "ld3 {v12.s-v14.s}[1], [x11], #12\n"
        "ld3 {v6.s-v8.s}[1], [x11], #12\n"
        "ld3 {v0.s-v2.s}[1], [x11], x13\n"
        "ld3 {v12.s-v14.s}[2], [x11], #12\n"
        "ld3 {v6.s-v8.s}[2], [x11], #12\n"
        "ld3 {v0.s-v2.s}[2], [x11], x13\n"
        "ld3 {v12.s-v14.s}[3], [x11], #12\n"
        "ld3 {v6.s-v8.s}[3], [x11], #12\n"
        "ld3 {v0.s-v2.s}[3], [x11], x13\n"
        
        "ld3 {v15.s-v17.s}[0], [x11], #12\n"
        "ld3 {v9.s-v11.s}[0], [x11], #12\n"
        "ld3 {v3.s-v5.s}[0], [x11], x13\n"
        "ld3 {v15.s-v17.s}[1], [x11], #12\n"
        "ld3 {v9.s-v11.s}[1], [x11], #12\n"
        "ld3 {v3.s-v5.s}[1], [x11], x13\n"
        "ld3 {v15.s-v17.s}[2], [x11], #12\n"
        "ld3 {v9.s-v11.s}[2], [x11], #12\n"
        "ld3 {v3.s-v5.s}[2], [x11], x13\n"
        "ld3 {v15.s-v17.s}[3], [x11], #12\n"
        "ld3 {v9.s-v11.s}[3], [x11], #12\n"
        "ld3 {v3.s-v5.s}[3], [x11]\n"

        "cmp x12, #0\n"
        "add x11, %[inW], %[inW], lsl #1\n" // x11 = inputWidth * 3
        "b.ge NO_PAD_SKIPPING_3TO8_STRIDE2\n"
        "add x10, x10, #1\n"
        "NO_PAD_SKIPPING_3TO8_STRIDE2:"
        "cmp x10, %[outH]\n"
        "mul x13, x10, %[outW]\n"
        "add x12, x12, x11, lsl #2\n"
        "add x12, x9, x10, lsl #1\n"
        "mul x12, x12, x11\n"
        "add x13, %[outP], x13, lsl# 5\n" // x13 = Output Ptr
        "add x12, %[inP], x12, lsl #2\n" // x12 = Input Ptr

        "HIDX_LOOP_3TO8_STRIDE2:\n"
        "add x11, x9, %[startH]\n"
        "add x11, x11, x10, lsl #1\n"
        "cmp x11, %[inH]\n"
        "b.ge HIDX_LOOP_3TO8_STRIDE2_EXIT\n" // HIDX_LOOP_3TO8_STRIDE2 EXIT
        "mov x11, #0\n" // x11 = wIdx

        "prfm pldl1strm, [x12]\n"
        "cmp %[inW], #14\n"
        "b.lt WIDX_LOOP_3TO8_STRIDE2_EXIT\n"

        "cmp %[pad], #1\n"
        "b.lt WIDX_LOOP_3TO8_STRIDE2\n"
        "b.eq START_WIDX_3TO8_STRIDE2_PAD1_SKIP\n"
        "ldp q21, q22, [x13], #32\n"
        "WIDX_LOOP_3TO8_STRIDE2:\n"
        // C: filvecW0 - v0 ~ v5,  B: filvecW1 - v6 ~ v11, A: filvecW2 - v12 ~ v17, 
        // inputVec0 - v18, inputVec1 - v19, inputVec2 - v20, outVec00 - v21, outVec01 - v22, outVec10 - v23, outVec11 - v24, outVec20 - v25, outVec21 - v26
        "ld3 {v18.4s-v20.4s}, [x12], #48\n" // Input 0, 1, 2, 3 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)

        "b.gt WIDX_3TO8_STRIDE2_PAD_NOT_SKIP\n"
        "ldp q25, q26, [x13], #32\n"
        "fmla v25.4s, v0.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v3.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v25.4s, v1.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v4.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v25.4s, v2.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v5.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        "stp q25, q26, [x13, #-32]\n"
        "b WIDX_3TO8_STRIDE2_PAD_SKIP_END\n"
        "WIDX_3TO8_STRIDE2_PAD_NOT_SKIP:\n"
        "ldp q25, q26, [x13], #32\n"
        "fmla v21.4s, v0.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v3.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v0.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v3.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v1.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v4.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v1.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v4.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v2.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v5.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v2.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v5.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        "stp q21, q22, [x13, #-64]\n"
        "stp q25, q26, [x13, #-32]\n"
        "WIDX_3TO8_STRIDE2_PAD_SKIP_END:\n"
        "ldp q21, q22, [x13], #32\n"
   
        "fmla v21.4s, v6.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v9.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v21.4s, v12.4s, v18.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v15.4s, v18.s[2]\n"  // 2A to idx 0

        "fmla v21.4s, v7.4s, v19.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v10.4s, v19.s[3]\n"  // 3B to idx 0
        "fmla v21.4s, v13.4s, v19.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v16.4s, v19.s[2]\n"  // 2A to idx 0

        "fmla v21.4s, v8.4s, v20.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v11.4s, v20.s[3]\n"  // 3B to idx 0
        "fmla v21.4s, v14.4s, v20.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v17.4s, v20.s[2]\n"  // 2A to idx 0
        "ld3 {v18.4s-v20.4s}, [x12], #48\n" // Input 4, 5, 6, 7 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)
        "ldp q25, q26, [x13], #32\n"
        "fmla v21.4s, v0.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v3.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v0.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v3.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v1.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v4.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v1.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v4.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v2.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v5.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v2.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v5.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        
        "stp q21, q22, [x13, #-64]\n"
        "stp q25, q26, [x13, #-32]\n"
        "ldp q21, q22, [x13], #32\n"
        
        "fmla v21.4s, v6.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v9.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v21.4s, v12.4s, v18.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v15.4s, v18.s[2]\n"  // 2A to idx 0

        "fmla v21.4s, v7.4s, v19.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v10.4s, v19.s[3]\n"  // 3B to idx 0
        "fmla v21.4s, v13.4s, v19.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v16.4s, v19.s[2]\n"  // 2A to idx 0

        "fmla v21.4s, v8.4s, v20.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v11.4s, v20.s[3]\n"  // 3B to idx 0
        "fmla v21.4s, v14.4s, v20.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v17.4s, v20.s[2]\n"  // 2A to idx 0
        "ld3 {v18.4s-v20.4s}, [x12], #48\n" // Input 8, 9, 10, 11 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)
        "ldp q25, q26, [x13], #32\n"
        "fmla v21.4s, v0.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v3.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v0.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v3.4s, v18.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v1.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v4.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v1.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v4.4s, v19.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v2.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v5.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v2.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v26.4s, v5.4s, v20.s[2]\n"   // 2C to idx 2
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        
        "stp q21, q22, [x13, #-64]\n"
        "stp q25, q26, [x13, #-32]\n"
        "ldp q21, q22, [x13], #32\n"
        
        "fmla v21.4s, v6.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v9.4s, v18.s[3]\n"   // 3B to idx 0
        "fmla v21.4s, v12.4s, v18.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v15.4s, v18.s[2]\n"  // 2A to idx 0

        "fmla v21.4s, v7.4s, v19.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v10.4s, v19.s[3]\n"  // 3B to idx 0
        "fmla v21.4s, v13.4s, v19.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v16.4s, v19.s[2]\n"  // 2A to idx 0

        "fmla v21.4s, v8.4s, v20.s[3]\n"   // 3B to idx 0
        "fmla v22.4s, v11.4s, v20.s[3]\n"  // 3B to idx 0
        "fmla v21.4s, v14.4s, v20.s[2]\n"  // 2A to idx 0
        "fmla v22.4s, v17.4s, v20.s[2]\n"  // 2A to idx 0
        "ld3 {v18.2s-v20.2s}, [x12], #24\n" // Input 12, 13 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)
        "ldp q25, q26, [x13], #32\n"

        "fmla v21.4s, v0.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v3.4s, v18.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v6.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v9.4s, v18.s[1]\n"   // 1B to idx 2
        "fmla v25.4s, v12.4s, v18.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v15.4s, v18.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v1.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v4.4s, v19.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v7.4s, v19.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v10.4s, v19.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v13.4s, v19.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v16.4s, v19.s[0]\n"  // 0A to idx 2

        "fmla v21.4s, v2.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v22.4s, v5.4s, v20.s[0]\n"   // 0C to idx 0
        "fmla v25.4s, v8.4s, v20.s[1]\n"   // 1B to idx 2
        "fmla v26.4s, v11.4s, v20.s[1]\n"  // 1B to idx 2
        "fmla v25.4s, v14.4s, v20.s[0]\n"  // 0A to idx 2
        "fmla v26.4s, v17.4s, v20.s[0]\n"  // 0A to idx 2
        
        "stp q21, q22, [x13, #-64]\n"
        "mov v21.16b, v25.16b\n"
        "mov v22.16b, v26.16b\n"
        
        "add x11, x11, #14\n"
        "cmp %[mVal], x11\n"
        "b.gt WIDX_LOOP_3TO8_STRIDE2\n" // WIDX_LOOP_3TO8_STRIDE2 EXIT
        "b WIDX_LOOP_3TO8_STRIDE2_EXIT\n"
        "START_WIDX_3TO8_STRIDE2_PAD1_SKIP:\n"
        "ldp q23, q24, [x13], #32\n"
        "WIDX_LOOP_PAD1_3TO8_STRIDE2:\n"
        "ld3 {v18.4s-v20.4s}, [x12], #48\n" // Input 0, 1, 2, 3 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)

        "fmla v23.4s, v0.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v3.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v23.4s, v6.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v9.4s, v18.s[0]\n"   // 0B to idx 1

        "fmla v23.4s, v1.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v4.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v23.4s, v7.4s, v19.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v10.4s, v19.s[0]\n"  // 0B to idx 1

        "fmla v23.4s, v2.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v5.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v23.4s, v8.4s, v20.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v11.4s, v20.s[0]\n"  // 0B to idx 1
        "stp q23, q24, [x13, #-32]\n"

        "ldp q25, q26, [x13], #32\n"
        "ldp q23, q24, [x13], #32\n"
        
        "fmla v25.4s, v0.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v3.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v25.4s, v6.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v9.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v25.4s, v12.4s, v18.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v15.4s, v18.s[1]\n"  // 1A to idx 2  
        "fmla v23.4s, v12.4s, v18.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v15.4s, v18.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v1.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v4.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v25.4s, v7.4s, v19.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v10.4s, v19.s[2]\n"  // 2B to idx 2
        "fmla v25.4s, v13.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v16.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v13.4s, v19.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v16.4s, v19.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v2.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v5.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v25.4s, v8.4s, v20.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v11.4s, v20.s[2]\n"  // 2B to idx 2
        "fmla v25.4s, v14.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v17.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v14.4s, v20.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v17.4s, v20.s[3]\n"  // 3A to idx 1
        "ld3 {v18.4s-v20.4s}, [x12], #48\n" // Input 4, 5, 6, 7 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)
        "stp q25, q26, [x13, #-64]\n"

        "fmla v23.4s, v0.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v3.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v23.4s, v6.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v9.4s, v18.s[0]\n"   // 0B to idx 1

        "fmla v23.4s, v1.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v4.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v23.4s, v7.4s, v19.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v10.4s, v19.s[0]\n"  // 0B to idx 1

        "fmla v23.4s, v2.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v5.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v23.4s, v8.4s, v20.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v11.4s, v20.s[0]\n"  // 0B to idx 1
        "stp q23, q24, [x13, #-32]\n"

        "ldp q25, q26, [x13], #32\n"
        "ldp q23, q24, [x13], #32\n"
        
        "fmla v25.4s, v0.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v3.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v25.4s, v6.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v9.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v25.4s, v12.4s, v18.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v15.4s, v18.s[1]\n"  // 1A to idx 2  
        "fmla v23.4s, v12.4s, v18.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v15.4s, v18.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v1.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v4.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v25.4s, v7.4s, v19.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v10.4s, v19.s[2]\n"  // 2B to idx 2
        "fmla v25.4s, v13.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v16.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v13.4s, v19.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v16.4s, v19.s[3]\n"  // 3A to idx 1

        "fmla v25.4s, v2.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v5.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v25.4s, v8.4s, v20.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v11.4s, v20.s[2]\n"  // 2B to idx 2
        "fmla v25.4s, v14.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v17.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v23.4s, v14.4s, v20.s[3]\n"  // 3A to idx 1
        "fmla v24.4s, v17.4s, v20.s[3]\n"  // 3A to idx 1
        "ld3 {v18.4s-v20.4s}, [x12], #48\n" // Input 8, 9, 10, 11 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)
        "stp q25, q26, [x13, #-64]\n"

        "fmla v23.4s, v0.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v3.4s, v18.s[1]\n"   // 1C to idx 1
        "fmla v23.4s, v6.4s, v18.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v9.4s, v18.s[0]\n"   // 0B to idx 1

        "fmla v23.4s, v1.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v4.4s, v19.s[1]\n"   // 1C to idx 1
        "fmla v23.4s, v7.4s, v19.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v10.4s, v19.s[0]\n"  // 0B to idx 1

        "fmla v23.4s, v2.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v24.4s, v5.4s, v20.s[1]\n"   // 1C to idx 1
        "fmla v23.4s, v8.4s, v20.s[0]\n"   // 0B to idx 1
        "fmla v24.4s, v11.4s, v20.s[0]\n"  // 0B to idx 1
        "stp q23, q24, [x13, #-32]\n"

        "ldp q25, q26, [x13], #32\n"
        "ldp q21, q22, [x13], #32\n"
        
        "fmla v25.4s, v0.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v3.4s, v18.s[3]\n"   // 3C to idx 2
        "fmla v25.4s, v6.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v9.4s, v18.s[2]\n"   // 2B to idx 2
        "fmla v25.4s, v12.4s, v18.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v15.4s, v18.s[1]\n"  // 1A to idx 2  
        "fmla v21.4s, v12.4s, v18.s[3]\n"  // 3A to idx 0
        "fmla v22.4s, v15.4s, v18.s[3]\n"  // 3A to idx 0

        "fmla v25.4s, v1.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v4.4s, v19.s[3]\n"   // 3C to idx 2
        "fmla v25.4s, v7.4s, v19.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v10.4s, v19.s[2]\n"  // 2B to idx 2
        "fmla v25.4s, v13.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v16.4s, v19.s[1]\n"  // 1A to idx 2
        "fmla v21.4s, v13.4s, v19.s[3]\n"  // 3A to idx 0
        "fmla v22.4s, v16.4s, v19.s[3]\n"  // 3A to idx 0

        "fmla v25.4s, v2.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v26.4s, v5.4s, v20.s[3]\n"   // 3C to idx 2
        "fmla v25.4s, v8.4s, v20.s[2]\n"   // 2B to idx 2
        "fmla v26.4s, v11.4s, v20.s[2]\n"  // 2B to idx 2
        "fmla v25.4s, v14.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v26.4s, v17.4s, v20.s[1]\n"  // 1A to idx 2
        "fmla v21.4s, v14.4s, v20.s[3]\n"  // 3A to idx 0
        "fmla v22.4s, v17.4s, v20.s[3]\n"  // 3A to idx 0
        "ld3 {v18.2s-v20.2s}, [x12], #24\n" // Input 12, 13 (v18 - Channel0, v19 - Channel 1, v20 - Channel 2)
        "stp q25, q26, [x13, #-64]\n"
        "ldp q23, q24, [x13], #32\n"
        "fmla v21.4s, v0.4s, v18.s[1]\n"   // 1C to idx 0
        "fmla v22.4s, v3.4s, v18.s[1]\n"   // 1C to idx 0
        "fmla v21.4s, v6.4s, v18.s[0]\n"   // 0B to idx 0
        "fmla v22.4s, v9.4s, v18.s[0]\n"   // 0B to idx 0
        "fmla v23.4s, v12.4s, v18.s[1]\n"  // 1A to idx 1
        "fmla v24.4s, v15.4s, v18.s[1]\n"  // 1A to idx 1 

        "fmla v21.4s, v1.4s, v19.s[1]\n"   // 1C to idx 0
        "fmla v22.4s, v4.4s, v19.s[1]\n"   // 1C to idx 0
        "fmla v21.4s, v7.4s, v19.s[0]\n"   // 0B to idx 0
        "fmla v22.4s, v10.4s, v19.s[0]\n"  // 0B to idx 0
        "fmla v23.4s, v13.4s, v19.s[1]\n"  // 1A to idx 1
        "fmla v24.4s, v16.4s, v19.s[1]\n"  // 1A to idx 1

        "fmla v21.4s, v2.4s, v20.s[1]\n"   // 1C to idx 0
        "fmla v22.4s, v5.4s, v20.s[1]\n"   // 1C to idx 0
        "fmla v21.4s, v8.4s, v20.s[0]\n"   // 0B to idx 0
        "fmla v22.4s, v11.4s, v20.s[0]\n"  // 0B to idx 0
        "fmla v23.4s, v14.4s, v20.s[1]\n"  // 1A to idx 1
        "fmla v24.4s, v17.4s, v20.s[1]\n"  // 1A to idx 1
        "stp q21, q22, [x13, #-64]\n"
        
        "add x11, x11, #14\n"
        "cmp %[mVal], x11\n"
        "b.gt WIDX_LOOP_PAD1_3TO8_STRIDE2\n" // WIDX_LOOP_3TO8_STRIDE2 EXIT
        "WIDX_LOOP_3TO8_STRIDE2_EXIT:\n"

        "cmp %[pad], #1\n"
        "b.eq END_WIDX_3TO8_STRIDE2_PAD1_SKIP\n"
        "b.gt END_WIDX_3TO8_STRIDE2_PAD_NOT_SKIP\n"
        "sub x13, x13, #64\n"
        "b END_WIDX_3TO8_STRIDE2_PAD_EXIT\n"
        "END_WIDX_3TO8_STRIDE2_PAD_NOT_SKIP:\n"
        "stp q21, q22, [x13, #-64]\n"
        "sub x13, x13, #32\n"
        "b END_WIDX_3TO8_STRIDE2_PAD_EXIT\n"
        "END_WIDX_3TO8_STRIDE2_PAD1_SKIP:\n"
        "sub x13, x13, #32\n"
        "END_WIDX_3TO8_STRIDE2_PAD_EXIT:\n"
        
        "add x10, x10, #1\n"
        "add x11, %[inW], %[inW], lsl #1\n" // x11 = inputWidth * 3
        "cmp x10, %[outH]\n"
        "add x12, x12, x11, lsl #2\n"
        "b.lt HIDX_LOOP_3TO8_STRIDE2\n" // HIDX_LOOP_3TO8_STRIDE2 EXIT
        "HIDX_LOOP_3TO8_STRIDE2_EXIT:"
        
        "add x9, x9, #1\n"
        "cmp x9, #3\n"
        "b.lt FIDX_LOOP_3TO8_STRIDE2\n" // FIDX_LOOP_3TO8_STRIDE2 EXIT
        "EXIT_3TO8_STRIDE2:\n"
    :  
    :   [inP] "r" (input), [filP] "r" (filter), [outP] "r" (output), [inW] "r" (inputWidth), [startH] "r" ((int64_t)inputHStart), [inH] "r" (inputHeight), [outW] "r" (outWidth),
        [outH] "r" (iterHeight), [mVal] "r" (moduloVal), [filS] "r" (filterBlockSize), [pad] "r" (padding)
    :   "x9", "x10", "x11", "x12", "x13",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "cc"
    );
    #ifdef __DEBUG_MTEN_OFF
        for (int i = 0; i < 128; i++)
        {
            printf("%6.3f\t", testArr[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
        for (int i = 0; i < 128; i++)
        {
            printf("%d\t", testArr2[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
    #endif
}

void* mTenConvThreadRoutine(void* threadArg)
{
    struct mTenThreadArg* threadData = (struct mTenThreadArg*) threadArg;
    int &id = threadData->id;
    struct mTenConvThreadData* dataPtr = (struct mTenConvThreadData*)(threadData->threadDataPtr);
    pthread_mutex_lock(&mTen::convThreadMutexes[id]);
    STARTCONVMTENTHREAD_3TO8: // Not sure if this is a good practice, but it  works.
    mTen* input = dataPtr->input;
    mTen* filter = dataPtr->filter;
    mTen* bias = dataPtr->bias;
    float* output = dataPtr->outputPtr;
    const int cacheRows = dataPtr->cacheRows;
    const int kernelRows = dataPtr->kernelRows;
    const int padding = dataPtr->padding;
    const int stride = dataPtr->stride;
    int* workLoadIndexPtr = dataPtr->workLoadIndexPtr;
    const int heightIn = input->height;
    const int widthIn = input->width;
    const int channels = input->channels;
    const int vecNum = input->vecNum;
    const int vecSize = input->vecSize;
    const int blocksFil = filter->blocks;
    const int heightFil = filter->height;
    const int widthFil = filter->width;
    const int heightOut = (heightIn - heightFil + padding*2)/stride + 1;
    const int widthOut = (widthIn - widthFil + padding*2)/stride + 1;
    const int vecSizeOut = 8;
    const int blocksNum = blocksFil/vecSizeOut;

    const int vecBlockFilSize = heightFil*widthFil*vecSize;
    const int vecWidthInputSize = widthIn*vecSize;
    const int hNum = input->getVectorNum(heightOut, cacheRows);

    // Kernel Selecter
    void (*kernelFnc)(float*, float*, float*, int, int, int, int, int, int, int);
    if (input->vecSize == 3)
    {
        switch (stride)
        {
        case 1:
            kernelFnc = aarch64_convKernel3x3_vec3_vec8_iterRow_ASM;
            break;
        case 2:
            kernelFnc = aarch64_convKernel3x3_vec3_vec8_iterRow_Stride2_ASM;
            break;
        default:
            printf("Conv 3 to 8 Error!! Unsupported stride value - %d\n", stride);
            kernelFnc = nullptr;
            break;
        }
    }
    else if (input->vecSize == 8)
    {
        switch (stride)
        {
        case 1:
            kernelFnc = aarch64_convKernel3x3_vec8_vec8_iterRow_ASM;
            break;
        // case 2:
        //     kernelFnc = aarch64_convKernel3x3_vec3_vec8_iterRow_Stride2_ASM;
        //     break;
        default:
            printf("Conv 8 to 8 Error!! Unsupported stride value - %d\n", stride);
            kernelFnc = nullptr;
            break;
        }
    }
    else
    {
        printf("Conv Error!! Unsupported vector size - %d\n", input->vecSize);
    }

    #ifdef __DEBUG_MTEN_OFF
        pthread_mutex_lock(&mTen::threadLockMutex);
    #endif
    while (true)
    {
        const int idx = __atomic_sub_fetch(workLoadIndexPtr, 1, __ATOMIC_RELAXED);
        if (idx < 0)
        {
            #ifdef __DEBUG_MTEN_OFF
                printf("Calculation for done. Thread exiting.\n\n");
                pthread_mutex_unlock(&mTen::threadLockMutex);
            #endif
            pthread_mutex_lock(&mTen::runningMutex);
            mTen::runningThreads--;
            if (mTen::runningThreads == 0)
            {
                pthread_cond_signal(&mTen::runningCondition); // Signals change of running condition to the main thread.
            }
            pthread_mutex_unlock(&mTen::runningMutex);
            pthread_cond_wait(&mTen::convThreadConditions[id], &mTen::convThreadMutexes[id]); // Thread waits here until the main thread signals convThreadConditions.
            goto STARTCONVMTENTHREAD_3TO8; // Child threads start with new thread arguments.
        }
        else
        {
            int hBatch = hNum - idx/blocksNum - 1;
            const int bIdx = (idx%blocksNum)*vecSizeOut;
            const int hIdxStart = hBatch * cacheRows;
            const int hIdxEnd =  (hIdxStart+cacheRows >= heightOut)? heightOut : (hIdxStart + cacheRows);
            const int inputHIdxStart = hIdxStart*stride - padding;
            #ifdef __DEBUG_MTEN_OFF
                printf("Idx %d, hBatch: %d Calculating for output height %d to %d, block %d to %d\n", idx, hBatch, hIdxStart, hIdxEnd-1, bIdx, bIdx+vecSizeOut);
            #endif
            float* tempOutPtr = output + bIdx*heightOut*widthOut + hIdxStart*widthOut*vecSizeOut;
            // Setting up pointers and temp. storage.
            float32x4_t vecBias0 = vld1q_f32(bias->tensorPtr + bIdx);
            float32x4_t vecBias1 = vld1q_f32(bias->tensorPtr + bIdx + 4);
            int initNum = (hIdxEnd - hIdxStart)*widthOut;      
            for (int i = 0; i < initNum; i++)
            {
                vst1q_f32(tempOutPtr, vecBias0);
                vst1q_f32(tempOutPtr + 4, vecBias1);
                tempOutPtr += 8;
            }
            #ifdef __DEBUG_MTEN_OFF
            tempOutPtr = output + bIdx*heightOut*widthOut + hIdxStart*widthOut*vecSizeOut;
            for (int bBDebug = 0; bBDebug < vecSizeOut; bBDebug++)
            {
                printf("Initialized Temp Output for bIdx %d\n", (bIdx + bBDebug));
                for (int hDebug = 0; hDebug < (hIdxEnd - hIdxStart); hDebug++)
                {
                    for (int wDebug = 0; wDebug < widthOut; wDebug++)
                    {
                        printf("%6.3f\t", *(tempOutPtr + (hDebug*widthOut+ wDebug)*vecSizeOut + bBDebug));
                    }
                    printf("- %d\n", (hDebug+1)*widthOut);
                }
                printf("\n");
            }
            #endif
            // Actual Calculation
            for (int vIdx = 0; vIdx < vecNum; vIdx++)
            {
                float* inputPtr = input->tensorPtr + inputHIdxStart*vecWidthInputSize + vIdx*heightIn*vecWidthInputSize;
                float* filterPtr = filter->tensorPtr + bIdx*vecNum*vecBlockFilSize + vIdx*vecBlockFilSize;
                tempOutPtr = output + bIdx*heightOut*widthOut + hIdxStart*widthOut*vecSizeOut;
                for (int hIdx = hIdxStart; hIdx < hIdxEnd;)
                {
                    const int kernelHeight = (hIdx + kernelRows > hIdxEnd)? hIdxEnd - hIdx : kernelRows;
                    (*kernelFnc)(inputPtr, filterPtr, tempOutPtr, widthIn, heightIn, hIdx*stride - padding, widthOut, kernelHeight, padding, vecBlockFilSize*vecNum);
                    #ifdef __DEBUG_MTEN_OFF
                        float* tempOutPtrDebug = output + bIdx*heightOut*widthOut + hIdxStart*widthOut*vecSizeOut;
                        for (int bBDebug = 0; bBDebug < vecSizeOut; bBDebug++)
                        {
                            printf("Temp Output from bIdx %d, vIdx %d, hIdx %d, kernel height %d\n", bIdx+bBDebug, vIdx, hIdx, kernelHeight);
                            for (int hDebug = 0; hDebug < (hIdxEnd - hIdxStart); hDebug++)
                            {
                                for (int wDebug = 0; wDebug < widthOut; wDebug++)
                                {
                                    printf("%6.3f\t", *(tempOutPtrDebug + (hDebug*widthOut+ wDebug)*vecSizeOut + bBDebug));
                                }
                                printf("- %d\n", (hDebug+1)*widthOut);
                            }
                            printf("\n");
                        }
                    #endif
                    tempOutPtr += widthOut*kernelHeight*vecSizeOut;
                    inputPtr += vecWidthInputSize*kernelHeight*stride;
                    hIdx += kernelHeight;
                }   
            }
        }
    }
}

void mTen::conv (mTen& input, mTen& filter, mTen& bias, int padding, int stride)
{
    if(!isConvThreadInitialized)
    {
        for (int i = 0; i < (mTen_num_threads); i++)
        {
            convThreads[i] = pthread_self();
            convThreadConditions[i] = PTHREAD_COND_INITIALIZER; 
            convThreadMutexes[i] = PTHREAD_MUTEX_INITIALIZER;
        }
        convThreadMutexes[mTen_num_threads] = PTHREAD_MUTEX_INITIALIZER;
        isConvThreadInitialized = true;
    }
    if (filter.vecSize != input.vecSize)
    {
        printf("conv - WARNING! Input & Filter vector size different! - Input Vector size: %d, Filter Vector size:%d\n", input.vecSize, filter.vecSize);
        return;
    }
    if (input.channels != filter.channels)
    {
        printf("conv - ERROR! Channel size different! - Input Channels: %d, Filter Channels: %d\n", input.channels, filter.channels);
        return;
    }
    if (filter.blocks != bias.channels)
    {
        printf("conv - ERROR! Output channel size different! - Filter blocks: %d, Bias Channles: %d\n", filter.blocks, bias.channels);
        return;
    }
    if (input.vecSize != 3 && input.vecSize != 8)
    {
        printf("conv - WARNING! Unsupported vector size! - Input & Filter Vector size: %d - Vectorizing to 8.\n", filter.vecSize);
        input.vectorize(8);
        filter.vectorize(8);
    }
    if (bias.vecSize != 0)
    {
        printf("conv - WARNING! Vectorized Bias! - Vector Size: %d - De-vectorizing.\n", bias.vecSize);
        bias.deVectorize();
    }
    else if (!(bias.isNHWC))
    {
        bias.NCHWtoNHWC();
    }
    #ifdef __DEBUG_MTEN_OFF
        printf("Input: ");
        input.printSize();
        printf("Filter: ");
        filter.printSize();
        printf("Bias: ");
        bias.printSize();
    #endif
    const int& vecSizeIn = input.vecSize;
    const int& vecNumIn = input.vecNum;
    const int& channelsIn = input.channels;
    const int& heightIn = input.height;
    const int& widthIn = input.width;
    const int& heightFil = filter.height;
    const int& widthFil = filter.width;

    const int& channelsOut = filter.blocks;
    const int heightOut = (heightIn - heightFil + padding*2)/stride + 1;
    const int widthOut = (widthIn - widthFil + padding*2)/stride + 1;
    const int vecSizeOut = 8;
    const int vecNumOut = getVectorNum(channelsOut, vecSizeOut);

    float* newTensorPtr = new float[vecNumOut*heightOut*widthOut*vecSizeOut];

    const int memPerRowStaticKernel = (heightFil*widthFil*vecSizeIn*vecSizeOut + widthIn*(heightFil-1)*vecSizeIn)*sizeof(float);
    const int memPerRowKernel = (widthIn*vecSizeIn + widthOut*vecSizeOut)*sizeof(float);
    int kernelRows = (__D1_CACHE_SIZE - 1024 - memPerRowStaticKernel) / memPerRowKernel;
    kernelRows = kernelRows > 0? kernelRows : 1;
    kernelRows = heightIn < kernelRows? heightIn : kernelRows;
    const int memPerRowStatic = (heightFil*widthFil*vecSizeIn*mTen_num_threads*2*vecSizeOut + widthOut*mTen_num_threads*2*(heightFil-1))*sizeof(float);
    const int memPerRow = (widthIn*vecSizeIn*vecNumIn + widthOut*mTen_num_threads*2)*sizeof(float);
    int cacheRows = (__LL_CACHE_SIZE - 100000 - memPerRowStatic) / memPerRow;
    cacheRows = cacheRows > 0? cacheRows : 1; 
    cacheRows = heightIn < cacheRows? heightIn : cacheRows;

    #ifdef __DEBUG_MTEN_OFF
        printf("conv - Pre Computation Memory Report.\n");
        printf("L1 data cache Size: %d, L1 mem usage static: %d, L1 mem usage per row: %d, L1 number of rows: %d, L1 mem remaining: %d\n",
            __D1_CACHE_SIZE, memPerRowStaticKernel, memPerRowKernel, kernelRows, __D1_CACHE_SIZE - memPerRowStaticKernel - kernelRows*memPerRowKernel);
            printf("LL data cache Size: %d, LL mem usage static: %d, LL mem usage per row: %d, LL number of rows: %d, LL mem remaining: %d\n\n",
            __LL_CACHE_SIZE, memPerRowStatic, memPerRow, cacheRows, __LL_CACHE_SIZE - memPerRowStatic - cacheRows*memPerRow);
    #endif

    int workLoadIndex;
    convThreadDataObj.input = &input;
    convThreadDataObj.filter = &filter;
    convThreadDataObj.bias = &bias;
    convThreadDataObj.outputPtr = newTensorPtr;
    convThreadDataObj.kernelRows = kernelRows;
    convThreadDataObj.padding = padding;
    convThreadDataObj.stride = stride;
    convThreadDataObj.workLoadIndexPtr = &workLoadIndex;
    workLoadIndex = vecNumOut * getVectorNum(heightOut, cacheRows);
    convThreadDataObj.cacheRows = cacheRows;
    //Custom threadpool implementation.
    pthread_mutex_lock(&mTen::runningMutex);
    for (int i = 0; i < (mTen_num_threads); i++)
    {       
        threadArgArr[i].threadDataPtr = (void*)&convThreadDataObj;
        if (pthread_equal(convThreads[i], pthread_self()))
        {
            threadArgArr[i].id = i;
            pthread_create (&convThreads[i], NULL, mTenConvThreadRoutine, (void* )&threadArgArr[i]);
        }
        else
        {
            pthread_mutex_unlock(&mTen::convThreadMutexes[i]);
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
        pthread_mutex_unlock(&mTen::runningMutex);
    }
    for (int i = 0; i < (mTen_num_threads); i++)
    {
        pthread_mutex_lock(&mTen::convThreadMutexes[i]);
    }
    pthread_mutex_unlock(&mTen::runningMutex);
    //Setting Output.
    if (tensorPtr != nullptr)
    {
        delete[] tensorPtr;
    }
    tensorPtr = newTensorPtr;
    blocks = 1;
    channels = channelsOut;
    height = heightOut;
    width = widthOut;
    vecSize = vecSizeOut;
    vecNum = vecNumOut;
    isNHWC = true;
}


void aarch64_convDepthKernel3x3_vec8_vec8_iterRow_ASM(float* input, float* filter, float* zeroPtr, float* output, int inputWidth, int inputHeight, int iterHeight, int padding)
{
    // 9 general purpose registers for input, 2 genneral purpose registers for hIdx, wIdx (x9, x10), 5 genneral purpose registers for calculation (x11, x12, x13, x14, x15).
    // Total of 16 general purpose registers.
    // 18 NEON vector registers for filter values (v0 ~ v17), 6 for input values (v18 ~ v23), 6 for output values (v24 ~ v29), 2 for bias values (v30, v31).
    // Total of 32 NEON vector registers.
    const int moduloVal = inputWidth - inputWidth%14;
    // float testArr[128] = {0};
    // int testArr2[128] = {0};
    __asm __volatile (
        "add x12, %[zeroP], %[inW], lsl #5\n"
        "ldp q0, q1, [%[filP]]\n"
        "ldp q2, q3, [%[filP], #32]\n"
        "ldp q4, q5, [%[filP], #64]\n"
        "ldp q6, q7, [%[filP], #96]\n"
        "ldp q8, q9, [%[filP], #128]\n"
        "ldp q10, q11, [%[filP], #160]\n"
        "ldp q12, q13, [%[filP], #192]\n"
        "ldp q14, q15, [%[filP], #224]\n"
        "ldp q16, q17, [%[filP], #256]\n"
        "ldp q30, q31, [x12]\n"
        "sub x9, xzr, %[pad]\n" // x9 = hIdx
        "mov x11, %[outP]\n" // x11 = outputPtr
        "cmp %[pad], #1\n"
        "b.eq INPUTSETUP_PAD1_DEPTH_8TO8\n"
        "cmp %[pad], #2\n"
        "b.eq INPUTSETUP_PAD2_DEPTH_8TO8\n"
        "mov x12, %[inP]\n" // x12 = basePtr0
        "add x13, x12, %[inW], lsl #5\n" // x13 = basePtr1
        "add x14, x13, %[inW], lsl #5\n" // x14 = basePtr2
        "b HIDX_LOOP_DEPTH_8TO8\n"
        "INPUTSETUP_PAD2_DEPTH_8TO8:\n"
        "mov x12, %[zeroP]\n" // x12 = basePtr0
        "mov x13, %[zeroP]\n" // x13 = basePtr1
        "mov x14, %[inP]\n" // x14 = basePtr2
        "b HIDX_LOOP_DEPTH_8TO8\n"
        "INPUTSETUP_PAD1_DEPTH_8TO8:\n"
        "mov x12, %[zeroP]\n" // x12 = basePtr0
        "mov x13, %[inP]\n" // x13 = basePtr1
        "add x14, %[inP], %[inW], lsl #5\n" // x14 = basePtr2
        "HIDX_LOOP_DEPTH_8TO8:\n"
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "prfm pldl1strm, [x14]\n"
        "mov x10, #0\n" // x10 = wIdx 
        "cmp %[inW], #14\n"
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "b.lt WIDX_LOOP_DEPTH_8TO8_EXIT\n"
        "cmp %[pad], #1\n"
        "WIDX_LOOP_DEPTH_8TO8:\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 0
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v0.4s, v18.4s\n"
        "fmla v29.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v6.4s, v20.4s\n"
        "fmla v29.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v12.4s, v22.4s\n"
        "fmla v29.4s, v13.4s, v23.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"
        "b.le SKIP_PAD2_DEPTH_8TO8\n"
        "stp q24, q25, [x11], #32\n" 
        "SKIP_PAD2_DEPTH_8TO8:\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 1
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "cmp x10, %[pad]\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "b.eq SKIP_PAD1_DEPTH_8TO8\n"
        "stp q26, q27, [x11], #32\n" 
        "SKIP_PAD1_DEPTH_8TO8:\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 2
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v4.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v19.4s\n"
        "fmla v26.4s, v0.4s, v18.4s\n"
        "fmla v27.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v10.4s, v20.4s\n"
        "fmla v29.4s, v11.4s, v21.4s\n"
        "fmla v26.4s, v6.4s, v20.4s\n"
        "fmla v27.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v23.4s\n"
        "fmla v26.4s, v12.4s, v22.4s\n"
        "fmla v27.4s, v13.4s, v23.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"
        "stp q28, q29, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 3
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v0.4s, v18.4s\n"
        "fmla v29.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v6.4s, v20.4s\n"
        "fmla v29.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v12.4s, v22.4s\n"
        "fmla v29.4s, v13.4s, v23.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 4
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "stp q26, q27, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 5
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v4.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v19.4s\n"
        "fmla v26.4s, v0.4s, v18.4s\n"
        "fmla v27.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v10.4s, v20.4s\n"
        "fmla v29.4s, v11.4s, v21.4s\n"
        "fmla v26.4s, v6.4s, v20.4s\n"
        "fmla v27.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v23.4s\n"
        "fmla v26.4s, v12.4s, v22.4s\n"
        "fmla v27.4s, v13.4s, v23.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"
        "stp q28, q29, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 6
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v0.4s, v18.4s\n"
        "fmla v29.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v6.4s, v20.4s\n"
        "fmla v29.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v12.4s, v22.4s\n"
        "fmla v29.4s, v13.4s, v23.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 7
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "stp q26, q27, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 8
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v4.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v19.4s\n"
        "fmla v26.4s, v0.4s, v18.4s\n"
        "fmla v27.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v10.4s, v20.4s\n"
        "fmla v29.4s, v11.4s, v21.4s\n"
        "fmla v26.4s, v6.4s, v20.4s\n"
        "fmla v27.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v23.4s\n"
        "fmla v26.4s, v12.4s, v22.4s\n"
        "fmla v27.4s, v13.4s, v23.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"
        "stp q28, q29, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 9
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v0.4s, v18.4s\n"
        "fmla v29.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v6.4s, v20.4s\n"
        "fmla v29.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v12.4s, v22.4s\n"
        "fmla v29.4s, v13.4s, v23.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 10
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "stp q26, q27, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 11
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v4.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v19.4s\n"
        "fmla v26.4s, v0.4s, v18.4s\n"
        "fmla v27.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v10.4s, v20.4s\n"
        "fmla v29.4s, v11.4s, v21.4s\n"
        "fmla v26.4s, v6.4s, v20.4s\n"
        "fmla v27.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v23.4s\n"
        "fmla v26.4s, v12.4s, v22.4s\n"
        "fmla v27.4s, v13.4s, v23.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"
        "stp q28, q29, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 12
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v0.4s, v18.4s\n"
        "fmla v29.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v6.4s, v20.4s\n"
        "fmla v29.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v12.4s, v22.4s\n"
        "fmla v29.4s, v13.4s, v23.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 13
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "add x10, x10, #14\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "stp q26, q27, [x11], #32\n" 
        "cmp %[mVal], x10\n"
        "mov v26.16b, v24.16b\n"
        "mov v27.16b, v25.16b\n"
        "mov v24.16b, v28.16b\n"
        "mov v25.16b, v29.16b\n"
        "b.gt WIDX_LOOP_DEPTH_8TO8\n" // WIDX_LOOP_DEPTH_8TO8 EXIT
        "WIDX_LOOP_DEPTH_8TO8_EXIT:\n"
        "sub x15, %[inW], x10\n"
        "cmp x15, #7\n"
        "b.lt WIDX_LOOP_DEPTH_8TO8_EXIT_7\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 0
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v0.4s, v18.4s\n"
        "fmla v29.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v6.4s, v20.4s\n"
        "fmla v29.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "add x15, x10, %[pad]\n"
        "fmla v28.4s, v12.4s, v22.4s\n"
        "fmla v29.4s, v13.4s, v23.4s\n"
        "cmp x15, #2\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"
        "b.lt SKIP_PAD2_DEPTH_8TO8_7_LEFTOVER\n"
        "stp q24, q25, [x11], #32\n" 
        "SKIP_PAD2_DEPTH_8TO8_7_LEFTOVER:\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 1
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "cmp x10, %[pad]\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "b.eq SKIP_PAD1_DEPTH_8TO8_7_LEFTOVER\n"
        "stp q26, q27, [x11], #32\n" 
        "SKIP_PAD1_DEPTH_8TO8_7_LEFTOVER:\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 2
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v4.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v19.4s\n"
        "fmla v26.4s, v0.4s, v18.4s\n"
        "fmla v27.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v10.4s, v20.4s\n"
        "fmla v29.4s, v11.4s, v21.4s\n"
        "fmla v26.4s, v6.4s, v20.4s\n"
        "fmla v27.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v23.4s\n"
        "fmla v26.4s, v12.4s, v22.4s\n"
        "fmla v27.4s, v13.4s, v23.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"
        "stp q28, q29, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 3
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v0.4s, v18.4s\n"
        "fmla v29.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v6.4s, v20.4s\n"
        "fmla v29.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v12.4s, v22.4s\n"
        "fmla v29.4s, v13.4s, v23.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 4
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "stp q26, q27, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 5
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v4.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v19.4s\n"
        "fmla v26.4s, v0.4s, v18.4s\n"
        "fmla v27.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v10.4s, v20.4s\n"
        "fmla v29.4s, v11.4s, v21.4s\n"
        "fmla v26.4s, v6.4s, v20.4s\n"
        "fmla v27.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v23.4s\n"
        "fmla v26.4s, v12.4s, v22.4s\n"
        "fmla v27.4s, v13.4s, v23.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"
        "stp q28, q29, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 6
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v0.4s, v18.4s\n"
        "fmla v29.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v6.4s, v20.4s\n"
        "fmla v29.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v12.4s, v22.4s\n"
        "fmla v29.4s, v13.4s, v23.4s\n"
        "add x10, x10, #7\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "mov v24.16b, v26.16b\n"
        "mov v25.16b, v27.16b\n"
        "mov v26.16b, v28.16b\n"
        "mov v27.16b, v29.16b\n"

        "WIDX_LOOP_DEPTH_8TO8_EXIT_7:\n"
        "cmp %[pad], #1\n"
        "b.eq PADDING_1_DEPTH_8TO8\n"
        "cmp %[pad], #0\n"
        "b.eq PADDING_EXIT_DEPTH_8TO8\n"
        "stp q24, q25, [x11], #32\n"
        "mov v24.16b, v26.16b\n"
        "mov v25.16b, v27.16b\n"
        "PADDING_1_DEPTH_8TO8:\n"
        "stp q24, q25, [x11], #32\n"
        "PADDING_EXIT_DEPTH_8TO8:\n"

        "add x9, x9, #1\n"
        "sub x12, x13, %[inW], lsl #5\n" // x12 = basePtr0
        "sub x13, x14, %[inW], lsl #5\n" // x13 = basePtr1

        "cmp x9, %[inH]\n"
        "b.lt PAD_HEIGHT_END_DEPTH_8TO8\n"
        "mov x14, %[zeroP]\n"
        "PAD_HEIGHT_END_DEPTH_8TO8:"
        "cmp x9, %[outH]\n"
        "b.lt HIDX_LOOP_DEPTH_8TO8\n" // HIDX_LOOP_DEPTH_8TO8 EXIT
        "EXIT_DEPTH_8TO8:\n"
    :  
    :   [inP] "r" (input), [filP] "r" (filter), [zeroP] "r" (zeroPtr), [outP] "r" (output), [inW] "r" (inputWidth), [inH] "r" (inputHeight - 2), [outH] "r" (iterHeight - padding), [mVal] "r" (moduloVal), [pad] "r" (padding)
    :   "x9", "x10", "x11", "x12", "x13", "x14", "x15",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc"
    );
    #ifdef __DEBUG_MTEN_OFF
        for (int i = 0; i < 128; i++)
        {
            printf("%6.3f\t", testArr[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
        for (int i = 0; i < 128; i++)
        {
            printf("%d\t", testArr2[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
    #endif
}

void aarch64_convDepthKernel3x3_vec8_vec8_Stride2_iterRow_ASM(float* input, float* filter, float* zeroPtr, float* output, int inputWidth, int inputHeight, int iterHeight, int padding)
{
    // 9 general purpose registers for input, 2 genneral purpose registers for hIdx, wIdx (x9, x10), 5 genneral purpose registers for calculation (x11, x12, x13, x14, x15).
    // Total of 16 general purpose registers.
    // 18 NEON vector registers for filter values (v0 ~ v17), 6 for input values (v18 ~ v23), 6 for output values (v24 ~ v29), 2 for bias values (v30, v31).
    // Total of 32 NEON vector registers.
    const int moduloVal = inputWidth - inputWidth%14;
    // float testArr[128] = {0};
    // int testArr2[128] = {0};
    __asm __volatile (
        "add x12, %[zeroP], %[inW], lsl #5\n"
        "ldp q0, q1, [%[filP]]\n"
        "ldp q2, q3, [%[filP], #32]\n"
        "ldp q4, q5, [%[filP], #64]\n"
        "ldp q6, q7, [%[filP], #96]\n"
        "ldp q8, q9, [%[filP], #128]\n"
        "ldp q10, q11, [%[filP], #160]\n"
        "ldp q12, q13, [%[filP], #192]\n"
        "ldp q14, q15, [%[filP], #224]\n"
        "ldp q16, q17, [%[filP], #256]\n"
        "ldp q30, q31, [x12]\n"
        "sub x9, xzr, %[pad]\n" // x9 = hIdx
        "mov x11, %[outP]\n" // x11 = outputPtr
        "cmp %[pad], #1\n"
        "b.eq INPUTSETUP_PAD1_DEPTH_8TO8_STRIDE2\n"
        "cmp %[pad], #2\n"
        "b.eq INPUTSETUP_PAD2_DEPTH_8TO8_STRIDE2\n"
        "mov x12, %[inP]\n" // x12 = basePtr0
        "add x13, x12, %[inW], lsl #5\n" // x13 = basePtr1
        "mov x14, x13\n"
        "b HIDX_LOOP_DEPTH_8TO8_STRIDE2\n"
        "INPUTSETUP_PAD2_DEPTH_8TO8_STRIDE2:\n"
        "mov x12, %[zeroP]\n" // x12 = basePtr0
        "mov x13, %[zeroP]\n" // x13 = basePtr1
        "sub x14, %[inP], %[inW], lsl #5\n" // x14 = basePtr2
        "b HIDX_LOOP_DEPTH_8TO8_STRIDE2\n"
        "INPUTSETUP_PAD1_DEPTH_8TO8_STRIDE2:\n"
        "mov x12, %[zeroP]\n" // x12 = basePtr0
        "mov x13, %[inP]\n" // x13 = basePtr1
        "mov x14, %[inP]\n"
        "HIDX_LOOP_DEPTH_8TO8_STRIDE2:\n"
        "add x14, x14, %[inW], lsl #5\n" // x14 = basePtr2
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "prfm pldl1strm, [x14]\n"
        "mov x10, #0\n" // x10 = wIdx 
        "cmp %[pad], #1\n"
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "b.eq WIDX_LOOP_DEPTH_8TO8_STRIDE2_PAD1\n"
        "WIDX_LOOP_DEPTH_8TO8_STRIDE2:\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 0
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v0.4s, v18.4s\n"
        "fmla v29.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v6.4s, v20.4s\n"
        "fmla v29.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v12.4s, v22.4s\n"
        "fmla v29.4s, v13.4s, v23.4s\n"
        "b.lt SKIP_PAD2_DEPTH_8TO8_STRIDE2\n"
        "stp q24, q25, [x11], #32\n" 
        "SKIP_PAD2_DEPTH_8TO8_STRIDE2:\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 1
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 2
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v4.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v10.4s, v20.4s\n"
        "fmla v29.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"
        "stp q28, q29, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 3
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 4
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 5
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 6
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v4.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v10.4s, v20.4s\n"
        "fmla v29.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"
        "stp q28, q29, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 7
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 8
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 9
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 10
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v4.4s, v18.4s\n"
        "fmla v29.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v10.4s, v20.4s\n"
        "fmla v29.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v16.4s, v22.4s\n"
        "fmla v29.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"
        "stp q28, q29, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 11
        "mov v28.16b, v30.16b\n"
        "mov v29.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 12
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 13
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v28.4s, v2.4s, v18.4s\n"
        "fmla v29.4s, v3.4s, v19.4s\n"
        "fmla v28.4s, v8.4s, v20.4s\n"
        "fmla v29.4s, v9.4s, v21.4s\n"
        "fmla v28.4s, v14.4s, v22.4s\n"
        "fmla v29.4s, v15.4s, v23.4s\n"

        "add x10, x10, #14\n"
        "cmp %[mVal], x10\n"
        "mov v24.16b, v28.16b\n"
        "mov v25.16b, v29.16b\n"

        "b.gt WIDX_LOOP_DEPTH_8TO8_STRIDE2\n" // WIDX_LOOP_DEPTH_8TO8_STRIDE2 EXIT
        "b WIDX_LOOP_DEPTH_8TO8_STRIDE2_EXIT\n"
        "WIDX_LOOP_DEPTH_8TO8_STRIDE2_PAD1:\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 0
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 1
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "stp q26, q27, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 2
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 3
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v26.4s, v0.4s, v18.4s\n"
        "fmla v27.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v26.4s, v6.4s, v20.4s\n"
        "fmla v27.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v26.4s, v12.4s, v22.4s\n"
        "fmla v27.4s, v13.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 4
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 5
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "stp q26, q27, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 6
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 7
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v26.4s, v0.4s, v18.4s\n"
        "fmla v27.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v26.4s, v6.4s, v20.4s\n"
        "fmla v27.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v26.4s, v12.4s, v22.4s\n"
        "fmla v27.4s, v13.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n"  

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 8
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 9
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "stp q26, q27, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 10
        "mov v26.16b, v30.16b\n"
        "mov v27.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v2.4s, v18.4s\n"
        "fmla v25.4s, v3.4s, v19.4s\n"
        "fmla v24.4s, v8.4s, v20.4s\n"
        "fmla v25.4s, v9.4s, v21.4s\n"
        "fmla v24.4s, v14.4s, v22.4s\n"
        "fmla v25.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 11
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v24.4s, v4.4s, v18.4s\n"
        "fmla v25.4s, v5.4s, v19.4s\n"
        "fmla v26.4s, v0.4s, v18.4s\n"
        "fmla v27.4s, v1.4s, v19.4s\n"
        "fmla v24.4s, v10.4s, v20.4s\n"
        "fmla v25.4s, v11.4s, v21.4s\n"
        "fmla v26.4s, v6.4s, v20.4s\n"
        "fmla v27.4s, v7.4s, v21.4s\n"
        "fmla v24.4s, v16.4s, v22.4s\n"
        "fmla v25.4s, v17.4s, v23.4s\n"
        "fmla v26.4s, v12.4s, v22.4s\n"
        "fmla v27.4s, v13.4s, v23.4s\n"
        "stp q24, q25, [x11], #32\n" 

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 12
        "mov v24.16b, v30.16b\n"
        "mov v25.16b, v31.16b\n"
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v2.4s, v18.4s\n"
        "fmla v27.4s, v3.4s, v19.4s\n"
        "fmla v26.4s, v8.4s, v20.4s\n"
        "fmla v27.4s, v9.4s, v21.4s\n"
        "fmla v26.4s, v14.4s, v22.4s\n"
        "fmla v27.4s, v15.4s, v23.4s\n"

        "ld1 {v18.4s,v19.4s}, [x12], #32\n" // Input 13
        "ld1 {v20.4s,v21.4s}, [x13], #32\n"
        "ld1 {v22.4s,v23.4s}, [x14], #32\n"
        "fmla v26.4s, v4.4s, v18.4s\n"
        "fmla v27.4s, v5.4s, v19.4s\n"
        "fmla v24.4s, v0.4s, v18.4s\n"
        "fmla v25.4s, v1.4s, v19.4s\n"
        "fmla v26.4s, v10.4s, v20.4s\n"
        "fmla v27.4s, v11.4s, v21.4s\n"
        "fmla v24.4s, v6.4s, v20.4s\n"
        "fmla v25.4s, v7.4s, v21.4s\n"
        "fmla v26.4s, v16.4s, v22.4s\n"
        "fmla v27.4s, v17.4s, v23.4s\n"
        "fmla v24.4s, v12.4s, v22.4s\n"
        "fmla v25.4s, v13.4s, v23.4s\n"
        "stp q26, q27, [x11], #32\n" 

        "mov v26.16b, v24.16b\n"
        "mov v27.16b, v25.16b\n"

        "add x10, x10, #14\n"
        "cmp %[mVal], x10\n"

        "b.gt WIDX_LOOP_DEPTH_8TO8_STRIDE2_PAD1\n" // WIDX_LOOP_DEPTH_8TO8_STRIDE2 EXIT
        "WIDX_LOOP_DEPTH_8TO8_STRIDE2_EXIT:\n"

        "cmp %[pad], #2\n"
        "b.ne PADDING_EXIT_DEPTH_8TO8_STRIDE2\n"
        "stp q24, q25, [x11], #32\n"
        "PADDING_EXIT_DEPTH_8TO8_STRIDE2:\n"

        "add x9, x9, #2\n"
        "mov x12, x13\n" // x12 = basePtr0
        "mov x13, x14\n" // x13 = basePtr1

        "cmp x9, %[inH]\n"
        "b.lt PAD_HEIGHT_END_DEPTH_8TO8_STRIDE2\n"
        "mov x14, %[zeroP]\n"
        "PAD_HEIGHT_END_DEPTH_8TO8_STRIDE2:"
        "cmp x9, %[outH]\n"
        "b.lt HIDX_LOOP_DEPTH_8TO8_STRIDE2\n" // HIDX_LOOP_DEPTH_8TO8_STRIDE2 EXIT
        "EXIT_DEPTH_8TO8_STRIDE2:\n"
    :  
    :   [inP] "r" (input), [filP] "r" (filter), [zeroP] "r" (zeroPtr), [outP] "r" (output), [inW] "r" (inputWidth), [inH] "r" (inputHeight - 2), [outH] "r" (iterHeight*2 - padding), [mVal] "r" (moduloVal), [pad] "r" (padding)
    :   "x9", "x10", "x11", "x12", "x13", "x14", "x15",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc"
    );
    #ifdef __DEBUG_MTEN_OFF
        for (int i = 0; i < 128; i++)
        {
            printf("%6.3f\t", testArr[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
        for (int i = 0; i < 128; i++)
        {
            printf("%d\t", testArr2[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
    #endif
}

void* mTenConvDepthThreadRoutine8to8(void* threadArg)
{
    struct mTenThreadArg* threadData = (struct mTenThreadArg*) threadArg;
    int &id = threadData->id;
    struct mTenConvThreadData* dataPtr = (struct mTenConvThreadData*)(threadData->threadDataPtr);
    pthread_mutex_lock(&mTen::convThreadMutexes[id]);
    STARTCONVMTENTHREAD_DEPTH_8TO8: // Not sure if this is a good practice, but it  works.
    mTen* input = dataPtr->input;
    mTen* filter = dataPtr->filter;
    mTen* bias = dataPtr->bias;
    float* output = dataPtr->outputPtr;
    const int stride = dataPtr->stride;
    const int padding = dataPtr->padding;
    int* workLoadIndexPtr = dataPtr->workLoadIndexPtr;
    const int heightIn = input->height;
    const int widthIn = input->width;
    const int vecSize = 8;
    const int heightFil = filter->height;
    const int widthFil = filter->width;
    const int heightOut = (heightIn - heightFil + padding*2)/stride + 1;
    const int widthOut = (widthIn - widthFil + padding*2)/stride + 1;
    const int vecSizeOut = 8;
    float* zeroPtr = new float [(widthIn+1)*vecSize];
    bzero (zeroPtr, widthIn*vecSize*sizeof(float));

    void (*kernelFnc)(float*, float*, float*, float*, int, int, int, int);
    switch (stride)
    {
    case 1:
        kernelFnc = aarch64_convDepthKernel3x3_vec8_vec8_iterRow_ASM;
        break;
    case 2:
        kernelFnc = aarch64_convDepthKernel3x3_vec8_vec8_Stride2_iterRow_ASM;
        break;
    default:
        printf("Depthwise Error!! Unsupported stride value - %d\n", stride);
        kernelFnc = nullptr;
        break;
    }

    #ifdef __DEBUG_MTEN_OFF
        pthread_mutex_lock(&mTen::threadLockMutex);
    #endif
    while (true)
    {
        const int idx = __atomic_sub_fetch(workLoadIndexPtr, 1, __ATOMIC_RELAXED);
        if (idx < 0)
        {
            delete[] zeroPtr;
            #ifdef __DEBUG_MTEN_OFF
                printf("Calculation for done. Thread exiting.\n\n");
                pthread_mutex_unlock(&mTen::threadLockMutex);
            #endif
            pthread_mutex_lock(&mTen::runningMutex);
            mTen::runningThreads--;
            if (mTen::runningThreads == 0)
            {
                pthread_cond_signal(&mTen::runningCondition); // Signals change of running condition to the main thread.
            }
            pthread_mutex_unlock(&mTen::runningMutex);
            pthread_cond_wait(&mTen::convThreadConditions[id], &mTen::convThreadMutexes[id]); // Thread waits here until the main thread signals convThreadConditions.
            goto STARTCONVMTENTHREAD_DEPTH_8TO8; // Child threads start with new thread arguments.
        }
        else
        {
            const int bIdx = idx*vecSizeOut;
            memcpy (zeroPtr + widthIn*vecSize, bias->tensorPtr + bIdx, vecSizeOut*sizeof(float));
            #ifdef __DEBUG_MTEN_OFF
                printf("Idx %d, Calculating for block %d to %d\n", idx, bIdx, bIdx+vecSizeOut-1);
            #endif
            // Actual Calculation
            float* inputPtr = input->tensorPtr + bIdx*(heightIn)*(widthIn);
            float* filterPtr = filter->tensorPtr + bIdx*heightFil*widthFil;
            float* tempOutPtr = output + bIdx*heightOut*widthOut;
            (*kernelFnc)(inputPtr, filterPtr, zeroPtr, tempOutPtr, widthIn, heightIn, heightOut, padding);
            #ifdef __DEBUG_MTEN_OFF
                float* tempOutPtrDebug = output + bIdx*heightOut*widthOut;
                for (int bBDebug = 0; bBDebug < vecSizeOut; bBDebug++)
                {
                    printf("Temp Output from bIdx %d\n", bIdx+bBDebug);
                    for (int hDebug = 0; hDebug < heightOut; hDebug++)
                    {
                        for (int wDebug = 0; wDebug < widthOut; wDebug++)
                        {
                            printf("%6.3f\t", *(tempOutPtrDebug + (hDebug*widthOut+ wDebug)*vecSizeOut + bBDebug));
                        }
                        printf("- %d\n", (hDebug+1)*widthOut);
                    }
                    printf("\n");
                }
            #endif
        }
    }
}

void mTen::convDepth8to8 (mTen& input, mTen& filter, mTen& bias, int padding, int stride)
{
    if(!isConvThreadInitialized)
    {
        for (int i = 0; i < (mTen_num_threads); i++)
        {
            convThreads[i] = pthread_self();
            convThreadConditions[i] = PTHREAD_COND_INITIALIZER; 
            convThreadMutexes[i] = PTHREAD_MUTEX_INITIALIZER;
        }
        convThreadMutexes[mTen_num_threads] = PTHREAD_MUTEX_INITIALIZER;
        isConvThreadInitialized = true;
    }
    if (input.vecSize != 8)
    {
        printf("Depthwise conv - WARNING! Wrong input vector size! - Input Vector size: %d - Vectorizing to 8.\n", input.vecSize);
        input.vectorize(8);
    }
    if (filter.vecSize != 8)
    {
        printf("Depthwise conv - WARNING! Wrong filter vector size! - Filter Vector size: %d - Vectorizing to 8.\n", filter.vecSize);
        filter.vectorize(8);
    }
    if (bias.vecSize != 0)
    {
        printf("Depthwise conv - WARNING! Vectorized Bias! - Vector Size: %d - De-vectorizing.\n", bias.vecSize);
        bias.deVectorize();
    }
    else if (!(bias.isNHWC))
    {
        bias.NCHWtoNHWC();
    }
    if (input.channels != filter.channels)
    {
        printf("Depthwise conv - ERROR! Channel size different! - Input Channels: %d, Filter Channels: %d\n", input.channels, filter.channels);
        return;
    }
    if (filter.channels != bias.channels)
    {
        printf("Depthwise conv - ERROR! Output channel size different! - Filter blocks: %d, Bias Channles: %d\n", filter.blocks, bias.channels);
        return;
    }
    if (stride != 1 && stride != 2)
    {
        printf("Depthwise conv - ERROR! Unsupported Stride - %d\n", stride);
        return;
    }
    #ifdef __DEBUG_MTEN_OFF
        printf("Input: ");
        input.printSize();
        printf("Filter: ");
        filter.printSize();
        printf("Bias: ");
        bias.printSize();
    #endif
    const int vecSizeIn = 8;
    const int& vecNumIn = input.vecNum;
    const int& channelsIn = input.channels;
    const int& heightIn = input.height;
    const int& widthIn = input.width;
    const int& heightFil = filter.height;
    const int& widthFil = filter.width;

    const int& channelsOut = input.channels;
    const int heightOut = (heightIn - heightFil + padding*2)/stride + 1;
    const int widthOut = (widthIn - widthFil + padding*2)/stride + 1;
    const int vecSizeOut = vecSizeIn;
    const int vecNumOut = vecNumIn;

    float* newTensorPtr = new float[vecNumOut*heightOut*widthOut*vecSizeOut];

    int workLoadIndex;
    convThreadDataObj.input = &input;
    convThreadDataObj.filter = &filter;
    convThreadDataObj.bias = &bias;
    convThreadDataObj.outputPtr = newTensorPtr;
    convThreadDataObj.padding = padding;
    convThreadDataObj.stride = stride;
    convThreadDataObj.workLoadIndexPtr = &workLoadIndex;
    workLoadIndex = vecNumOut;
    //Custom threadpool implementation.
    pthread_mutex_lock(&mTen::runningMutex);
    for (int i = 0; i < (mTen_num_threads); i++)
    {       
        threadArgArr[i].threadDataPtr = (void*)&convThreadDataObj;
        if (pthread_equal(convThreads[i], pthread_self()))
        {
            threadArgArr[i].id = i;
            pthread_create (&convThreads[i], NULL, mTenConvDepthThreadRoutine8to8, (void* )&threadArgArr[i]);
        }
        else
        {
            pthread_mutex_unlock(&mTen::convThreadMutexes[i]);
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
        pthread_mutex_unlock(&mTen::runningMutex);
    }
    for (int i = 0; i < (mTen_num_threads); i++)
    {
        pthread_mutex_lock(&mTen::convThreadMutexes[i]);
    }
    pthread_mutex_unlock(&mTen::runningMutex);
    // //Setting Output.
    if (tensorPtr != nullptr)
    {
        delete[] tensorPtr;
    }
    tensorPtr = newTensorPtr;
    blocks = 1;
    channels = channelsOut;
    height = heightOut;
    width = widthOut;
    vecSize = vecSizeOut;
    vecNum = vecNumOut;
    isNHWC = true;
}


void aarch64_convKernelPoint1x1_vec24_vec8_iterRow_ASM(float* input, float* filter, float* output, int inputWidth, int inputHeight, int iterHeight, int filterBlockSize)
{
    // 7 general purpose registers for input, 2 genneral purpose registers for bIdx, idx (x9, x10), 4 genneral purpose registers for calculation (x11, x12, x13, x14).
    // Total of 13 general purpose registers.
    // 24 NEON vector registers for filter values (v0 ~ v23), 4 for input values (v24 ~ v27), 4 for output values (v28 ~ v31).
    // Total of 32 NEON vector registers.
    const int moduloVal = (inputWidth*iterHeight) - (inputWidth*iterHeight)%14;
    // float testArr[128] = {0};
    // int testArr2[128] = {0};
    __asm __volatile (
        "mov x9, #0\n" // x9 = bIdx
        "mov x10, %[filS], lsl #2\n"
        "mov x11, %[filP]\n"
        "mov x12, #16\n"
        "sub x10, x10, #32\n"
        "BIDX_LOOP_POINTWISE%=_24TO8:\n"
        "ld4 {v0.s-v3.s}[0], [x11], #16\n"
        "ld4 {v4.s-v7.s}[0], [x11], x12\n"
        "ld4 {v8.s-v11.s}[0], [x11], x10\n"
        "ld4 {v0.s-v3.s}[1], [x11], #16\n"
        "ld4 {v4.s-v7.s}[1], [x11], x12\n"
        "ld4 {v8.s-v11.s}[1], [x11], x10\n"
        "ld4 {v0.s-v3.s}[2], [x11], #16\n"
        "ld4 {v4.s-v7.s}[2], [x11], x12\n"
        "ld4 {v8.s-v11.s}[2], [x11], x10\n"
        "ld4 {v0.s-v3.s}[3], [x11], #16\n"
        "ld4 {v4.s-v7.s}[3], [x11], x12\n"
        "ld4 {v8.s-v11.s}[3], [x11], x10\n"

        "ld4 {v12.s-v15.s}[0], [x11], #16\n"
        "ld4 {v16.s-v19.s}[0], [x11], x12\n"
        "ld4 {v20.s-v23.s}[0], [x11], x10\n"
        "ld4 {v12.s-v15.s}[1], [x11], #16\n"
        "ld4 {v16.s-v19.s}[1], [x11], x12\n"
        "ld4 {v20.s-v23.s}[1], [x11], x10\n"
        "ld4 {v12.s-v15.s}[2], [x11], #16\n"
        "ld4 {v16.s-v19.s}[2], [x11], x12\n"
        "ld4 {v20.s-v23.s}[2], [x11], x10\n"
        "ld4 {v12.s-v15.s}[3], [x11], #16\n"
        "ld4 {v16.s-v19.s}[3], [x11], x12\n"
        "ld4 {v20.s-v23.s}[3], [x11], x10\n"

        "mov x11, %[outP]\n" // x11 = outputPtr
        "mov x12, %[inP]\n" // x12 = basePtr
        "cmp x9, #1\n"
        "mov x10, #0\n" // x9 = idx
        "b.eq JVAL_BIDX_1_POINTWISE_24TO8\n"
        "mov x13, #-32\n"
        "add x13, x13, %[inN], lsl #5\n"
        "sub x14, xzr, x13\n"
        "b IDX_LOOP_POINTWISE%=_24TO8\n"
        "JVAL_BIDX_1_POINTWISE_24TO8:\n"
        "mov x13, #-16\n"
        "sub x13, x13, %[inN], lsl #5\n"
        "add x12, x12, %[inN], lsl #6\n" // x12 = basePtr
        "sub x14, xzr, x13\n"
        "IDX_LOOP_POINTWISE%=_24TO8:\n"
        //"prfm pldl1strm, [x11]\n"

        "ldp q24, q25, [x12], #32\n"
        "ldp q28, q29, [x11], #32\n" 
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "ldp q30, q31, [x11], #32\n" 

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "ldp q24, q25, [x12], #32\n"
        "stp q28, q29, [x11, #-64]\n" // Output 0
        "ldp q28, q29, [x11], #32\n"
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "stp q30, q31, [x11, #-64]\n" // Output 1
        "ldp q30, q31, [x11], #32\n"

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "ldp q24, q25, [x12], #32\n"
        "stp q28, q29, [x11, #-64]\n" // Output 2
        "ldp q28, q29, [x11], #32\n"
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "stp q30, q31, [x11, #-64]\n" // Output 3
        "ldp q30, q31, [x11], #32\n"

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "ldp q24, q25, [x12], #32\n"
        "stp q28, q29, [x11, #-64]\n" // Output 4
        "ldp q28, q29, [x11], #32\n"
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "stp q30, q31, [x11, #-64]\n" // Output 5
        "ldp q30, q31, [x11], #32\n"

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "ldp q24, q25, [x12], #32\n"
        "stp q28, q29, [x11, #-64]\n" // Output 6
        "ldp q28, q29, [x11], #32\n"
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "stp q30, q31, [x11, #-64]\n" // Output 7
        "ldp q30, q31, [x11], #32\n"

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "ldp q24, q25, [x12], #32\n"
        "stp q28, q29, [x11, #-64]\n" // Output 8
        "ldp q28, q29, [x11], #32\n"
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "stp q30, q31, [x11, #-64]\n" // Output 9
        "ldp q30, q31, [x11], #32\n"

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "ldp q24, q25, [x12], #32\n"
        "stp q28, q29, [x11, #-64]\n" // Output 10
        "ldp q28, q29, [x11], #32\n"
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "stp q30, q31, [x11, #-64]\n" // Output 11
        "ldp q30, q31, [x11], #32\n"

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "stp q28, q29, [x11, #-64]\n" // Output 12
        "stp q30, q31, [x11, #-32]\n" // Output 13

        "add x10, x10, #14\n"
        "cmp x10, %[mVal]\n"
        
        "b.lt IDX_LOOP_POINTWISE%=_24TO8\n" // IDX_LOOP_POINTWISE%=_24TO8 EXIT
        "cmp x10, %[inIdx]\n"
        "b.ge IDX_LOOP_POINTWISE%=_24TO8_EXIT\n"

        "ldp q24, q25, [x12], #32\n"
        "ldp q28, q29, [x11], #32\n" 
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "ldp q30, q31, [x11], #32\n" 

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "ldp q24, q25, [x12], #32\n"
        "stp q28, q29, [x11, #-64]\n" // Output 0
        "ldp q28, q29, [x11], #32\n"
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "stp q30, q31, [x11, #-64]\n" // Output 1
        "ldp q30, q31, [x11], #32\n"

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "ldp q24, q25, [x12], #32\n"
        "stp q28, q29, [x11, #-64]\n" // Output 2
        "ldp q28, q29, [x11], #32\n"
        "ld1 {v26.4s,v27.4s}, [x12], x13\n"
        "stp q30, q31, [x11, #-64]\n" // Output 3
        "ldp q30, q31, [x11], #32\n"

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v30.4s, v0.4s, v26.s[0]\n"
        "fmla v31.4s, v12.4s, v26.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v30.4s, v1.4s, v26.s[1]\n"
        "fmla v31.4s, v13.4s, v26.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v30.4s, v2.4s, v26.s[2]\n"
        "fmla v31.4s, v14.4s, v26.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"
        "fmla v30.4s, v3.4s, v26.s[3]\n"
        "fmla v31.4s, v15.4s, v26.s[3]\n"

        "ldr q24, [x12], #32\n"
        "ld1 {v26.4s}, [x12], x14\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v30.4s, v4.4s, v27.s[0]\n"
        "fmla v31.4s, v16.4s, v27.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v30.4s, v5.4s, v27.s[1]\n"
        "fmla v31.4s, v17.4s, v27.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v30.4s, v6.4s, v27.s[2]\n"
        "fmla v31.4s, v18.4s, v27.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"
        "fmla v30.4s, v7.4s, v27.s[3]\n"
        "fmla v31.4s, v19.4s, v27.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v30.4s, v8.4s, v26.s[0]\n"
        "fmla v31.4s, v20.4s, v26.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v30.4s, v9.4s, v26.s[1]\n"
        "fmla v31.4s, v21.4s, v26.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v30.4s, v10.4s, v26.s[2]\n"
        "fmla v31.4s, v22.4s, v26.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"
        "fmla v30.4s, v11.4s, v26.s[3]\n"
        "fmla v31.4s, v23.4s, v26.s[3]\n"

        "ldp q24, q25, [x12], #32\n"
        "stp q28, q29, [x11, #-64]\n" // Output 4
        "ldp q28, q29, [x11], #32\n"
        "add x12, x12, x13\n"
        "stp q30, q31, [x11, #-64]\n" // Output 5

        "fmla v28.4s, v0.4s, v24.s[0]\n"
        "fmla v29.4s, v12.4s, v24.s[0]\n"
        "fmla v28.4s, v1.4s, v24.s[1]\n"
        "fmla v29.4s, v13.4s, v24.s[1]\n"
        "fmla v28.4s, v2.4s, v24.s[2]\n"
        "fmla v29.4s, v14.4s, v24.s[2]\n"
        "fmla v28.4s, v3.4s, v24.s[3]\n"
        "fmla v29.4s, v15.4s, v24.s[3]\n"

        "ldr q24, [x12], #32\n"

        "fmla v28.4s, v4.4s, v25.s[0]\n"
        "fmla v29.4s, v16.4s, v25.s[0]\n"
        "fmla v28.4s, v5.4s, v25.s[1]\n"
        "fmla v29.4s, v17.4s, v25.s[1]\n"
        "fmla v28.4s, v6.4s, v25.s[2]\n"
        "fmla v29.4s, v18.4s, v25.s[2]\n"
        "fmla v28.4s, v7.4s, v25.s[3]\n"
        "fmla v29.4s, v19.4s, v25.s[3]\n"

        "fmla v28.4s, v8.4s, v24.s[0]\n"
        "fmla v29.4s, v20.4s, v24.s[0]\n"
        "fmla v28.4s, v9.4s, v24.s[1]\n"
        "fmla v29.4s, v21.4s, v24.s[1]\n"
        "fmla v28.4s, v10.4s, v24.s[2]\n"
        "fmla v29.4s, v22.4s, v24.s[2]\n"
        "fmla v28.4s, v11.4s, v24.s[3]\n"
        "fmla v29.4s, v23.4s, v24.s[3]\n"

        "stp q28, q29, [x11, #-32]\n" // Output 6

        "add x10, x10, #7\n"

        "IDX_LOOP_POINTWISE%=_24TO8_EXIT:\n"
        "add x9, x9, #1\n"
        "cmp x9, #2\n"
        "mov x10, %[filS], lsl #2\n"
        "add x11, %[filP], #64\n"
        "add x10, x10, #16\n"
        "mov x12, #-32\n"
        "b.lt BIDX_LOOP_POINTWISE%=_24TO8\n"
        "EXIT_POINTWISE%=_24TO8:\n"
    :  
    :   [inP] "r" (input), [filP] "r" (filter), [outP] "r" (output), [inN] "r" (inputWidth*inputHeight), [inIdx] "r" (inputWidth*iterHeight), [filS] "r" (filterBlockSize), [mVal] "r" (moduloVal)
    :   "x9", "x10", "x11", "x12", "x13", "x14",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc"
    );
    #ifdef __DEBUG_MTEN_OFF
        for (int i = 0; i < 128; i++)
        {
            printf("%6.3f\t", testArr[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
        for (int i = 0; i < 128; i++)
        {
            printf("%d\t", testArr2[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
    #endif
}

void aarch64_convKernelPoint1x1_vec8_vec8_iterRow_ASM(float* input, float* filter, float* output, int inputWidth, int inputHeight, int iterHeight, int filterBlockSize)
{
    // 6 general purpose registers for input, 1 genneral purpose registers for idx (x9), 2 genneral purpose registers for calculation (x10, x11).
    // Total of 9 general purpose registers.
    // 16 NEON vector registers for filter values (v0 ~ v15), 8 for input values (v16 ~ , v23), 8 for output values (v24 ~ v31).
    // Total of 32 NEON vector registers.
    const int moduloVal = (inputWidth*iterHeight) - (inputWidth*iterHeight)%28;
    // float testArr[128] = {0};
    // int testArr2[128] = {0};
    __asm __volatile (
        "prfm pldl1strm, [%[filP]]\n"
        "prfm pldl1strm, [%[inP]]\n"
        "prfm pldl1strm, [%[outP]]\n"
        "mov x10, %[filS], lsl #2\n"
        "mov x11, %[filP]\n"
        "sub x10, x10, #16\n"
        "ld4 {v0.s-v3.s}[0], [x11], #16\n"
        "ld4 {v4.s-v7.s}[0], [x11], x10\n"
        "ld4 {v0.s-v3.s}[1], [x11], #16\n"
        "ld4 {v4.s-v7.s}[1], [x11], x10\n"
        "ld4 {v0.s-v3.s}[2], [x11], #16\n"
        "ld4 {v4.s-v7.s}[2], [x11], x10\n"
        "ld4 {v0.s-v3.s}[3], [x11], #16\n"
        "ld4 {v4.s-v7.s}[3], [x11], x10\n"
        "ld4 {v8.s-v11.s}[0], [x11], #16\n"
        "ld4 {v12.s-v15.s}[0], [x11], x10\n"
        "ld4 {v8.s-v11.s}[1], [x11], #16\n"
        "ld4 {v12.s-v15.s}[1], [x11], x10\n"
        "ld4 {v8.s-v11.s}[2], [x11], #16\n"
        "ld4 {v12.s-v15.s}[2], [x11], x10\n"
        "ld4 {v8.s-v11.s}[3], [x11], #16\n"
        "ld4 {v12.s-v15.s}[3], [x11], x10\n"

        "mov x11, %[inP]\n" // x11 = basePtr
        "mov x10, %[outP]\n" // x10 = outputPtr
        "mov x9, #0\n" // x9 = idx

        "ldr q16, [x11], #32\n"
        "ldr q18, [x11], #32\n"
        "ldr q20, [x11], #32\n"
        "ldr q22, [x11], #-80\n"

        "ldr q24, [x10], #16\n"
        "ldr q25, [x10], #16\n"
        "ldr q26, [x10], #16\n"
        "ldr q27, [x10], #16\n"
        "ldr q28, [x10], #16\n"
        "ldr q29, [x10], #16\n"
        "ldr q30, [x10], #16\n"
        "ldr q31, [x10], #16\n"

        "IDX_LOOP_POINTWISE%=:\n"

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 0
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 0
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 1
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 1
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 2
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 2
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 3
        "ldr  q30, [x10], #16\n"
        "str q31, [x10, #-128]\n" // Output 3
        "ldr  q31, [x10], #16\n"
        
        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        
        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 4
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 4
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 5
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 5
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 6
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 6
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 7
        "ldr  q30, [x10], #16\n"
        "str q31, [x10, #-128]\n" // Output 7
        "ldr  q31, [x10], #16\n"

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        
        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 8
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 8
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 9
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 9
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 10
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 10
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 11
        "ldr  q30, [x10], #16\n"
        "str q31, [x10, #-128]\n" // Output 11
        "ldr  q31, [x10], #16\n"

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        
        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 12
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 12
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 13
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 13
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 14
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 14
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 15
        "ldr  q30, [x10], #16\n"
        "str q31, [x10, #-128]\n" // Output 15
        "ldr  q31, [x10], #16\n"

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        
        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 16
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 16
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 17
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 17
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 18
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 18
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 19
        "ldr  q30, [x10], #16\n"
        "str q31, [x10, #-128]\n" // Output 19
        "ldr  q31, [x10], #16\n"

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        
        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 20
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 20
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 21
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 21
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 22
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 22
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 23
        "ldr  q30, [x10], #16\n"
        "str q31, [x10, #-128]\n" // Output 23
        "ldr  q31, [x10], #16\n"

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        
        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 20
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 20
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 21
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 21
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 22
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 22
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 23
        "ldr  q30, [x10], #16\n"
        "str q31, [x10, #-128]\n" // Output 23
        "ldr  q31, [x10], #16\n"
        
        "add x9, x9, #28\n"
        "cmp x9, %[mVal]\n"
        
        "b.lt IDX_LOOP_POINTWISE%=\n" // WIDX_LOOP_POINTWISE%= EXIT
        
        "cmp x9, %[inIdx]\n"
        "b.ge EXIT_POINTWISE%=\n"

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 0
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 0
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 1
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 1
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 2
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 2
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 3
        "ldr  q30, [x10], #16\n"
        "str q31, [x10, #-128]\n" // Output 3
        "ldr  q31, [x10], #16\n"
        
        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        
        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 4
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 4
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 5
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 5
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 6
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 6
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 7
        "ldr  q30, [x10], #16\n"
        "str q31, [x10, #-128]\n" // Output 7
        "ldr  q31, [x10], #16\n"
        
        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"

        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "ldr  q16, [x11], #32\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"
        "ldr  q18, [x11], #-16\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"

        "str q24, [x10, #-128]\n" // Output 8
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 8
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 9
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 9
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 10
        "ldr  q17, [x11], #32\n"
        "str q29, [x10, #-112]\n" // Output 10
        "ldr  q19, [x11], #16\n"
        "str q30, [x10, #-96]\n" // Output 11
        "str q31, [x10, #-80]\n" // Output 11

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "ldr  q16, [x11], #32\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "ldr  q18, [x11], #32\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"

        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"

        "str q24, [x10, #-64]\n" // Output 12
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-64]\n" // Output 12
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-64]\n" // Output 13
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-64]\n" // Output 13
        "ldr  q27, [x10], #16\n"
        "ldr  q28, [x10], #16\n"
        "ldr  q29, [x10], #16\n"
        "ldr  q30, [x10], #16\n"
        "ldr  q31, [x10], #16\n"
        
        "add x9, x9, #14\n"
        "cmp x9, %[inIdx]\n"
        
        "b.ge EXIT_POINTWISE%=\n"

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n"
        "fmla v30.4s, v0.4s, v22.s[0]\n"
        "fmla v31.4s, v8.4s, v22.s[0]\n"
        

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "fmla v30.4s, v1.4s, v22.s[1]\n"
        "fmla v31.4s, v9.4s, v22.s[1]\n"
        "ldr  q21, [x11], #32\n"
        "ldr  q23, [x11], #16\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"
        "fmla v30.4s, v2.4s, v22.s[2]\n"
        "fmla v31.4s, v10.4s, v22.s[2]\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"
        "fmla v30.4s, v3.4s, v22.s[3]\n"
        "fmla v31.4s, v11.4s, v22.s[3]\n"
        

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "ldr  q16, [x11], #32\n"
        "ldr  q18, [x11], #32\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"
        "fmla v30.4s, v4.4s, v23.s[0]\n"
        "fmla v31.4s, v12.4s, v23.s[0]\n"
        "ldr  q20, [x11], #32\n"
        "ldr  q22, [x11], #-80\n"

        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"
        "fmla v30.4s, v5.4s, v23.s[1]\n"
        "fmla v31.4s, v13.4s, v23.s[1]\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"
        "fmla v30.4s, v6.4s, v23.s[2]\n"
        "fmla v31.4s, v14.4s, v23.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        "fmla v30.4s, v7.4s, v23.s[3]\n"
        "fmla v31.4s, v15.4s, v23.s[3]\n"
        
        "str q24, [x10, #-128]\n" // Output 0
        "ldr  q24 , [x10], #16\n"
        "str q25, [x10, #-128]\n" // Output 0
        "ldr  q25, [x10], #16\n"
        "str q26, [x10, #-128]\n" // Output 1
        "ldr  q26, [x10], #16\n"
        "str q27, [x10, #-128]\n" // Output 1
        "ldr  q27, [x10], #16\n"
        "str q28, [x10, #-128]\n" // Output 2
        "ldr  q28, [x10], #16\n"
        "str q29, [x10, #-128]\n" // Output 2
        "ldr  q29, [x10], #16\n"
        "str q30, [x10, #-128]\n" // Output 3
        "str q31, [x10, #-112]\n" // Output 3

        "fmla v24.4s, v0.4s, v16.s[0]\n"
        "fmla v25.4s, v8.4s, v16.s[0]\n"
        "fmla v26.4s, v0.4s, v18.s[0]\n"
        "fmla v27.4s, v8.4s, v18.s[0]\n"
        "fmla v28.4s, v0.4s, v20.s[0]\n"
        "fmla v29.4s, v8.4s, v20.s[0]\n" 

        "fmla v24.4s, v1.4s, v16.s[1]\n"
        "fmla v25.4s, v9.4s, v16.s[1]\n"
        "ldr  q17, [x11], #32\n"
        "ldr  q19, [x11], #32\n"
        "fmla v26.4s, v1.4s, v18.s[1]\n"
        "fmla v27.4s, v9.4s, v18.s[1]\n"
        "fmla v28.4s, v1.4s, v20.s[1]\n"
        "fmla v29.4s, v9.4s, v20.s[1]\n"
        "ldr  q21, [x11], #32\n"

        "fmla v24.4s, v2.4s, v16.s[2]\n"
        "fmla v25.4s, v10.4s, v16.s[2]\n"
        "fmla v26.4s, v2.4s, v18.s[2]\n"
        "fmla v27.4s, v10.4s, v18.s[2]\n"
        "fmla v28.4s, v2.4s, v20.s[2]\n"
        "fmla v29.4s, v10.4s, v20.s[2]\n"

        "fmla v24.4s, v3.4s, v16.s[3]\n"
        "fmla v25.4s, v11.4s, v16.s[3]\n"
        "fmla v26.4s, v3.4s, v18.s[3]\n"
        "fmla v27.4s, v11.4s, v18.s[3]\n"
        "fmla v28.4s, v3.4s, v20.s[3]\n"
        "fmla v29.4s, v11.4s, v20.s[3]\n"

        "fmla v24.4s, v4.4s, v17.s[0]\n"
        "fmla v25.4s, v12.4s, v17.s[0]\n"
        "fmla v26.4s, v4.4s, v19.s[0]\n"
        "fmla v27.4s, v12.4s, v19.s[0]\n"
        "fmla v28.4s, v4.4s, v21.s[0]\n"
        "fmla v29.4s, v12.4s, v21.s[0]\n"

        "fmla v24.4s, v5.4s, v17.s[1]\n"
        "fmla v25.4s, v13.4s, v17.s[1]\n"
        "fmla v26.4s, v5.4s, v19.s[1]\n"
        "fmla v27.4s, v13.4s, v19.s[1]\n"
        "fmla v28.4s, v5.4s, v21.s[1]\n"
        "fmla v29.4s, v13.4s, v21.s[1]\n"

        "fmla v24.4s, v6.4s, v17.s[2]\n"
        "fmla v25.4s, v14.4s, v17.s[2]\n"
        "fmla v26.4s, v6.4s, v19.s[2]\n"
        "fmla v27.4s, v14.4s, v19.s[2]\n"
        "fmla v28.4s, v6.4s, v21.s[2]\n"
        "fmla v29.4s, v14.4s, v21.s[2]\n"

        "fmla v24.4s, v7.4s, v17.s[3]\n"
        "fmla v25.4s, v15.4s, v17.s[3]\n"
        "fmla v26.4s, v7.4s, v19.s[3]\n"
        "fmla v27.4s, v15.4s, v19.s[3]\n"
        "fmla v28.4s, v7.4s, v21.s[3]\n"
        "fmla v29.4s, v15.4s, v21.s[3]\n"
        
        "str q24, [x10, #-96]\n" // Output 4
        "str q25, [x10, #-80]\n" // Output 4
        "str q26, [x10, #-64]\n" // Output 5
        "str q27, [x10, #-48]\n" // Output 5
        "str q28, [x10, #-32]\n" // Output 6
        "str q29, [x10, #-16]\n" // Output 6

        "EXIT_POINTWISE%=:\n"
    :  
    :   [inP] "r" (input), [filP] "r" (filter), [outP] "r" (output), [inIdx] "r" (inputWidth*iterHeight), [filS] "r" (filterBlockSize), [mVal] "r" (moduloVal)
    :   "x9", "x10", "x11", "x12",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc"
    );
    #ifdef __DEBUG_MTEN_OFF
        for (int i = 0; i < 128; i++)
        {
            printf("%6.3f\t", testArr[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
        for (int i = 0; i < 128; i++)
        {
            printf("%d\t", testArr2[i]);
            if (i%16 == 15)
            {
                printf("\n");
            }
        }
        printf("\n");
    #endif
}

void* mTenConvPointwiseThreadRoutine8to8(void* threadArg)
{
    struct mTenThreadArg* threadData = (struct mTenThreadArg*) threadArg;
    int &id = threadData->id;
    struct mTenConvThreadData* dataPtr = (struct mTenConvThreadData*)(threadData->threadDataPtr);
    pthread_mutex_lock(&mTen::convThreadMutexes[id]);
    STARTCONVPOINTWISEMTENTHREAD: // Not sure if this is a good practice, but it  works.
    mTen* input = dataPtr->input;
    mTen* filter = dataPtr->filter;
    mTen* bias = dataPtr->bias;
    float* output = dataPtr->outputPtr;
    const int cacheRows = dataPtr->cacheRows;
    const int kernelRows = dataPtr->kernelRows;
    int* workLoadIndexPtr = dataPtr->workLoadIndexPtr;
    const int heightIn = input->height;
    const int widthIn = input->width;
    const int channels = input->channels;
    const int vecNum = input->vecNum;
    const int vecSize = 8;
    const int blocksFil = filter->blocks;
    const int heightFil = 1;
    const int widthFil = 1;
    const int& heightOut = heightIn;
    const int& widthOut = widthIn;
    const int vecSizeOut = 8;
    const int blocksNum = blocksFil/vecSizeOut;

    const int vecWidthInputSize = widthIn*vecSize;
    const int hNum = input->getVectorNum(heightIn, cacheRows);

    #ifdef __DEBUG_MTEN_OFF
        pthread_mutex_lock(&mTen::threadLockMutex);
    #endif
    while (true)
    {
        const int idx = __atomic_sub_fetch(workLoadIndexPtr, 1, __ATOMIC_RELAXED);
        if (idx < 0)
        {
            #ifdef __DEBUG_MTEN_OFF
                printf("Calculation for done. Thread exiting.\n\n");
                pthread_mutex_unlock(&mTen::threadLockMutex);
            #endif
            pthread_mutex_lock(&mTen::runningMutex);
            mTen::runningThreads--;
            if (mTen::runningThreads == 0)
            {
                pthread_cond_signal(&mTen::runningCondition); // Signals change of running condition to the main thread.
            }
            pthread_mutex_unlock(&mTen::runningMutex);
            pthread_cond_wait(&mTen::convThreadConditions[id], &mTen::convThreadMutexes[id]); // Thread waits here until the main thread signals convThreadConditions.
            goto STARTCONVPOINTWISEMTENTHREAD; // Child threads start with new thread arguments.
        }
        else
        {
            int hBatch = hNum - idx/blocksNum - 1;
            const int bIdx = (idx%blocksNum)*vecSizeOut;
            const int hIdxStart = hBatch * cacheRows;
            const int hIdxEnd =  (hIdxStart+cacheRows >= heightIn)? heightIn : (hIdxStart + cacheRows);
            #ifdef __DEBUG_MTEN_OFF
                printf("Idx %d, hBatch: %d Calculating for height %d to %d, block %d to %d\n", idx, hBatch, hIdxStart, hIdxEnd-1, bIdx, bIdx+vecSizeOut);
            #endif
            float* tempOutPtr = output + bIdx*heightOut*widthOut + hIdxStart*widthOut*vecSizeOut;
            // Setting up pointers and temp. storage.
            float32x4_t vecBias0 = vld1q_f32(bias->tensorPtr + bIdx);
            float32x4_t vecBias1 = vld1q_f32(bias->tensorPtr + bIdx + 4);
            for (int i = 0; i < (hIdxEnd - hIdxStart)*widthOut; i++)
            {
                vst1q_f32(tempOutPtr, vecBias0);
                vst1q_f32(tempOutPtr + 4, vecBias1);
                tempOutPtr += 8;
            }
            #ifdef __DEBUG_MTEN_OFF
            tempOutPtr = output + bIdx*heightOut*widthOut + outHeightStartReal*widthOut*vecSizeOut;
            for (int bBDebug = 0; bBDebug < vecSizeOut; bBDebug++)
            {
                printf("Initialized Temp Output for bIdx %d\n", (bIdx + bBDebug));
                for (int hDebug = 0; hDebug < (outHeightEndReal - outHeightStartReal); hDebug++)
                {
                    for (int wDebug = 0; wDebug < widthOut; wDebug++)
                    {
                        printf("%6.3f\t", *(tempOutPtr + (hDebug*widthOut+ wDebug)*vecSizeOut + bBDebug));
                    }
                    printf("- %d\n", (hDebug+1)*widthOut);
                }
                printf("\n");
            }
            #endif
            // Actual Calculation
            tempOutPtr = output + bIdx*heightOut*widthOut + hIdxStart*widthOut*vecSizeOut;
            float* inputPtr = input->tensorPtr + hIdxStart*widthIn*vecSize;
            float* filterPtr = filter->tensorPtr + bIdx*vecNum*vecSize;
            int vIdx = 0;
            for (; vIdx < vecNum; vIdx++)
            { 
                aarch64_convKernelPoint1x1_vec8_vec8_iterRow_ASM(inputPtr, filterPtr, tempOutPtr, widthIn, heightIn, (hIdxEnd - hIdxStart), vecNum*vecSize);
                inputPtr += heightIn*widthIn*vecSize;
                filterPtr += vecSize;
                #ifdef __DEBUG_MTEN_OFF
                    float* tempOutPtrDebug = output + bIdx*heightOut*widthOut + hIdxStart*widthOut*vecSizeOut;
                    for (int bBDebug = 0; bBDebug < vecSizeOut; bBDebug++)
                    {
                        printf("Temp Output from bIdx %d, vIdx %d, hIdx %d, kernel height %d\n", bIdx+bBDebug, vIdx, hIdx, (hIdxEnd - hIdxStart));
                        for (int hDebug = 0; hDebug < (hIdxEnd - hIdxStart); hDebug++)
                        {
                            for (int wDebug = 0; wDebug < widthOut; wDebug++)
                            {
                                printf("%6.3f\t", *(tempOutPtrDebug + (hDebug*widthOut+ wDebug)*vecSizeOut + bBDebug));
                            }
                            printf("- %d\n", (hDebug+1)*widthOut);
                        }
                        printf("\n");
                    }
                #endif
            }
        }
    }
}

void mTen::convPointwise8to8 (mTen& input, mTen& filter, mTen& bias, int padding, int stride)
{
    if(!isConvThreadInitialized)
    {
        for (int i = 0; i < (mTen_num_threads); i++)
        {
            convThreads[i] = pthread_self();
            convThreadConditions[i] = PTHREAD_COND_INITIALIZER; 
            convThreadMutexes[i] = PTHREAD_MUTEX_INITIALIZER;
        }
        convThreadMutexes[mTen_num_threads] = PTHREAD_MUTEX_INITIALIZER;
        isConvThreadInitialized = true;
    }
    if (input.vecSize != 8)
    {
        printf("Pointwise conv - WARNING! Wrong input vector size! - Input Vector size: %d - Vectorizing to 8.\n", input.vecSize);
        input.vectorize(8);
    }
    if (filter.vecSize != 8)
    {
        printf("Pointwise conv - WARNING! Wrong filter vector size! - Filter Vector size: %d - Vectorizing to 8.\n", filter.vecSize);
        filter.vectorize(8);
    }
    if (bias.vecSize != 0)
    {
        printf("Pointwise conv - WARNING! Vectorized Bias! - Vector Size: %d - De-vectorizing.\n", bias.vecSize);
        bias.deVectorize();
    }
    else if (!(bias.isNHWC))
    {
        bias.NCHWtoNHWC();
    }
    if (input.channels != filter.channels)
    {
        printf("Pointwise conv - ERROR! Channel size different! - Input Channels: %d, Filter Channels: %d\n", input.channels, filter.channels);
        return;
    }
    if (filter.blocks != bias.channels)
    {
        printf("Pointwise conv - ERROR! Output channel size different! - Filter blocks: %d, Bias Channles: %d\n", filter.blocks, bias.channels);
        return;
    }
    if (padding != 0)
    {
        printf("Pointwise conv - ERROR! Padding not supported Pointwise Convolution.\n");
        return;
    }
    if (stride != 1)
    {
        printf("Pointwise Conv - ERROR! Unsupported stride value.\n");
        return;
    }
    #ifdef __DEBUG_MTEN_OFF
        printf("Input: ");
        input.printSize();
        printf("Filter: ");
        filter.printSize();
        printf("Bias: ");
        bias.printSize();
    #endif
    const int vecSizeIn = 8;
    const int& vecNumIn = input.vecNum;
    const int& channelsIn = input.channels;
    const int& heightIn = input.height;
    const int& widthIn = input.width;
    const int& heightFil = filter.height;
    const int& widthFil = filter.width;

    const int& channelsOut = filter.blocks;
    const int heightOut = heightIn - heightFil + padding*2 + 1;
    const int widthOut = widthIn - widthFil + padding*2 + 1;
    const int vecSizeOut = 8;
    const int vecNumOut = getVectorNum(channelsOut, vecSizeOut);

    float* newTensorPtr = new float[vecNumOut*heightOut*widthOut*vecSizeOut];

    const int memPerRowStaticKernel = (heightFil*widthFil*vecSizeIn*vecSizeOut + widthIn*(heightFil-1)*vecSizeIn)*sizeof(float);
    const int memPerRowKernel = (widthIn*vecSizeIn + widthOut*2*vecSizeOut)*sizeof(float);
    int kernelRows = (__D1_CACHE_SIZE - 1024 - memPerRowStaticKernel) / memPerRowKernel;
    kernelRows = kernelRows > 0? kernelRows : 1;
    kernelRows = heightIn < kernelRows? heightIn : kernelRows;
    const int memPerRowStatic = (heightFil*widthFil*vecSizeIn*mTen_num_threads*2*vecSizeOut + widthOut*mTen_num_threads*2*(heightFil-1))*sizeof(float);
    const int memPerRow = (widthIn*vecSizeIn*vecNumIn + widthOut*mTen_num_threads*2)*sizeof(float);
    int cacheRows = kernelRows;

    #ifdef __DEBUG_MTEN_OFF
        printf("conv - Pre Computation Memory Report.\n");
        printf("L1 data cache Size: %d, L1 mem usage static: %d, L1 mem usage per row: %d, L1 number of rows: %d, L1 mem remaining: %d\n",
            __D1_CACHE_SIZE, memPerRowStaticKernel, memPerRowKernel, kernelRows, __D1_CACHE_SIZE - memPerRowStaticKernel - kernelRows*memPerRowKernel);
            printf("LL data cache Size: %d, LL mem usage static: %d, LL mem usage per row: %d, LL number of rows: %d, LL mem remaining: %d\n\n",
            __LL_CACHE_SIZE, memPerRowStatic, memPerRow, cacheRows, __LL_CACHE_SIZE - memPerRowStatic - cacheRows*memPerRow);
    #endif

    int workLoadIndex;
    convThreadDataObj.input = &input;
    convThreadDataObj.filter = &filter;
    convThreadDataObj.bias = &bias;
    convThreadDataObj.outputPtr = newTensorPtr;
    convThreadDataObj.kernelRows = kernelRows;
    convThreadDataObj.padding = padding;
    convThreadDataObj.workLoadIndexPtr = &workLoadIndex;
    workLoadIndex = vecNumOut * getVectorNum(heightIn, cacheRows);
    convThreadDataObj.cacheRows = cacheRows;
    // Custom threadpool implementation.
    pthread_mutex_lock(&mTen::runningMutex);
    for (int i = 0; i < (mTen_num_threads); i++)
    {       
        threadArgArr[i].threadDataPtr = (void*)&convThreadDataObj;
        if (pthread_equal(convThreads[i], pthread_self()))
        {
            threadArgArr[i].id = i;
            pthread_create (&convThreads[i], NULL, mTenConvPointwiseThreadRoutine8to8, (void* )&threadArgArr[i]);
        }
        else
        {
            pthread_mutex_unlock(&mTen::convThreadMutexes[i]);
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
        pthread_mutex_unlock(&mTen::runningMutex);
    }
    for (int i = 0; i < (mTen_num_threads); i++)
    {
        pthread_mutex_lock(&mTen::convThreadMutexes[i]);
    }
    pthread_mutex_unlock(&mTen::runningMutex);
    // Setting Output.
    if (tensorPtr != nullptr)
    {
        delete[] tensorPtr;
    }
    tensorPtr = newTensorPtr;
    blocks = 1;
    channels = channelsOut;
    height = heightOut;
    width = widthOut;
    vecSize = vecSizeOut;
    vecNum = vecNumOut;
    isNHWC = true;
}


void mTen::loadData(std::string fileLocation)
{
    std::ifstream file(fileLocation, std::ios::in | std::ios::binary);
    char s[5];
    file.read(s, sizeof(char));
    if(s[0] != 'S')
    {
        std::cout << "//// mTen Tensor Loading Error!! - Wrong file format!! (S) ////" << std::endl;
        return;
    }
    file.read(s, sizeof(uint32_t));
    blocks = *((uint32_t*)&s);
    file.read(s, sizeof(uint32_t));
    channels = *((uint32_t*)&s);
    file.read(s, sizeof(uint32_t));
    height = *((uint32_t*)&s);
    file.read(s, sizeof(uint32_t));
    width = *((uint32_t*)&s);
    file.read(s, sizeof(uint32_t));
    isNHWC = !!(*((uint32_t*)&s));
    file.read(s, sizeof(uint32_t));
    vecSize = *((uint32_t*)&s);
    file.read(s, sizeof(uint32_t));
    vecNum = *((uint32_t*)&s);
    file.read(s, sizeof(char));
    if(s[0] != 'D')
    {
        std::cout << "//// mTen Tensor Loading Error!! - Wrong file format!! (D) ////";
        return;
    }
    else
    {
        std::cout << "Loading mTen Tensor with Blocks: " << blocks << ", Channels: " << channels << ", Height: " << height << ", Width: " << width <<
            ", In NHWC format?: " << isNHWC << ", Vector Size: " << vecSize << ", Vector Blocks: " << vecNum << std::endl;
    }
    if (tensorPtr != nullptr)
    {
        delete[] tensorPtr;
    }
    tensorPtr = new float[getMemSizeInBytes()/sizeof(float)];
    file.read((char*)tensorPtr, getMemSizeInBytes());
    file.read(s, sizeof(char));
    if(s[0] != 'E')
    {
        std::cout << std::endl << "//// mTen Tensor Loading Error!! - Wrong file format!! (E) ////" << std::endl;
        return;
    }
}

void mTen::saveData(std::string fileLocation)
{
    std::ofstream file(fileLocation, std::ios::out | std::ios::binary);
    uint32_t blocks32 = blocks;
    uint32_t channels32 = channels;
    uint32_t height32 = height;
    uint32_t width32 = width;
    uint32_t isNHWC32 = isNHWC;
    uint32_t vecSize32 = vecSize;
    uint32_t vecNum32 = vecNum;
    std::cout << "Saving mTen Tensor with Blocks: " << blocks32 << ", Channels: " << channels32 << ", Height: " << height32 << ", Width: " << width32 <<
        ", In NHWC format?: " << isNHWC32 << ", Vector Size: " << vecSize32 << ", Vector Blocks: " << vecNum32 << std::endl;
    const char s[4] = "SDE";
    file.write(s, sizeof(char));
    file.write((char*)&blocks32, sizeof(uint32_t));
    file.write((char*)&channels32, sizeof(uint32_t));
    file.write((char*)&height32, sizeof(uint32_t));
    file.write((char*)&width32, sizeof(uint32_t));
    file.write((char*)&isNHWC32, sizeof(uint32_t));
    file.write((char*)&vecSize32, sizeof(uint32_t));
    file.write((char*)&vecNum32, sizeof(uint32_t));
    file.write(s + 1, sizeof(char));
    file.write((char*)tensorPtr, getMemSizeInBytes());
    file.write(s + 2, sizeof(char));
}

void mTen::printSize()
{
    if (vecSize)
    {
        printf ("Vectorized tensor with Blocks %d, Channels %d, Height %d, Width %d, Vector Size %d, Vector blocks %d.\n", blocks, channels, height, width, vecSize, vecNum);
    }
    else
    {
        if (isNHWC)
        {
            printf ("NHWC tensor with Blocks %d, Channels %d, Height %d, Width %d.\n", blocks, channels, height, width);
        }
        else
        {
            printf ("NCHW tensor with Blocks %d, Channels %d, Height %d, Width %d.\n", blocks, channels, height, width);
        }
    }
}

void mTen::print(int newlineNum)
{
    printf("Printing ");
    printSize();
    if (vecSize)
    {
        
        for (int bIdx = 0; bIdx < blocks; bIdx++)
        {
            float* loadTarget = tensorPtr + bIdx*vecNum*height*width*vecSize;
            for (int vbIdx = 0; vbIdx < vecNum; vbIdx++)
            {
                printf ("//Block %d, Vector Block %d//\n", bIdx, vbIdx);
                for (int hIdx = 0; hIdx < height; hIdx++)
                {
                    for (int wIdx = 0; wIdx < width; wIdx++)
                    {
                        for (int vIdx = 0; vIdx < vecSize; vIdx++)
                        {
                            printf("%6.4f\t", *loadTarget);
                            loadTarget++;
                            if (vIdx%newlineNum == (newlineNum-1) && vIdx != vecSize-1)
                            {
                                printf("- (%d,%d)\n", hIdx, wIdx);
                            }
                        }
                        printf("- (%d,%d)\n", hIdx, wIdx);
                    }
                }
            }
        }
    }
    else
    {
        if (isNHWC)
        {
            for (int bIdx = 0; bIdx < blocks; bIdx++)
            {
                float* loadTarget = tensorPtr + bIdx*height*width*channels;
                printf ("//Block %d//\n", bIdx);
                for (int hIdx = 0; hIdx < height; hIdx++)
                {
                    for (int wIdx = 0; wIdx < width; wIdx++)
                    { 
                        for (int cIdx = 0; cIdx < channels; cIdx++)
                        {
                            printf("%6.4f\t", *loadTarget);
                            loadTarget++;
                            if (cIdx%newlineNum == (newlineNum-1) && cIdx != channels-1)
                            {
                                printf("- (%d,%d)\n", hIdx, wIdx);
                            }
                        }
                        printf("- (%d,%d)\n", hIdx, wIdx);
                    }
                }
            }
        }
        else
        {
            for (int bIdx = 0; bIdx < blocks; bIdx++)
            {
                for (int cIdx = 0; cIdx < channels; cIdx++)
                {
                    float* loadTarget = tensorPtr + bIdx*height*width*channels + cIdx*height*width;
                    printf ("//Block %d, Channel %d//\n", bIdx, cIdx);
                    for (int hIdx = 0; hIdx < height; hIdx++)
                    {
                        for (int wIdx = 0; wIdx < width; wIdx++)
                        {
                            printf("%6.4f\t", *loadTarget);
                            loadTarget++;
                            if (wIdx == width-1)
                            {
                                printf("\n");
                            }
                            else if (wIdx%newlineNum == (newlineNum-1))
                            {
                                printf("\n");
                            }
                        }
                        printf("\n");
                    }
                }
            }
        }
    }
}

void* mTenCompareThreadRoutine(void* threadArg)
{
    struct mTenCompareThreadData* dataPtr = (struct mTenCompareThreadData*) threadArg;
    mTen* inputmTen = dataPtr->inputMTEN;
    
    void* inputPtr = dataPtr->inputPtr;
    int mode = dataPtr->mode;
    int* cIdxPtr = dataPtr->indexPtr;
    int blocks = inputmTen->getBlock();
    int height = inputmTen->getHeight();
    int width = inputmTen->getWidth();
    int channels = inputmTen->getChannels();
    double epsilon = dataPtr->epsilon;
    bool out = true;
    int num = 0;
    double total = 0;
    int errNum = 0;
    double var = 0;
    
    while(1)
    {
        const int cIdx = __atomic_sub_fetch(cIdxPtr, 1, __ATOMIC_RELAXED);
        if (cIdx >= 0) // Checks if there are workloadleft.
        {
            #ifdef __INCLUDE_TORCH
            if (mode == 1)
            {
                torch::Tensor& tensor = *((torch::Tensor*)inputPtr);
                for (int bIdx = 0; bIdx < blocks; bIdx++)
                {
                    for (int hIdx = 0; hIdx < height; hIdx++)
                    {
                        for (int wIdx = 0; wIdx < width; wIdx++)
                        {
                            num++;
                            float mTenVal = (double)(inputmTen->getTensorVal(bIdx, cIdx, hIdx, wIdx));
                            if (tensor.sizes().size() == 2)
                            {
                                total += tensor[bIdx][cIdx].item<double>();
                                double error = tensor[bIdx][cIdx].item<double>() - (double)mTenVal;
                                var += error*error;
                                if(fabs(error) > epsilon)
                                {
                                    out = false;
                                    errNum++;
                                    pthread_mutex_lock(&mTen::printMutex);
                                    std::cout << "//////////// Warning!! - mTen to Float Comapare check failed!! Under epsilon " << std::fixed << std::setprecision(8) << epsilon << " - At block, channel, height, width index of " << bIdx << ", " << cIdx << ", " << hIdx << ", " << wIdx <<  " ////////////" << std::endl;
                                    std::cout << "\tFloat value: " << std::fixed << std::setprecision(5) << tensor[bIdx][cIdx].item<float>() << ", mTen value: " << mTenVal << ", Error: " << std::setprecision(8) << error << std::endl;
                                    pthread_mutex_unlock(&mTen::printMutex);
                                }
                            }
                            else if (tensor.sizes().size() == 3)
                            {
                                total += tensor[cIdx][hIdx][wIdx].item<double>();
                                double error = tensor[cIdx][hIdx][wIdx].item<double>() - (double)mTenVal;
                                var += error*error;
                                if(fabs(error) > epsilon)
                                {
                                    out = false;
                                    errNum++;
                                    pthread_mutex_lock(&mTen::printMutex);
                                    std::cout << "//////////// Warning!! - mTen to Float Comapare check failed!! Under epsilon " << std::fixed << std::setprecision(8) << epsilon << " - At block, channel, height, width index of " << bIdx << ", " << cIdx << ", " << hIdx << ", " << wIdx <<  " ////////////" << std::endl;
                                    std::cout << "\tFloat value: " << std::fixed << std::setprecision(5) << tensor[cIdx][hIdx][wIdx].item<float>() << ", mTen value: " << mTenVal << ", Error: " << std::setprecision(8) << error << std::endl;
                                    pthread_mutex_unlock(&mTen::printMutex);
                                }
                            }
                            else
                            {
                                total += tensor[bIdx][cIdx][hIdx][wIdx].item<double>();
                                double error = tensor[bIdx][cIdx][hIdx][wIdx].item<double>() - (double)mTenVal;
                                var += error*error;
                                if(fabs(error) > epsilon)
                                {
                                    out = false;
                                    errNum++;
                                    pthread_mutex_lock(&mTen::printMutex);
                                    std::cout << "//////////// Warning!! - mTen to Float Comapare check failed!! Under epsilon " << std::fixed << std::setprecision(8) << epsilon << " - At block, channel, height, width index of " << bIdx << ", " << cIdx << ", " << hIdx << ", " << wIdx <<  " ////////////" << std::endl;
                                    std::cout << "\tFloat value: " << std::fixed << std::setprecision(5) << tensor[bIdx][cIdx][hIdx][wIdx].item<float>() << ", mTen value: " << mTenVal << ", Error: " << std::setprecision(8) << error << std::endl;
                                    pthread_mutex_unlock(&mTen::printMutex);
                                }
                            }
                            
                        }
                    }
                }
            }
            else
            #endif
            {
                if (dataPtr->NHWCinput)
                {
                    for (int bIdx = 0; bIdx < blocks; bIdx++)
                    {
                        for (int hIdx = 0; hIdx < height; hIdx++)
                        {
                            for (int wIdx = 0; wIdx < width; wIdx++)
                            {
                                num++;
                                float mTenVal = inputmTen->getTensorVal(bIdx, cIdx, hIdx, wIdx);
                                float inputVal = *(((float*)inputPtr) + bIdx*height*width*channels + hIdx*width*channels + wIdx*channels + cIdx);
                                total += inputVal;
                                double error = (double)inputVal - (double)mTenVal;
                                var += error*error;
                                if(fabs(error) > epsilon)
                                {
                                    out = false;
                                    errNum++;
                                    pthread_mutex_lock(&mTen::printMutex);
                                    std::cout << "//////////// Warning!! - mTen to Float Comapare check failed!! Under epsilon " << std::fixed << std::setprecision(8) << epsilon << " - At block, channel, height, width index of " << bIdx << ", " << cIdx << ", " << hIdx << ", " << wIdx <<  " ////////////" << std::endl;
                                    std::cout << "\tFloat value: " << std::fixed << std::setprecision(5) << inputVal << ", mTen value: " << mTenVal << ", Error: " << std::setprecision(8) << error << std::endl;
                                    pthread_mutex_unlock(&mTen::printMutex);
                                }
                                
                            }
                        }
                    }
                }
                else
                {             
                    for (int bIdx = 0; bIdx < blocks; bIdx++)
                    {
                        for (int hIdx = 0; hIdx < height; hIdx++)
                        {
                            for (int wIdx = 0; wIdx < width; wIdx++)
                            {
                                num++;
                                float mTenVal = inputmTen->getTensorVal(bIdx, cIdx, hIdx, wIdx);
                                float inputVal = *(((float*)inputPtr) + bIdx*channels*height*width + cIdx*height*width + hIdx*width + wIdx);
                                total += inputVal;
                                double error = (double)inputVal - (double)mTenVal;
                                var += error*error;
                                if(fabs(error) > epsilon)
                                {
                                    out = false;
                                    errNum++;
                                    pthread_mutex_lock(&mTen::printMutex);
                                    std::cout << "//////////// Warning!! - mTen to Float Comapare check failed!! Under epsilon " << std::fixed << std::setprecision(8) << epsilon << " - At block, channel, height, width index of " << bIdx << ", " << cIdx << ", " << hIdx << ", " << wIdx <<  " ////////////" << std::endl;
                                    std::cout << "\tFloat value: " << std::fixed << std::setprecision(5) << inputVal << ", mTen value: " << mTenVal << ", Error: " << std::setprecision(8) << error << std::endl;
                                    pthread_mutex_unlock(&mTen::printMutex);
                                }
                                
                            }
                        }
                    }
                }
            }
        }
        else
        {
            *(dataPtr->outPtr) &= out; 
            *(dataPtr->numPtr) += num; 
            *(dataPtr->totalPtr) += total; 
            *(dataPtr->errNumPtr) += errNum; 
            *(dataPtr->varPtr) += var; 
            return nullptr;
        }
    }
}

bool mTen::compareToFloatArr(float* input, float epsilon, bool NHWCin)
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

    pthread_t threads[mTen_num_threads];
    struct mTenCompareThreadData  threadDataArr;
    int indexCounter = channels;

    threadDataArr.epsilon = epsilon;
    threadDataArr.inputMTEN = this;
    threadDataArr.inputPtr = (void*)input;
    threadDataArr.outPtr = &out;
    threadDataArr.indexPtr = &indexCounter;
    threadDataArr.numPtr = &num;
    threadDataArr.NHWCinput = NHWCin;
    threadDataArr.totalPtr = &total;
    threadDataArr.errNumPtr = &errNum;
    threadDataArr.varPtr = &var;
    threadDataArr.mode = 0;

    // Starting threads.
    for (int i = 0; i < mTen_num_threads; i++)
    {
        pthread_create (&threads[i], NULL, mTenCompareThreadRoutine, (void* )&threadDataArr);
    }

    for (int i = 0; i < mTen_num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    if(out)
    {
        this->printSize();
        std::cout << "mTen to Float Arr Compare passed under epsilon " << std::fixed << std::setprecision(6) << epsilon << std::endl;
        std::cout << "Total Number of elements: " << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(8) << sqrt(var/num) << std::endl;
    }
    else
    {
        this->printSize();
        std::cout << "//////////// Warning!! - mTen compare to Float Arr failed. ////////////" << std::endl;
        std::cout << "Total Number of elements: " << std::fixed << std::setprecision(6) << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(8) << sqrt(var/num) << std::endl;
    }
    std::cout << std::endl;
    return out;
}

#ifdef __INCLUDE_TORCH
mTen::mTen(torch::Tensor& tensor, bool NHWC, int vecSizeIn)
{
    if (tensor.sizes().size() == 2)
    {
        blocks = tensor.sizes()[0];
        channels = tensor.sizes()[1];
        height = 1;
        width = 1;
    }
    else if (tensor.sizes().size() == 3)
    {
        blocks = 1;
        channels = tensor.sizes()[0];
        height = tensor.sizes()[1];
        width = tensor.sizes()[2];
    }
    else if (tensor.sizes().size() == 1)
    {
        blocks = 1;
        channels = tensor.sizes()[0];
        height = 1;
        width = 1;
    }
    else
    {
        blocks = tensor.sizes()[0];
        channels = tensor.sizes()[1];
        height = tensor.sizes()[2];
        width = tensor.sizes()[3];
    }
    isNHWC = NHWC;
    vecSize = vecSizeIn;
    if (vecSize > 0)
    {
        vecNum = getVectorNum(channels, vecSizeIn);
    }
    else
    {
        vecNum = 0;
    }
    tensorPtr = new float[getMemSizeInBytes()/sizeof(float)];

    for (int bIdx = 0; bIdx < blocks; bIdx++)
    {
        if (vecSize > 0)
        {
            bzero(tensorPtr + ((bIdx+1)*vecNum - 1)*height*width*vecSize, height*width*vecSize*sizeof(float));
        }
        for (int cIdx = 0; cIdx < channels; cIdx++)
        {
            for (int hIdx = 0; hIdx < height; hIdx++)
            {
                for (int wIdx = 0; wIdx < width; wIdx++)
                {
                    if (tensor.sizes().size() == 2)
                    {
                        setTensorVal(tensor[bIdx][cIdx].item<float>(), bIdx, cIdx, hIdx, wIdx);
                    }
                    else if (tensor.sizes().size() == 3)
                    {
                        setTensorVal(tensor[cIdx][hIdx][wIdx].item<float>(), bIdx, cIdx, hIdx, wIdx);
                    }
                    else if (tensor.sizes().size() == 1)
                    {
                        setTensorVal(tensor[cIdx].item<float>(), bIdx, cIdx, hIdx, wIdx);
                    }
                    else
                    {
                        setTensorVal(tensor[bIdx][cIdx][hIdx][wIdx].item<float>(), bIdx, cIdx, hIdx, wIdx);
                    }
                }
            }
        }
    }
}

bool mTen::compareToTorch(torch::Tensor& input, float epsilon)
{
    bool out = true;
    int tblocks, tchannels, theight, twidth;
    if (input.sizes().size() == 2)
    {
        tblocks = input.sizes()[0];
        tchannels = input.sizes()[1];
        theight = 1;
        twidth = 1;
    }
    else if (input.sizes().size() == 3)
    {
        tblocks = 1;
        tchannels = input.sizes()[0];
        theight = input.sizes()[1];
        twidth = input.sizes()[2];
    }
    else
    {
        tblocks = input.sizes()[0];
        tchannels = input.sizes()[1];
        theight = input.sizes()[2];
        twidth = input.sizes()[3];
    }
    
    if(blocks != tblocks)
    {
        std::cout << "//////////// Warning!! - mTen to Torch Comapare check failed!! - Number of blocks diffrent!!! ////////////" << std::endl;
        std::cout << "\tTensor value: " << tblocks << ", mTen value: " << blocks << std::endl;
        out = false;
    }
    if(channels != tchannels)
    {
        std::cout << "//////////// Warning!! - mTen to Torch Comapare check failed!! - Number of channels diffrent!!! ////////////" << std::endl;
        std::cout << "\tTensor value: " << tchannels << ", mTen value: " << channels << std::endl;
        out = false;
    }
    if(height != theight)
    {
        std::cout << "//////////// Warning!! - mTen to Torch Comapare check failed!! - Diffrent height!!! ////////////" << std::endl;
        std::cout << "\tTensor value: " << theight << ", mTen value: " << height << std::endl;
        out = false;
    }
    if(width != twidth)
    {
        std::cout << "//////////// Warning!! - mTen to Torch Comapare check failed!! - Diffrent width!!! ////////////" << std::endl;
        std::cout << "\tTensor value: " << twidth << ", mTen value: " << width << std::endl;
        out = false;
    }
    if(!out)
    {
        return out;
    }
    int num = 0;
    double total = 0;
    int errNum = 0;
    double var = 0;

    pthread_t threads[mTen_num_threads];
    struct mTenCompareThreadData  threadDataArr;
    int indexCounter = channels;

    threadDataArr.epsilon = epsilon;
    threadDataArr.inputMTEN = this;
    threadDataArr.inputPtr = (void*)(&input);
    threadDataArr.outPtr = &out;
    threadDataArr.indexPtr = &indexCounter;
    threadDataArr.numPtr = &num;
    threadDataArr.totalPtr = &total;
    threadDataArr.errNumPtr = &errNum;
    threadDataArr.varPtr = &var;
    threadDataArr.mode = 1;

    // Starting threads.
    for (int i = 0; i < mTen_num_threads; i++)
    {
        pthread_create (&threads[i], NULL, mTenCompareThreadRoutine, (void* )&threadDataArr);
    }

    for (int i = 0; i < mTen_num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    if(out)
    {
        this->printSize();
        std::cout << "mTen to Torch Compare passed under epsilon " << std::fixed << std::setprecision(6) << epsilon << std::endl;
        std::cout << "Total Number of elements: " << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(8) << sqrt(var/num) << std::endl;
    }
    else
    {
        this->printSize();
        std::cout << "//////////// Warning!! - mTen compare to Tensor failed. ////////////" << std::endl;
        std::cout << "Total Number of elements: " << std::fixed << std::setprecision(6) << num << " mean value: " << total/num << " Total Number of errors: " << errNum << ", Std. deviation: " << std::fixed << std::setprecision(8) << sqrt(var/num) << std::endl;
    }
    std::cout << std::endl;
    return out;
}
#endif
