void kernel_1_9_8_1_1_0_0(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Duplicate index: 0
// [1, 1, 1, 1, 1, 1, 1, 1, 1]
// Number of Input Regs: 9, Filter Regs: 4 Output Regs: 18
// Total number of registers required: 31
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Out - [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
// Fil - [27, 28, 29, 30, 27, 28]
// Register mapping diagram
//     0  1  2  3  4  5  6  7  8 
//
//27   9 11 13 15 17 19 21 23 25 
//28  10 12 14 16 18 20 22 24 26 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < (k/8); i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*2 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*3 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*4 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*5 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*6 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*7 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*8 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 1; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*1*8 + wf*8 + i));
            }
            printf("\n");
        }
    }
    printf ("Output:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf("%6.3f\t", *(outputPtr + 8*5 + i));
        printf("%6.3f\t", *(outputPtr + 8*6 + i));
        printf("%6.3f\t", *(outputPtr + 8*7 + i));
        printf("%6.3f\t", *(outputPtr + 8*8 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif

    float* in = inputPtr + 0*8;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
    "add x8, %[in], %[inStr]\n"
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "prfm pldl1keep, [%[in], #64]\n"
    "prfm pldl1keep, [%[in], #128]\n"
    "prfm pldl1keep, [%[in], #192]\n"
    "prfm pldl1keep, [%[in], #256]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "prfm pldl1keep, [x8, #192]\n"
    "prfm pldl1keep, [x8, #256]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q9, q10, [%[out], #0]\n"
    "ldp q11, q12, [%[out], #32]\n"
    "ldp q13, q14, [%[out], #64]\n"
    "ldp q15, q16, [%[out], #96]\n"
    "ldp q17, q18, [%[out], #128]\n"
    "ldp q19, q20, [%[out], #160]\n"
    "ldp q21, q22, [%[out], #192]\n"
    "ldp q23, q24, [%[out], #224]\n"
    "ldp q25, q26, [%[out], #256]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #32]\n"
    "ldr q2, [x10, #64]\n"
    "ldr q3, [x10, #96]\n"
    "ldr q4, [x10, #128]\n"
    "ldr q5, [x10, #160]\n"
    "ldr q6, [x10, #192]\n"
    "ldr q7, [x10, #224]\n"
    "ldr q8, [x10, #256]\n"

    "sub w9, %w[k], #8\n"
    "LOOP_START%=:\n"
    "subs x9, x9, #8\n"
// K index 0
// Filter width idx 0
    "fmla v9.4s, v27.4s, v0.s[0]\n"
    "fmla v10.4s, v28.4s, v0.s[0]\n"
    "fmla v11.4s, v27.4s, v1.s[0]\n"
    "fmla v12.4s, v28.4s, v1.s[0]\n"
    "fmla v13.4s, v27.4s, v2.s[0]\n"
    "fmla v14.4s, v28.4s, v2.s[0]\n"
    "fmla v15.4s, v27.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v3.s[0]\n"
    "fmla v17.4s, v27.4s, v4.s[0]\n"
    "fmla v18.4s, v28.4s, v4.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v19.4s, v27.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v6.s[0]\n"
    "fmla v22.4s, v28.4s, v6.s[0]\n"
    "fmla v23.4s, v27.4s, v7.s[0]\n"
    "fmla v24.4s, v28.4s, v7.s[0]\n"
    "fmla v25.4s, v27.4s, v8.s[0]\n"
    "fmla v26.4s, v28.4s, v8.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v9.4s, v29.4s, v0.s[1]\n"
    "fmla v10.4s, v30.4s, v0.s[1]\n"
    "fmla v11.4s, v29.4s, v1.s[1]\n"
    "fmla v12.4s, v30.4s, v1.s[1]\n"
    "fmla v13.4s, v29.4s, v2.s[1]\n"
    "fmla v14.4s, v30.4s, v2.s[1]\n"
    "fmla v15.4s, v29.4s, v3.s[1]\n"
    "fmla v16.4s, v30.4s, v3.s[1]\n"
    "fmla v17.4s, v29.4s, v4.s[1]\n"
    "fmla v18.4s, v30.4s, v4.s[1]\n"
    "fmla v19.4s, v29.4s, v5.s[1]\n"
    "fmla v20.4s, v30.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v6.s[1]\n"
    "fmla v22.4s, v30.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v7.s[1]\n"
    "fmla v24.4s, v30.4s, v7.s[1]\n"
    "fmla v25.4s, v29.4s, v8.s[1]\n"
    "fmla v26.4s, v30.4s, v8.s[1]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v9.4s, v27.4s, v0.s[2]\n"
    "fmla v10.4s, v28.4s, v0.s[2]\n"
    "fmla v11.4s, v27.4s, v1.s[2]\n"
    "fmla v12.4s, v28.4s, v1.s[2]\n"
    "fmla v13.4s, v27.4s, v2.s[2]\n"
    "fmla v14.4s, v28.4s, v2.s[2]\n"
    "fmla v15.4s, v27.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v4.s[2]\n"
    "fmla v18.4s, v28.4s, v4.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v19.4s, v27.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v6.s[2]\n"
    "fmla v22.4s, v28.4s, v6.s[2]\n"
    "fmla v23.4s, v27.4s, v7.s[2]\n"
    "fmla v24.4s, v28.4s, v7.s[2]\n"
    "fmla v25.4s, v27.4s, v8.s[2]\n"
    "fmla v26.4s, v28.4s, v8.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v9.4s, v29.4s, v0.s[3]\n"
    "fmla v10.4s, v30.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v11.4s, v29.4s, v1.s[3]\n"
    "fmla v12.4s, v30.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v13.4s, v29.4s, v2.s[3]\n"
    "fmla v14.4s, v30.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v15.4s, v29.4s, v3.s[3]\n"
    "fmla v16.4s, v30.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v17.4s, v29.4s, v4.s[3]\n"
    "fmla v18.4s, v30.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "fmla v19.4s, v29.4s, v5.s[3]\n"
    "fmla v20.4s, v30.4s, v5.s[3]\n"
    "ldr q5, [x10, #176]\n"
    "fmla v21.4s, v29.4s, v6.s[3]\n"
    "fmla v22.4s, v30.4s, v6.s[3]\n"
    "ldr q6, [x10, #208]\n"
    "fmla v23.4s, v29.4s, v7.s[3]\n"
    "fmla v24.4s, v30.4s, v7.s[3]\n"
    "ldr q7, [x10, #240]\n"
    "fmla v25.4s, v29.4s, v8.s[3]\n"
    "fmla v26.4s, v30.4s, v8.s[3]\n"
    "ldr q8, [x10, #272]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v9.4s, v27.4s, v0.s[0]\n"
    "fmla v10.4s, v28.4s, v0.s[0]\n"
    "fmla v11.4s, v27.4s, v1.s[0]\n"
    "fmla v12.4s, v28.4s, v1.s[0]\n"
    "fmla v13.4s, v27.4s, v2.s[0]\n"
    "fmla v14.4s, v28.4s, v2.s[0]\n"
    "fmla v15.4s, v27.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v3.s[0]\n"
    "add x10, x10, %[inStr]\n"
    "add x8, x8, %[inStr]\n"
    "fmla v17.4s, v27.4s, v4.s[0]\n"
    "fmla v18.4s, v28.4s, v4.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v19.4s, v27.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v6.s[0]\n"
    "fmla v22.4s, v28.4s, v6.s[0]\n"
    "fmla v23.4s, v27.4s, v7.s[0]\n"
    "fmla v24.4s, v28.4s, v7.s[0]\n"
    "fmla v25.4s, v27.4s, v8.s[0]\n"
    "fmla v26.4s, v28.4s, v8.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v9.4s, v29.4s, v0.s[1]\n"
    "fmla v10.4s, v30.4s, v0.s[1]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "fmla v11.4s, v29.4s, v1.s[1]\n"
    "fmla v12.4s, v30.4s, v1.s[1]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "fmla v13.4s, v29.4s, v2.s[1]\n"
    "fmla v14.4s, v30.4s, v2.s[1]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "fmla v15.4s, v29.4s, v3.s[1]\n"
    "fmla v16.4s, v30.4s, v3.s[1]\n"
    "prfm pldl1keep, [x8, #192]\n"
    "fmla v17.4s, v29.4s, v4.s[1]\n"
    "fmla v18.4s, v30.4s, v4.s[1]\n"
    "prfm pldl1keep, [x8, #256]\n"
    "fmla v19.4s, v29.4s, v5.s[1]\n"
    "fmla v20.4s, v30.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v6.s[1]\n"
    "fmla v22.4s, v30.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v7.s[1]\n"
    "fmla v24.4s, v30.4s, v7.s[1]\n"
    "fmla v25.4s, v29.4s, v8.s[1]\n"
    "fmla v26.4s, v30.4s, v8.s[1]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v9.4s, v27.4s, v0.s[2]\n"
    "fmla v10.4s, v28.4s, v0.s[2]\n"
    "fmla v11.4s, v27.4s, v1.s[2]\n"
    "fmla v12.4s, v28.4s, v1.s[2]\n"
    "fmla v13.4s, v27.4s, v2.s[2]\n"
    "fmla v14.4s, v28.4s, v2.s[2]\n"
    "fmla v15.4s, v27.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v4.s[2]\n"
    "fmla v18.4s, v28.4s, v4.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v19.4s, v27.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v6.s[2]\n"
    "fmla v22.4s, v28.4s, v6.s[2]\n"
    "fmla v23.4s, v27.4s, v7.s[2]\n"
    "fmla v24.4s, v28.4s, v7.s[2]\n"
    "fmla v25.4s, v27.4s, v8.s[2]\n"
    "fmla v26.4s, v28.4s, v8.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v9.4s, v29.4s, v0.s[3]\n"
    "fmla v10.4s, v30.4s, v0.s[3]\n"
    "ldr q0, [x10, #0]\n"
    "fmla v11.4s, v29.4s, v1.s[3]\n"
    "fmla v12.4s, v30.4s, v1.s[3]\n"
    "ldr q1, [x10, #32]\n"
    "fmla v13.4s, v29.4s, v2.s[3]\n"
    "fmla v14.4s, v30.4s, v2.s[3]\n"
    "ldr q2, [x10, #64]\n"
    "fmla v15.4s, v29.4s, v3.s[3]\n"
    "fmla v16.4s, v30.4s, v3.s[3]\n"
    "ldr q3, [x10, #96]\n"
    "fmla v17.4s, v29.4s, v4.s[3]\n"
    "fmla v18.4s, v30.4s, v4.s[3]\n"
    "ldr q4, [x10, #128]\n"
    "fmla v19.4s, v29.4s, v5.s[3]\n"
    "fmla v20.4s, v30.4s, v5.s[3]\n"
    "ldr q5, [x10, #160]\n"
    "fmla v21.4s, v29.4s, v6.s[3]\n"
    "fmla v22.4s, v30.4s, v6.s[3]\n"
    "ldr q6, [x10, #192]\n"
    "fmla v23.4s, v29.4s, v7.s[3]\n"
    "fmla v24.4s, v30.4s, v7.s[3]\n"
    "ldr q7, [x10, #224]\n"
    "fmla v25.4s, v29.4s, v8.s[3]\n"
    "fmla v26.4s, v30.4s, v8.s[3]\n"
    "ldr q8, [x10, #256]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
    "b.ne LOOP_START%=\n"
// Remaining 8 channels
// K index 0
// Filter width idx 0
    "fmla v9.4s, v27.4s, v0.s[0]\n"
    "fmla v10.4s, v28.4s, v0.s[0]\n"
    "fmla v11.4s, v27.4s, v1.s[0]\n"
    "fmla v12.4s, v28.4s, v1.s[0]\n"
    "fmla v13.4s, v27.4s, v2.s[0]\n"
    "fmla v14.4s, v28.4s, v2.s[0]\n"
    "fmla v15.4s, v27.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v3.s[0]\n"
    "fmla v17.4s, v27.4s, v4.s[0]\n"
    "fmla v18.4s, v28.4s, v4.s[0]\n"
    "fmla v19.4s, v27.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v6.s[0]\n"
    "fmla v22.4s, v28.4s, v6.s[0]\n"
    "fmla v23.4s, v27.4s, v7.s[0]\n"
    "fmla v24.4s, v28.4s, v7.s[0]\n"
    "fmla v25.4s, v27.4s, v8.s[0]\n"
    "fmla v26.4s, v28.4s, v8.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v9.4s, v29.4s, v0.s[1]\n"
    "fmla v10.4s, v30.4s, v0.s[1]\n"
    "fmla v11.4s, v29.4s, v1.s[1]\n"
    "fmla v12.4s, v30.4s, v1.s[1]\n"
    "fmla v13.4s, v29.4s, v2.s[1]\n"
    "fmla v14.4s, v30.4s, v2.s[1]\n"
    "fmla v15.4s, v29.4s, v3.s[1]\n"
    "fmla v16.4s, v30.4s, v3.s[1]\n"
    "fmla v17.4s, v29.4s, v4.s[1]\n"
    "fmla v18.4s, v30.4s, v4.s[1]\n"
    "fmla v19.4s, v29.4s, v5.s[1]\n"
    "fmla v20.4s, v30.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v6.s[1]\n"
    "fmla v22.4s, v30.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v7.s[1]\n"
    "fmla v24.4s, v30.4s, v7.s[1]\n"
    "fmla v25.4s, v29.4s, v8.s[1]\n"
    "fmla v26.4s, v30.4s, v8.s[1]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v9.4s, v27.4s, v0.s[2]\n"
    "fmla v10.4s, v28.4s, v0.s[2]\n"
    "fmla v11.4s, v27.4s, v1.s[2]\n"
    "fmla v12.4s, v28.4s, v1.s[2]\n"
    "fmla v13.4s, v27.4s, v2.s[2]\n"
    "fmla v14.4s, v28.4s, v2.s[2]\n"
    "fmla v15.4s, v27.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v4.s[2]\n"
    "fmla v18.4s, v28.4s, v4.s[2]\n"
    "fmla v19.4s, v27.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v6.s[2]\n"
    "fmla v22.4s, v28.4s, v6.s[2]\n"
    "fmla v23.4s, v27.4s, v7.s[2]\n"
    "fmla v24.4s, v28.4s, v7.s[2]\n"
    "fmla v25.4s, v27.4s, v8.s[2]\n"
    "fmla v26.4s, v28.4s, v8.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v9.4s, v29.4s, v0.s[3]\n"
    "fmla v10.4s, v30.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v11.4s, v29.4s, v1.s[3]\n"
    "fmla v12.4s, v30.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v13.4s, v29.4s, v2.s[3]\n"
    "fmla v14.4s, v30.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v15.4s, v29.4s, v3.s[3]\n"
    "fmla v16.4s, v30.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v17.4s, v29.4s, v4.s[3]\n"
    "fmla v18.4s, v30.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "fmla v19.4s, v29.4s, v5.s[3]\n"
    "fmla v20.4s, v30.4s, v5.s[3]\n"
    "ldr q5, [x10, #176]\n"
    "fmla v21.4s, v29.4s, v6.s[3]\n"
    "fmla v22.4s, v30.4s, v6.s[3]\n"
    "ldr q6, [x10, #208]\n"
    "fmla v23.4s, v29.4s, v7.s[3]\n"
    "fmla v24.4s, v30.4s, v7.s[3]\n"
    "ldr q7, [x10, #240]\n"
    "fmla v25.4s, v29.4s, v8.s[3]\n"
    "fmla v26.4s, v30.4s, v8.s[3]\n"
    "ldr q8, [x10, #272]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v9.4s, v27.4s, v0.s[0]\n"
    "fmla v10.4s, v28.4s, v0.s[0]\n"
    "fmla v11.4s, v27.4s, v1.s[0]\n"
    "fmla v12.4s, v28.4s, v1.s[0]\n"
    "fmla v13.4s, v27.4s, v2.s[0]\n"
    "fmla v14.4s, v28.4s, v2.s[0]\n"
    "fmla v15.4s, v27.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v3.s[0]\n"
    "fmla v17.4s, v27.4s, v4.s[0]\n"
    "fmla v18.4s, v28.4s, v4.s[0]\n"
    "fmla v19.4s, v27.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v6.s[0]\n"
    "fmla v22.4s, v28.4s, v6.s[0]\n"
    "fmla v23.4s, v27.4s, v7.s[0]\n"
    "fmla v24.4s, v28.4s, v7.s[0]\n"
    "fmla v25.4s, v27.4s, v8.s[0]\n"
    "fmla v26.4s, v28.4s, v8.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v9.4s, v29.4s, v0.s[1]\n"
    "fmla v10.4s, v30.4s, v0.s[1]\n"
    "fmla v11.4s, v29.4s, v1.s[1]\n"
    "fmla v12.4s, v30.4s, v1.s[1]\n"
    "fmla v13.4s, v29.4s, v2.s[1]\n"
    "fmla v14.4s, v30.4s, v2.s[1]\n"
    "fmla v15.4s, v29.4s, v3.s[1]\n"
    "fmla v16.4s, v30.4s, v3.s[1]\n"
    "fmla v17.4s, v29.4s, v4.s[1]\n"
    "fmla v18.4s, v30.4s, v4.s[1]\n"
    "fmla v19.4s, v29.4s, v5.s[1]\n"
    "fmla v20.4s, v30.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v6.s[1]\n"
    "fmla v22.4s, v30.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v7.s[1]\n"
    "fmla v24.4s, v30.4s, v7.s[1]\n"
    "fmla v25.4s, v29.4s, v8.s[1]\n"
    "fmla v26.4s, v30.4s, v8.s[1]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v9.4s, v27.4s, v0.s[2]\n"
    "fmla v10.4s, v28.4s, v0.s[2]\n"
    "fmla v11.4s, v27.4s, v1.s[2]\n"
    "fmla v12.4s, v28.4s, v1.s[2]\n"
    "fmla v13.4s, v27.4s, v2.s[2]\n"
    "fmla v14.4s, v28.4s, v2.s[2]\n"
    "fmla v15.4s, v27.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v4.s[2]\n"
    "fmla v18.4s, v28.4s, v4.s[2]\n"
    "fmla v19.4s, v27.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v6.s[2]\n"
    "fmla v22.4s, v28.4s, v6.s[2]\n"
    "fmla v23.4s, v27.4s, v7.s[2]\n"
    "fmla v24.4s, v28.4s, v7.s[2]\n"
    "fmla v25.4s, v27.4s, v8.s[2]\n"
    "fmla v26.4s, v28.4s, v8.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v9.4s, v29.4s, v0.s[3]\n"
    "fmla v10.4s, v30.4s, v0.s[3]\n"
    "fmla v11.4s, v29.4s, v1.s[3]\n"
    "fmla v12.4s, v30.4s, v1.s[3]\n"
    "fmla v13.4s, v29.4s, v2.s[3]\n"
    "fmla v14.4s, v30.4s, v2.s[3]\n"
    "fmla v15.4s, v29.4s, v3.s[3]\n"
    "fmla v16.4s, v30.4s, v3.s[3]\n"
    "fmla v17.4s, v29.4s, v4.s[3]\n"
    "fmla v18.4s, v30.4s, v4.s[3]\n"
    "fmla v19.4s, v29.4s, v5.s[3]\n"
    "fmla v20.4s, v30.4s, v5.s[3]\n"
    "fmla v21.4s, v29.4s, v6.s[3]\n"
    "fmla v22.4s, v30.4s, v6.s[3]\n"
    "fmla v23.4s, v29.4s, v7.s[3]\n"
    "fmla v24.4s, v30.4s, v7.s[3]\n"
    "fmla v25.4s, v29.4s, v8.s[3]\n"
    "fmla v26.4s, v30.4s, v8.s[3]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
    "stp q9, q10, [%[out], #0]\n"
    "stp q11, q12, [%[out], #32]\n"
    "stp q13, q14, [%[out], #64]\n"
    "stp q15, q16, [%[out], #96]\n"
    "stp q17, q18, [%[out], #128]\n"
    "stp q19, q20, [%[out], #160]\n"
    "stp q21, q22, [%[out], #192]\n"
    "stp q23, q24, [%[out], #224]\n"
    "stp q25, q26, [%[out], #256]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [k] "r" (k), [inStr] "r" (inStride*sizeof(float))
    :   "x8", "x9", "x10", "x11",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
    );
    }
    #ifdef __DEBUG_PTMM_OFF
    printf ("Output After Kernel:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf("%6.3f\t", *(outputPtr + 8*5 + i));
        printf("%6.3f\t", *(outputPtr + 8*6 + i));
        printf("%6.3f\t", *(outputPtr + 8*7 + i));
        printf("%6.3f\t", *(outputPtr + 8*8 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif
}
void kernel_1_8_8_1_1_0_0(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0], [1], [2], [3], [4], [5], [6], [7]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7]
// Duplicate index: 0
// [1, 1, 1, 1, 1, 1, 1, 1]
// Number of Input Regs: 8, Filter Regs: 4 Output Regs: 16
// Total number of registers required: 28
// In  - [0, 1, 2, 3, 4, 5, 6, 7]
// Out - [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
// Fil - [24, 25, 26, 27, 24, 25]
// Register mapping diagram
//     0  1  2  3  4  5  6  7 
//
//24   8 10 12 14 16 18 20 22 
//25   9 11 13 15 17 19 21 23 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < (k/8); i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*2 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*3 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*4 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*5 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*6 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*7 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 1; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*1*8 + wf*8 + i));
            }
            printf("\n");
        }
    }
    printf ("Output:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf("%6.3f\t", *(outputPtr + 8*5 + i));
        printf("%6.3f\t", *(outputPtr + 8*6 + i));
        printf("%6.3f\t", *(outputPtr + 8*7 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif

    float* in = inputPtr + 0*8;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
    "add x8, %[in], %[inStr]\n"
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "prfm pldl1keep, [%[in], #64]\n"
    "prfm pldl1keep, [%[in], #128]\n"
    "prfm pldl1keep, [%[in], #192]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "prfm pldl1keep, [x8, #192]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q8, q9, [%[out], #0]\n"
    "ldp q10, q11, [%[out], #32]\n"
    "ldp q12, q13, [%[out], #64]\n"
    "ldp q14, q15, [%[out], #96]\n"
    "ldp q16, q17, [%[out], #128]\n"
    "ldp q18, q19, [%[out], #160]\n"
    "ldp q20, q21, [%[out], #192]\n"
    "ldp q22, q23, [%[out], #224]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v24.4s - v25.4s}, [x11], #32\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #32]\n"
    "ldr q2, [x10, #64]\n"
    "ldr q3, [x10, #96]\n"
    "ldr q4, [x10, #128]\n"
    "ldr q5, [x10, #160]\n"
    "ldr q6, [x10, #192]\n"
    "ldr q7, [x10, #224]\n"

    "sub w9, %w[k], #8\n"
    "LOOP_START%=:\n"
    "subs x9, x9, #8\n"
// K index 0
// Filter width idx 0
    "fmla v8.4s, v24.4s, v0.s[0]\n"
    "fmla v9.4s, v25.4s, v0.s[0]\n"
    "fmla v10.4s, v24.4s, v1.s[0]\n"
    "fmla v11.4s, v25.4s, v1.s[0]\n"
    "fmla v12.4s, v24.4s, v2.s[0]\n"
    "fmla v13.4s, v25.4s, v2.s[0]\n"
    "fmla v14.4s, v24.4s, v3.s[0]\n"
    "fmla v15.4s, v25.4s, v3.s[0]\n"
    "fmla v16.4s, v24.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v4.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v18.4s, v24.4s, v5.s[0]\n"
    "fmla v19.4s, v25.4s, v5.s[0]\n"
    "fmla v20.4s, v24.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v6.s[0]\n"
    "fmla v22.4s, v24.4s, v7.s[0]\n"
    "fmla v23.4s, v25.4s, v7.s[0]\n"
    "ld1 {v24.4s - v25.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v8.4s, v26.4s, v0.s[1]\n"
    "fmla v9.4s, v27.4s, v0.s[1]\n"
    "fmla v10.4s, v26.4s, v1.s[1]\n"
    "fmla v11.4s, v27.4s, v1.s[1]\n"
    "fmla v12.4s, v26.4s, v2.s[1]\n"
    "fmla v13.4s, v27.4s, v2.s[1]\n"
    "fmla v14.4s, v26.4s, v3.s[1]\n"
    "fmla v15.4s, v27.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v4.s[1]\n"
    "fmla v17.4s, v27.4s, v4.s[1]\n"
    "fmla v18.4s, v26.4s, v5.s[1]\n"
    "fmla v19.4s, v27.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v6.s[1]\n"
    "fmla v21.4s, v27.4s, v6.s[1]\n"
    "fmla v22.4s, v26.4s, v7.s[1]\n"
    "fmla v23.4s, v27.4s, v7.s[1]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v8.4s, v24.4s, v0.s[2]\n"
    "fmla v9.4s, v25.4s, v0.s[2]\n"
    "fmla v10.4s, v24.4s, v1.s[2]\n"
    "fmla v11.4s, v25.4s, v1.s[2]\n"
    "fmla v12.4s, v24.4s, v2.s[2]\n"
    "fmla v13.4s, v25.4s, v2.s[2]\n"
    "fmla v14.4s, v24.4s, v3.s[2]\n"
    "fmla v15.4s, v25.4s, v3.s[2]\n"
    "fmla v16.4s, v24.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v4.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v18.4s, v24.4s, v5.s[2]\n"
    "fmla v19.4s, v25.4s, v5.s[2]\n"
    "fmla v20.4s, v24.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v6.s[2]\n"
    "fmla v22.4s, v24.4s, v7.s[2]\n"
    "fmla v23.4s, v25.4s, v7.s[2]\n"
    "ld1 {v24.4s - v25.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v8.4s, v26.4s, v0.s[3]\n"
    "fmla v9.4s, v27.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v10.4s, v26.4s, v1.s[3]\n"
    "fmla v11.4s, v27.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v12.4s, v26.4s, v2.s[3]\n"
    "fmla v13.4s, v27.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v14.4s, v26.4s, v3.s[3]\n"
    "fmla v15.4s, v27.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v16.4s, v26.4s, v4.s[3]\n"
    "fmla v17.4s, v27.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "fmla v18.4s, v26.4s, v5.s[3]\n"
    "fmla v19.4s, v27.4s, v5.s[3]\n"
    "ldr q5, [x10, #176]\n"
    "fmla v20.4s, v26.4s, v6.s[3]\n"
    "fmla v21.4s, v27.4s, v6.s[3]\n"
    "ldr q6, [x10, #208]\n"
    "fmla v22.4s, v26.4s, v7.s[3]\n"
    "fmla v23.4s, v27.4s, v7.s[3]\n"
    "ldr q7, [x10, #240]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v8.4s, v24.4s, v0.s[0]\n"
    "fmla v9.4s, v25.4s, v0.s[0]\n"
    "fmla v10.4s, v24.4s, v1.s[0]\n"
    "fmla v11.4s, v25.4s, v1.s[0]\n"
    "fmla v12.4s, v24.4s, v2.s[0]\n"
    "fmla v13.4s, v25.4s, v2.s[0]\n"
    "add x10, x10, %[inStr]\n"
    "add x8, x8, %[inStr]\n"
    "fmla v14.4s, v24.4s, v3.s[0]\n"
    "fmla v15.4s, v25.4s, v3.s[0]\n"
    "fmla v16.4s, v24.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v4.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v18.4s, v24.4s, v5.s[0]\n"
    "fmla v19.4s, v25.4s, v5.s[0]\n"
    "fmla v20.4s, v24.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v6.s[0]\n"
    "fmla v22.4s, v24.4s, v7.s[0]\n"
    "fmla v23.4s, v25.4s, v7.s[0]\n"
    "ld1 {v24.4s - v25.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v8.4s, v26.4s, v0.s[1]\n"
    "fmla v9.4s, v27.4s, v0.s[1]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "fmla v10.4s, v26.4s, v1.s[1]\n"
    "fmla v11.4s, v27.4s, v1.s[1]\n"
    "fmla v12.4s, v26.4s, v2.s[1]\n"
    "fmla v13.4s, v27.4s, v2.s[1]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "fmla v14.4s, v26.4s, v3.s[1]\n"
    "fmla v15.4s, v27.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v4.s[1]\n"
    "fmla v17.4s, v27.4s, v4.s[1]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "fmla v18.4s, v26.4s, v5.s[1]\n"
    "fmla v19.4s, v27.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v6.s[1]\n"
    "fmla v21.4s, v27.4s, v6.s[1]\n"
    "prfm pldl1keep, [x8, #192]\n"
    "fmla v22.4s, v26.4s, v7.s[1]\n"
    "fmla v23.4s, v27.4s, v7.s[1]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v8.4s, v24.4s, v0.s[2]\n"
    "fmla v9.4s, v25.4s, v0.s[2]\n"
    "fmla v10.4s, v24.4s, v1.s[2]\n"
    "fmla v11.4s, v25.4s, v1.s[2]\n"
    "fmla v12.4s, v24.4s, v2.s[2]\n"
    "fmla v13.4s, v25.4s, v2.s[2]\n"
    "fmla v14.4s, v24.4s, v3.s[2]\n"
    "fmla v15.4s, v25.4s, v3.s[2]\n"
    "fmla v16.4s, v24.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v4.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v18.4s, v24.4s, v5.s[2]\n"
    "fmla v19.4s, v25.4s, v5.s[2]\n"
    "fmla v20.4s, v24.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v6.s[2]\n"
    "fmla v22.4s, v24.4s, v7.s[2]\n"
    "fmla v23.4s, v25.4s, v7.s[2]\n"
    "ld1 {v24.4s - v25.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v8.4s, v26.4s, v0.s[3]\n"
    "fmla v9.4s, v27.4s, v0.s[3]\n"
    "ldr q0, [x10, #0]\n"
    "fmla v10.4s, v26.4s, v1.s[3]\n"
    "fmla v11.4s, v27.4s, v1.s[3]\n"
    "ldr q1, [x10, #32]\n"
    "fmla v12.4s, v26.4s, v2.s[3]\n"
    "fmla v13.4s, v27.4s, v2.s[3]\n"
    "ldr q2, [x10, #64]\n"
    "fmla v14.4s, v26.4s, v3.s[3]\n"
    "fmla v15.4s, v27.4s, v3.s[3]\n"
    "ldr q3, [x10, #96]\n"
    "fmla v16.4s, v26.4s, v4.s[3]\n"
    "fmla v17.4s, v27.4s, v4.s[3]\n"
    "ldr q4, [x10, #128]\n"
    "fmla v18.4s, v26.4s, v5.s[3]\n"
    "fmla v19.4s, v27.4s, v5.s[3]\n"
    "ldr q5, [x10, #160]\n"
    "fmla v20.4s, v26.4s, v6.s[3]\n"
    "fmla v21.4s, v27.4s, v6.s[3]\n"
    "ldr q6, [x10, #192]\n"
    "fmla v22.4s, v26.4s, v7.s[3]\n"
    "fmla v23.4s, v27.4s, v7.s[3]\n"
    "ldr q7, [x10, #224]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
    "b.ne LOOP_START%=\n"
// Remaining 8 channels
// K index 0
// Filter width idx 0
    "fmla v8.4s, v24.4s, v0.s[0]\n"
    "fmla v9.4s, v25.4s, v0.s[0]\n"
    "fmla v10.4s, v24.4s, v1.s[0]\n"
    "fmla v11.4s, v25.4s, v1.s[0]\n"
    "fmla v12.4s, v24.4s, v2.s[0]\n"
    "fmla v13.4s, v25.4s, v2.s[0]\n"
    "fmla v14.4s, v24.4s, v3.s[0]\n"
    "fmla v15.4s, v25.4s, v3.s[0]\n"
    "fmla v16.4s, v24.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v4.s[0]\n"
    "fmla v18.4s, v24.4s, v5.s[0]\n"
    "fmla v19.4s, v25.4s, v5.s[0]\n"
    "fmla v20.4s, v24.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v6.s[0]\n"
    "fmla v22.4s, v24.4s, v7.s[0]\n"
    "fmla v23.4s, v25.4s, v7.s[0]\n"
    "ld1 {v24.4s - v25.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v8.4s, v26.4s, v0.s[1]\n"
    "fmla v9.4s, v27.4s, v0.s[1]\n"
    "fmla v10.4s, v26.4s, v1.s[1]\n"
    "fmla v11.4s, v27.4s, v1.s[1]\n"
    "fmla v12.4s, v26.4s, v2.s[1]\n"
    "fmla v13.4s, v27.4s, v2.s[1]\n"
    "fmla v14.4s, v26.4s, v3.s[1]\n"
    "fmla v15.4s, v27.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v4.s[1]\n"
    "fmla v17.4s, v27.4s, v4.s[1]\n"
    "fmla v18.4s, v26.4s, v5.s[1]\n"
    "fmla v19.4s, v27.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v6.s[1]\n"
    "fmla v21.4s, v27.4s, v6.s[1]\n"
    "fmla v22.4s, v26.4s, v7.s[1]\n"
    "fmla v23.4s, v27.4s, v7.s[1]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v8.4s, v24.4s, v0.s[2]\n"
    "fmla v9.4s, v25.4s, v0.s[2]\n"
    "fmla v10.4s, v24.4s, v1.s[2]\n"
    "fmla v11.4s, v25.4s, v1.s[2]\n"
    "fmla v12.4s, v24.4s, v2.s[2]\n"
    "fmla v13.4s, v25.4s, v2.s[2]\n"
    "fmla v14.4s, v24.4s, v3.s[2]\n"
    "fmla v15.4s, v25.4s, v3.s[2]\n"
    "fmla v16.4s, v24.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v4.s[2]\n"
    "fmla v18.4s, v24.4s, v5.s[2]\n"
    "fmla v19.4s, v25.4s, v5.s[2]\n"
    "fmla v20.4s, v24.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v6.s[2]\n"
    "fmla v22.4s, v24.4s, v7.s[2]\n"
    "fmla v23.4s, v25.4s, v7.s[2]\n"
    "ld1 {v24.4s - v25.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v8.4s, v26.4s, v0.s[3]\n"
    "fmla v9.4s, v27.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v10.4s, v26.4s, v1.s[3]\n"
    "fmla v11.4s, v27.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v12.4s, v26.4s, v2.s[3]\n"
    "fmla v13.4s, v27.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v14.4s, v26.4s, v3.s[3]\n"
    "fmla v15.4s, v27.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v16.4s, v26.4s, v4.s[3]\n"
    "fmla v17.4s, v27.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "fmla v18.4s, v26.4s, v5.s[3]\n"
    "fmla v19.4s, v27.4s, v5.s[3]\n"
    "ldr q5, [x10, #176]\n"
    "fmla v20.4s, v26.4s, v6.s[3]\n"
    "fmla v21.4s, v27.4s, v6.s[3]\n"
    "ldr q6, [x10, #208]\n"
    "fmla v22.4s, v26.4s, v7.s[3]\n"
    "fmla v23.4s, v27.4s, v7.s[3]\n"
    "ldr q7, [x10, #240]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v8.4s, v24.4s, v0.s[0]\n"
    "fmla v9.4s, v25.4s, v0.s[0]\n"
    "fmla v10.4s, v24.4s, v1.s[0]\n"
    "fmla v11.4s, v25.4s, v1.s[0]\n"
    "fmla v12.4s, v24.4s, v2.s[0]\n"
    "fmla v13.4s, v25.4s, v2.s[0]\n"
    "fmla v14.4s, v24.4s, v3.s[0]\n"
    "fmla v15.4s, v25.4s, v3.s[0]\n"
    "fmla v16.4s, v24.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v4.s[0]\n"
    "fmla v18.4s, v24.4s, v5.s[0]\n"
    "fmla v19.4s, v25.4s, v5.s[0]\n"
    "fmla v20.4s, v24.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v6.s[0]\n"
    "fmla v22.4s, v24.4s, v7.s[0]\n"
    "fmla v23.4s, v25.4s, v7.s[0]\n"
    "ld1 {v24.4s - v25.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v8.4s, v26.4s, v0.s[1]\n"
    "fmla v9.4s, v27.4s, v0.s[1]\n"
    "fmla v10.4s, v26.4s, v1.s[1]\n"
    "fmla v11.4s, v27.4s, v1.s[1]\n"
    "fmla v12.4s, v26.4s, v2.s[1]\n"
    "fmla v13.4s, v27.4s, v2.s[1]\n"
    "fmla v14.4s, v26.4s, v3.s[1]\n"
    "fmla v15.4s, v27.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v4.s[1]\n"
    "fmla v17.4s, v27.4s, v4.s[1]\n"
    "fmla v18.4s, v26.4s, v5.s[1]\n"
    "fmla v19.4s, v27.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v6.s[1]\n"
    "fmla v21.4s, v27.4s, v6.s[1]\n"
    "fmla v22.4s, v26.4s, v7.s[1]\n"
    "fmla v23.4s, v27.4s, v7.s[1]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v8.4s, v24.4s, v0.s[2]\n"
    "fmla v9.4s, v25.4s, v0.s[2]\n"
    "fmla v10.4s, v24.4s, v1.s[2]\n"
    "fmla v11.4s, v25.4s, v1.s[2]\n"
    "fmla v12.4s, v24.4s, v2.s[2]\n"
    "fmla v13.4s, v25.4s, v2.s[2]\n"
    "fmla v14.4s, v24.4s, v3.s[2]\n"
    "fmla v15.4s, v25.4s, v3.s[2]\n"
    "fmla v16.4s, v24.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v4.s[2]\n"
    "fmla v18.4s, v24.4s, v5.s[2]\n"
    "fmla v19.4s, v25.4s, v5.s[2]\n"
    "fmla v20.4s, v24.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v6.s[2]\n"
    "fmla v22.4s, v24.4s, v7.s[2]\n"
    "fmla v23.4s, v25.4s, v7.s[2]\n"
    "ld1 {v24.4s - v25.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v8.4s, v26.4s, v0.s[3]\n"
    "fmla v9.4s, v27.4s, v0.s[3]\n"
    "fmla v10.4s, v26.4s, v1.s[3]\n"
    "fmla v11.4s, v27.4s, v1.s[3]\n"
    "fmla v12.4s, v26.4s, v2.s[3]\n"
    "fmla v13.4s, v27.4s, v2.s[3]\n"
    "fmla v14.4s, v26.4s, v3.s[3]\n"
    "fmla v15.4s, v27.4s, v3.s[3]\n"
    "fmla v16.4s, v26.4s, v4.s[3]\n"
    "fmla v17.4s, v27.4s, v4.s[3]\n"
    "fmla v18.4s, v26.4s, v5.s[3]\n"
    "fmla v19.4s, v27.4s, v5.s[3]\n"
    "fmla v20.4s, v26.4s, v6.s[3]\n"
    "fmla v21.4s, v27.4s, v6.s[3]\n"
    "fmla v22.4s, v26.4s, v7.s[3]\n"
    "fmla v23.4s, v27.4s, v7.s[3]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
    "stp q8, q9, [%[out], #0]\n"
    "stp q10, q11, [%[out], #32]\n"
    "stp q12, q13, [%[out], #64]\n"
    "stp q14, q15, [%[out], #96]\n"
    "stp q16, q17, [%[out], #128]\n"
    "stp q18, q19, [%[out], #160]\n"
    "stp q20, q21, [%[out], #192]\n"
    "stp q22, q23, [%[out], #224]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [k] "r" (k), [inStr] "r" (inStride*sizeof(float))
    :   "x8", "x9", "x10", "x11",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
    );
    }
    #ifdef __DEBUG_PTMM_OFF
    printf ("Output After Kernel:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf("%6.3f\t", *(outputPtr + 8*5 + i));
        printf("%6.3f\t", *(outputPtr + 8*6 + i));
        printf("%6.3f\t", *(outputPtr + 8*7 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif
}
void kernel_1_7_8_1_1_0_0(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0], [1], [2], [3], [4], [5], [6]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6]
// Duplicate index: 0
// [1, 1, 1, 1, 1, 1, 1]
// Number of Input Regs: 7, Filter Regs: 4 Output Regs: 14
// Total number of registers required: 25
// In  - [0, 1, 2, 3, 4, 5, 6]
// Out - [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
// Fil - [21, 22, 23, 24, 21, 22]
// Register mapping diagram
//     0  1  2  3  4  5  6 
//
//21   7  9 11 13 15 17 19 
//22   8 10 12 14 16 18 20 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < (k/8); i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*2 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*3 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*4 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*5 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*6 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 1; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*1*8 + wf*8 + i));
            }
            printf("\n");
        }
    }
    printf ("Output:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf("%6.3f\t", *(outputPtr + 8*5 + i));
        printf("%6.3f\t", *(outputPtr + 8*6 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif

    float* in = inputPtr + 0*8;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
    "add x8, %[in], %[inStr]\n"
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "prfm pldl1keep, [%[in], #64]\n"
    "prfm pldl1keep, [%[in], #128]\n"
    "prfm pldl1keep, [%[in], #192]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "prfm pldl1keep, [x8, #192]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q7, q8, [%[out], #0]\n"
    "ldp q9, q10, [%[out], #32]\n"
    "ldp q11, q12, [%[out], #64]\n"
    "ldp q13, q14, [%[out], #96]\n"
    "ldp q15, q16, [%[out], #128]\n"
    "ldp q17, q18, [%[out], #160]\n"
    "ldp q19, q20, [%[out], #192]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #32]\n"
    "ldr q2, [x10, #64]\n"
    "ldr q3, [x10, #96]\n"
    "ldr q4, [x10, #128]\n"
    "ldr q5, [x10, #160]\n"
    "ldr q6, [x10, #192]\n"

    "sub w9, %w[k], #8\n"
    "LOOP_START%=:\n"
    "subs x9, x9, #8\n"
// K index 0
// Filter width idx 0
    "fmla v7.4s, v21.4s, v0.s[0]\n"
    "fmla v8.4s, v22.4s, v0.s[0]\n"
    "fmla v9.4s, v21.4s, v1.s[0]\n"
    "fmla v10.4s, v22.4s, v1.s[0]\n"
    "fmla v11.4s, v21.4s, v2.s[0]\n"
    "fmla v12.4s, v22.4s, v2.s[0]\n"
    "fmla v13.4s, v21.4s, v3.s[0]\n"
    "fmla v14.4s, v22.4s, v3.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v15.4s, v21.4s, v4.s[0]\n"
    "fmla v16.4s, v22.4s, v4.s[0]\n"
    "fmla v17.4s, v21.4s, v5.s[0]\n"
    "fmla v18.4s, v22.4s, v5.s[0]\n"
    "fmla v19.4s, v21.4s, v6.s[0]\n"
    "fmla v20.4s, v22.4s, v6.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v7.4s, v23.4s, v0.s[1]\n"
    "fmla v8.4s, v24.4s, v0.s[1]\n"
    "fmla v9.4s, v23.4s, v1.s[1]\n"
    "fmla v10.4s, v24.4s, v1.s[1]\n"
    "fmla v11.4s, v23.4s, v2.s[1]\n"
    "fmla v12.4s, v24.4s, v2.s[1]\n"
    "fmla v13.4s, v23.4s, v3.s[1]\n"
    "fmla v14.4s, v24.4s, v3.s[1]\n"
    "fmla v15.4s, v23.4s, v4.s[1]\n"
    "fmla v16.4s, v24.4s, v4.s[1]\n"
    "fmla v17.4s, v23.4s, v5.s[1]\n"
    "fmla v18.4s, v24.4s, v5.s[1]\n"
    "fmla v19.4s, v23.4s, v6.s[1]\n"
    "fmla v20.4s, v24.4s, v6.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v7.4s, v21.4s, v0.s[2]\n"
    "fmla v8.4s, v22.4s, v0.s[2]\n"
    "fmla v9.4s, v21.4s, v1.s[2]\n"
    "fmla v10.4s, v22.4s, v1.s[2]\n"
    "fmla v11.4s, v21.4s, v2.s[2]\n"
    "fmla v12.4s, v22.4s, v2.s[2]\n"
    "fmla v13.4s, v21.4s, v3.s[2]\n"
    "fmla v14.4s, v22.4s, v3.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v15.4s, v21.4s, v4.s[2]\n"
    "fmla v16.4s, v22.4s, v4.s[2]\n"
    "fmla v17.4s, v21.4s, v5.s[2]\n"
    "fmla v18.4s, v22.4s, v5.s[2]\n"
    "fmla v19.4s, v21.4s, v6.s[2]\n"
    "fmla v20.4s, v22.4s, v6.s[2]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v7.4s, v23.4s, v0.s[3]\n"
    "fmla v8.4s, v24.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v9.4s, v23.4s, v1.s[3]\n"
    "fmla v10.4s, v24.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v11.4s, v23.4s, v2.s[3]\n"
    "fmla v12.4s, v24.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v13.4s, v23.4s, v3.s[3]\n"
    "fmla v14.4s, v24.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v15.4s, v23.4s, v4.s[3]\n"
    "fmla v16.4s, v24.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "fmla v17.4s, v23.4s, v5.s[3]\n"
    "fmla v18.4s, v24.4s, v5.s[3]\n"
    "ldr q5, [x10, #176]\n"
    "fmla v19.4s, v23.4s, v6.s[3]\n"
    "fmla v20.4s, v24.4s, v6.s[3]\n"
    "ldr q6, [x10, #208]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v7.4s, v21.4s, v0.s[0]\n"
    "fmla v8.4s, v22.4s, v0.s[0]\n"
    "fmla v9.4s, v21.4s, v1.s[0]\n"
    "fmla v10.4s, v22.4s, v1.s[0]\n"
    "fmla v11.4s, v21.4s, v2.s[0]\n"
    "fmla v12.4s, v22.4s, v2.s[0]\n"
    "add x10, x10, %[inStr]\n"
    "add x8, x8, %[inStr]\n"
    "fmla v13.4s, v21.4s, v3.s[0]\n"
    "fmla v14.4s, v22.4s, v3.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v15.4s, v21.4s, v4.s[0]\n"
    "fmla v16.4s, v22.4s, v4.s[0]\n"
    "fmla v17.4s, v21.4s, v5.s[0]\n"
    "fmla v18.4s, v22.4s, v5.s[0]\n"
    "fmla v19.4s, v21.4s, v6.s[0]\n"
    "fmla v20.4s, v22.4s, v6.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v7.4s, v23.4s, v0.s[1]\n"
    "fmla v8.4s, v24.4s, v0.s[1]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "fmla v9.4s, v23.4s, v1.s[1]\n"
    "fmla v10.4s, v24.4s, v1.s[1]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "fmla v11.4s, v23.4s, v2.s[1]\n"
    "fmla v12.4s, v24.4s, v2.s[1]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "fmla v13.4s, v23.4s, v3.s[1]\n"
    "fmla v14.4s, v24.4s, v3.s[1]\n"
    "prfm pldl1keep, [x8, #192]\n"
    "fmla v15.4s, v23.4s, v4.s[1]\n"
    "fmla v16.4s, v24.4s, v4.s[1]\n"
    "fmla v17.4s, v23.4s, v5.s[1]\n"
    "fmla v18.4s, v24.4s, v5.s[1]\n"
    "fmla v19.4s, v23.4s, v6.s[1]\n"
    "fmla v20.4s, v24.4s, v6.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v7.4s, v21.4s, v0.s[2]\n"
    "fmla v8.4s, v22.4s, v0.s[2]\n"
    "fmla v9.4s, v21.4s, v1.s[2]\n"
    "fmla v10.4s, v22.4s, v1.s[2]\n"
    "fmla v11.4s, v21.4s, v2.s[2]\n"
    "fmla v12.4s, v22.4s, v2.s[2]\n"
    "fmla v13.4s, v21.4s, v3.s[2]\n"
    "fmla v14.4s, v22.4s, v3.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v15.4s, v21.4s, v4.s[2]\n"
    "fmla v16.4s, v22.4s, v4.s[2]\n"
    "fmla v17.4s, v21.4s, v5.s[2]\n"
    "fmla v18.4s, v22.4s, v5.s[2]\n"
    "fmla v19.4s, v21.4s, v6.s[2]\n"
    "fmla v20.4s, v22.4s, v6.s[2]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v7.4s, v23.4s, v0.s[3]\n"
    "fmla v8.4s, v24.4s, v0.s[3]\n"
    "ldr q0, [x10, #0]\n"
    "fmla v9.4s, v23.4s, v1.s[3]\n"
    "fmla v10.4s, v24.4s, v1.s[3]\n"
    "ldr q1, [x10, #32]\n"
    "fmla v11.4s, v23.4s, v2.s[3]\n"
    "fmla v12.4s, v24.4s, v2.s[3]\n"
    "ldr q2, [x10, #64]\n"
    "fmla v13.4s, v23.4s, v3.s[3]\n"
    "fmla v14.4s, v24.4s, v3.s[3]\n"
    "ldr q3, [x10, #96]\n"
    "fmla v15.4s, v23.4s, v4.s[3]\n"
    "fmla v16.4s, v24.4s, v4.s[3]\n"
    "ldr q4, [x10, #128]\n"
    "fmla v17.4s, v23.4s, v5.s[3]\n"
    "fmla v18.4s, v24.4s, v5.s[3]\n"
    "ldr q5, [x10, #160]\n"
    "fmla v19.4s, v23.4s, v6.s[3]\n"
    "fmla v20.4s, v24.4s, v6.s[3]\n"
    "ldr q6, [x10, #192]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
    "b.ne LOOP_START%=\n"
// Remaining 8 channels
// K index 0
// Filter width idx 0
    "fmla v7.4s, v21.4s, v0.s[0]\n"
    "fmla v8.4s, v22.4s, v0.s[0]\n"
    "fmla v9.4s, v21.4s, v1.s[0]\n"
    "fmla v10.4s, v22.4s, v1.s[0]\n"
    "fmla v11.4s, v21.4s, v2.s[0]\n"
    "fmla v12.4s, v22.4s, v2.s[0]\n"
    "fmla v13.4s, v21.4s, v3.s[0]\n"
    "fmla v14.4s, v22.4s, v3.s[0]\n"
    "fmla v15.4s, v21.4s, v4.s[0]\n"
    "fmla v16.4s, v22.4s, v4.s[0]\n"
    "fmla v17.4s, v21.4s, v5.s[0]\n"
    "fmla v18.4s, v22.4s, v5.s[0]\n"
    "fmla v19.4s, v21.4s, v6.s[0]\n"
    "fmla v20.4s, v22.4s, v6.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v7.4s, v23.4s, v0.s[1]\n"
    "fmla v8.4s, v24.4s, v0.s[1]\n"
    "fmla v9.4s, v23.4s, v1.s[1]\n"
    "fmla v10.4s, v24.4s, v1.s[1]\n"
    "fmla v11.4s, v23.4s, v2.s[1]\n"
    "fmla v12.4s, v24.4s, v2.s[1]\n"
    "fmla v13.4s, v23.4s, v3.s[1]\n"
    "fmla v14.4s, v24.4s, v3.s[1]\n"
    "fmla v15.4s, v23.4s, v4.s[1]\n"
    "fmla v16.4s, v24.4s, v4.s[1]\n"
    "fmla v17.4s, v23.4s, v5.s[1]\n"
    "fmla v18.4s, v24.4s, v5.s[1]\n"
    "fmla v19.4s, v23.4s, v6.s[1]\n"
    "fmla v20.4s, v24.4s, v6.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v7.4s, v21.4s, v0.s[2]\n"
    "fmla v8.4s, v22.4s, v0.s[2]\n"
    "fmla v9.4s, v21.4s, v1.s[2]\n"
    "fmla v10.4s, v22.4s, v1.s[2]\n"
    "fmla v11.4s, v21.4s, v2.s[2]\n"
    "fmla v12.4s, v22.4s, v2.s[2]\n"
    "fmla v13.4s, v21.4s, v3.s[2]\n"
    "fmla v14.4s, v22.4s, v3.s[2]\n"
    "fmla v15.4s, v21.4s, v4.s[2]\n"
    "fmla v16.4s, v22.4s, v4.s[2]\n"
    "fmla v17.4s, v21.4s, v5.s[2]\n"
    "fmla v18.4s, v22.4s, v5.s[2]\n"
    "fmla v19.4s, v21.4s, v6.s[2]\n"
    "fmla v20.4s, v22.4s, v6.s[2]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v7.4s, v23.4s, v0.s[3]\n"
    "fmla v8.4s, v24.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v9.4s, v23.4s, v1.s[3]\n"
    "fmla v10.4s, v24.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v11.4s, v23.4s, v2.s[3]\n"
    "fmla v12.4s, v24.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v13.4s, v23.4s, v3.s[3]\n"
    "fmla v14.4s, v24.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v15.4s, v23.4s, v4.s[3]\n"
    "fmla v16.4s, v24.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "fmla v17.4s, v23.4s, v5.s[3]\n"
    "fmla v18.4s, v24.4s, v5.s[3]\n"
    "ldr q5, [x10, #176]\n"
    "fmla v19.4s, v23.4s, v6.s[3]\n"
    "fmla v20.4s, v24.4s, v6.s[3]\n"
    "ldr q6, [x10, #208]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v7.4s, v21.4s, v0.s[0]\n"
    "fmla v8.4s, v22.4s, v0.s[0]\n"
    "fmla v9.4s, v21.4s, v1.s[0]\n"
    "fmla v10.4s, v22.4s, v1.s[0]\n"
    "fmla v11.4s, v21.4s, v2.s[0]\n"
    "fmla v12.4s, v22.4s, v2.s[0]\n"
    "fmla v13.4s, v21.4s, v3.s[0]\n"
    "fmla v14.4s, v22.4s, v3.s[0]\n"
    "fmla v15.4s, v21.4s, v4.s[0]\n"
    "fmla v16.4s, v22.4s, v4.s[0]\n"
    "fmla v17.4s, v21.4s, v5.s[0]\n"
    "fmla v18.4s, v22.4s, v5.s[0]\n"
    "fmla v19.4s, v21.4s, v6.s[0]\n"
    "fmla v20.4s, v22.4s, v6.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v7.4s, v23.4s, v0.s[1]\n"
    "fmla v8.4s, v24.4s, v0.s[1]\n"
    "fmla v9.4s, v23.4s, v1.s[1]\n"
    "fmla v10.4s, v24.4s, v1.s[1]\n"
    "fmla v11.4s, v23.4s, v2.s[1]\n"
    "fmla v12.4s, v24.4s, v2.s[1]\n"
    "fmla v13.4s, v23.4s, v3.s[1]\n"
    "fmla v14.4s, v24.4s, v3.s[1]\n"
    "fmla v15.4s, v23.4s, v4.s[1]\n"
    "fmla v16.4s, v24.4s, v4.s[1]\n"
    "fmla v17.4s, v23.4s, v5.s[1]\n"
    "fmla v18.4s, v24.4s, v5.s[1]\n"
    "fmla v19.4s, v23.4s, v6.s[1]\n"
    "fmla v20.4s, v24.4s, v6.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v7.4s, v21.4s, v0.s[2]\n"
    "fmla v8.4s, v22.4s, v0.s[2]\n"
    "fmla v9.4s, v21.4s, v1.s[2]\n"
    "fmla v10.4s, v22.4s, v1.s[2]\n"
    "fmla v11.4s, v21.4s, v2.s[2]\n"
    "fmla v12.4s, v22.4s, v2.s[2]\n"
    "fmla v13.4s, v21.4s, v3.s[2]\n"
    "fmla v14.4s, v22.4s, v3.s[2]\n"
    "fmla v15.4s, v21.4s, v4.s[2]\n"
    "fmla v16.4s, v22.4s, v4.s[2]\n"
    "fmla v17.4s, v21.4s, v5.s[2]\n"
    "fmla v18.4s, v22.4s, v5.s[2]\n"
    "fmla v19.4s, v21.4s, v6.s[2]\n"
    "fmla v20.4s, v22.4s, v6.s[2]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v7.4s, v23.4s, v0.s[3]\n"
    "fmla v8.4s, v24.4s, v0.s[3]\n"
    "fmla v9.4s, v23.4s, v1.s[3]\n"
    "fmla v10.4s, v24.4s, v1.s[3]\n"
    "fmla v11.4s, v23.4s, v2.s[3]\n"
    "fmla v12.4s, v24.4s, v2.s[3]\n"
    "fmla v13.4s, v23.4s, v3.s[3]\n"
    "fmla v14.4s, v24.4s, v3.s[3]\n"
    "fmla v15.4s, v23.4s, v4.s[3]\n"
    "fmla v16.4s, v24.4s, v4.s[3]\n"
    "fmla v17.4s, v23.4s, v5.s[3]\n"
    "fmla v18.4s, v24.4s, v5.s[3]\n"
    "fmla v19.4s, v23.4s, v6.s[3]\n"
    "fmla v20.4s, v24.4s, v6.s[3]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
    "stp q7, q8, [%[out], #0]\n"
    "stp q9, q10, [%[out], #32]\n"
    "stp q11, q12, [%[out], #64]\n"
    "stp q13, q14, [%[out], #96]\n"
    "stp q15, q16, [%[out], #128]\n"
    "stp q17, q18, [%[out], #160]\n"
    "stp q19, q20, [%[out], #192]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [k] "r" (k), [inStr] "r" (inStride*sizeof(float))
    :   "x8", "x9", "x10", "x11",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
    );
    }
    #ifdef __DEBUG_PTMM_OFF
    printf ("Output After Kernel:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf("%6.3f\t", *(outputPtr + 8*5 + i));
        printf("%6.3f\t", *(outputPtr + 8*6 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif
}
void kernel_1_6_8_1_1_0_0(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0], [1], [2], [3], [4], [5]]
// Input registers required
// [0, 1, 2, 3, 4, 5]
// Duplicate index: 0
// [1, 1, 1, 1, 1, 1]
// Number of Input Regs: 6, Filter Regs: 4 Output Regs: 12
// Total number of registers required: 22
// In  - [0, 1, 2, 3, 4, 5]
// Out - [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
// Fil - [18, 19, 20, 21, 18, 19]
// Register mapping diagram
//     0  1  2  3  4  5 
//
//18   6  8 10 12 14 16 
//19   7  9 11 13 15 17 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < (k/8); i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*2 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*3 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*4 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*5 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 1; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*1*8 + wf*8 + i));
            }
            printf("\n");
        }
    }
    printf ("Output:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf("%6.3f\t", *(outputPtr + 8*5 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif

    float* in = inputPtr + 0*8;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
    "add x8, %[in], %[inStr]\n"
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "prfm pldl1keep, [%[in], #64]\n"
    "prfm pldl1keep, [%[in], #128]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q6, q7, [%[out], #0]\n"
    "ldp q8, q9, [%[out], #32]\n"
    "ldp q10, q11, [%[out], #64]\n"
    "ldp q12, q13, [%[out], #96]\n"
    "ldp q14, q15, [%[out], #128]\n"
    "ldp q16, q17, [%[out], #160]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #32]\n"
    "ldr q2, [x10, #64]\n"
    "ldr q3, [x10, #96]\n"
    "ldr q4, [x10, #128]\n"
    "ldr q5, [x10, #160]\n"

    "sub w9, %w[k], #8\n"
    "LOOP_START%=:\n"
    "subs x9, x9, #8\n"
// K index 0
// Filter width idx 0
    "fmla v6.4s, v18.4s, v0.s[0]\n"
    "fmla v7.4s, v19.4s, v0.s[0]\n"
    "fmla v8.4s, v18.4s, v1.s[0]\n"
    "fmla v9.4s, v19.4s, v1.s[0]\n"
    "fmla v10.4s, v18.4s, v2.s[0]\n"
    "fmla v11.4s, v19.4s, v2.s[0]\n"
    "fmla v12.4s, v18.4s, v3.s[0]\n"
    "fmla v13.4s, v19.4s, v3.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v14.4s, v18.4s, v4.s[0]\n"
    "fmla v15.4s, v19.4s, v4.s[0]\n"
    "fmla v16.4s, v18.4s, v5.s[0]\n"
    "fmla v17.4s, v19.4s, v5.s[0]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v6.4s, v20.4s, v0.s[1]\n"
    "fmla v7.4s, v21.4s, v0.s[1]\n"
    "fmla v8.4s, v20.4s, v1.s[1]\n"
    "fmla v9.4s, v21.4s, v1.s[1]\n"
    "fmla v10.4s, v20.4s, v2.s[1]\n"
    "fmla v11.4s, v21.4s, v2.s[1]\n"
    "fmla v12.4s, v20.4s, v3.s[1]\n"
    "fmla v13.4s, v21.4s, v3.s[1]\n"
    "fmla v14.4s, v20.4s, v4.s[1]\n"
    "fmla v15.4s, v21.4s, v4.s[1]\n"
    "fmla v16.4s, v20.4s, v5.s[1]\n"
    "fmla v17.4s, v21.4s, v5.s[1]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v6.4s, v18.4s, v0.s[2]\n"
    "fmla v7.4s, v19.4s, v0.s[2]\n"
    "fmla v8.4s, v18.4s, v1.s[2]\n"
    "fmla v9.4s, v19.4s, v1.s[2]\n"
    "fmla v10.4s, v18.4s, v2.s[2]\n"
    "fmla v11.4s, v19.4s, v2.s[2]\n"
    "fmla v12.4s, v18.4s, v3.s[2]\n"
    "fmla v13.4s, v19.4s, v3.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v14.4s, v18.4s, v4.s[2]\n"
    "fmla v15.4s, v19.4s, v4.s[2]\n"
    "fmla v16.4s, v18.4s, v5.s[2]\n"
    "fmla v17.4s, v19.4s, v5.s[2]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v6.4s, v20.4s, v0.s[3]\n"
    "fmla v7.4s, v21.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v8.4s, v20.4s, v1.s[3]\n"
    "fmla v9.4s, v21.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v10.4s, v20.4s, v2.s[3]\n"
    "fmla v11.4s, v21.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v12.4s, v20.4s, v3.s[3]\n"
    "fmla v13.4s, v21.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v14.4s, v20.4s, v4.s[3]\n"
    "fmla v15.4s, v21.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "fmla v16.4s, v20.4s, v5.s[3]\n"
    "fmla v17.4s, v21.4s, v5.s[3]\n"
    "ldr q5, [x10, #176]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v6.4s, v18.4s, v0.s[0]\n"
    "fmla v7.4s, v19.4s, v0.s[0]\n"
    "fmla v8.4s, v18.4s, v1.s[0]\n"
    "fmla v9.4s, v19.4s, v1.s[0]\n"
    "fmla v10.4s, v18.4s, v2.s[0]\n"
    "fmla v11.4s, v19.4s, v2.s[0]\n"
    "add x10, x10, %[inStr]\n"
    "add x8, x8, %[inStr]\n"
    "fmla v12.4s, v18.4s, v3.s[0]\n"
    "fmla v13.4s, v19.4s, v3.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v14.4s, v18.4s, v4.s[0]\n"
    "fmla v15.4s, v19.4s, v4.s[0]\n"
    "fmla v16.4s, v18.4s, v5.s[0]\n"
    "fmla v17.4s, v19.4s, v5.s[0]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v6.4s, v20.4s, v0.s[1]\n"
    "fmla v7.4s, v21.4s, v0.s[1]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "fmla v8.4s, v20.4s, v1.s[1]\n"
    "fmla v9.4s, v21.4s, v1.s[1]\n"
    "fmla v10.4s, v20.4s, v2.s[1]\n"
    "fmla v11.4s, v21.4s, v2.s[1]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "fmla v12.4s, v20.4s, v3.s[1]\n"
    "fmla v13.4s, v21.4s, v3.s[1]\n"
    "fmla v14.4s, v20.4s, v4.s[1]\n"
    "fmla v15.4s, v21.4s, v4.s[1]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "fmla v16.4s, v20.4s, v5.s[1]\n"
    "fmla v17.4s, v21.4s, v5.s[1]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v6.4s, v18.4s, v0.s[2]\n"
    "fmla v7.4s, v19.4s, v0.s[2]\n"
    "fmla v8.4s, v18.4s, v1.s[2]\n"
    "fmla v9.4s, v19.4s, v1.s[2]\n"
    "fmla v10.4s, v18.4s, v2.s[2]\n"
    "fmla v11.4s, v19.4s, v2.s[2]\n"
    "fmla v12.4s, v18.4s, v3.s[2]\n"
    "fmla v13.4s, v19.4s, v3.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v14.4s, v18.4s, v4.s[2]\n"
    "fmla v15.4s, v19.4s, v4.s[2]\n"
    "fmla v16.4s, v18.4s, v5.s[2]\n"
    "fmla v17.4s, v19.4s, v5.s[2]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v6.4s, v20.4s, v0.s[3]\n"
    "fmla v7.4s, v21.4s, v0.s[3]\n"
    "ldr q0, [x10, #0]\n"
    "fmla v8.4s, v20.4s, v1.s[3]\n"
    "fmla v9.4s, v21.4s, v1.s[3]\n"
    "ldr q1, [x10, #32]\n"
    "fmla v10.4s, v20.4s, v2.s[3]\n"
    "fmla v11.4s, v21.4s, v2.s[3]\n"
    "ldr q2, [x10, #64]\n"
    "fmla v12.4s, v20.4s, v3.s[3]\n"
    "fmla v13.4s, v21.4s, v3.s[3]\n"
    "ldr q3, [x10, #96]\n"
    "fmla v14.4s, v20.4s, v4.s[3]\n"
    "fmla v15.4s, v21.4s, v4.s[3]\n"
    "ldr q4, [x10, #128]\n"
    "fmla v16.4s, v20.4s, v5.s[3]\n"
    "fmla v17.4s, v21.4s, v5.s[3]\n"
    "ldr q5, [x10, #160]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
    "b.ne LOOP_START%=\n"
// Remaining 8 channels
// K index 0
// Filter width idx 0
    "fmla v6.4s, v18.4s, v0.s[0]\n"
    "fmla v7.4s, v19.4s, v0.s[0]\n"
    "fmla v8.4s, v18.4s, v1.s[0]\n"
    "fmla v9.4s, v19.4s, v1.s[0]\n"
    "fmla v10.4s, v18.4s, v2.s[0]\n"
    "fmla v11.4s, v19.4s, v2.s[0]\n"
    "fmla v12.4s, v18.4s, v3.s[0]\n"
    "fmla v13.4s, v19.4s, v3.s[0]\n"
    "fmla v14.4s, v18.4s, v4.s[0]\n"
    "fmla v15.4s, v19.4s, v4.s[0]\n"
    "fmla v16.4s, v18.4s, v5.s[0]\n"
    "fmla v17.4s, v19.4s, v5.s[0]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v6.4s, v20.4s, v0.s[1]\n"
    "fmla v7.4s, v21.4s, v0.s[1]\n"
    "fmla v8.4s, v20.4s, v1.s[1]\n"
    "fmla v9.4s, v21.4s, v1.s[1]\n"
    "fmla v10.4s, v20.4s, v2.s[1]\n"
    "fmla v11.4s, v21.4s, v2.s[1]\n"
    "fmla v12.4s, v20.4s, v3.s[1]\n"
    "fmla v13.4s, v21.4s, v3.s[1]\n"
    "fmla v14.4s, v20.4s, v4.s[1]\n"
    "fmla v15.4s, v21.4s, v4.s[1]\n"
    "fmla v16.4s, v20.4s, v5.s[1]\n"
    "fmla v17.4s, v21.4s, v5.s[1]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v6.4s, v18.4s, v0.s[2]\n"
    "fmla v7.4s, v19.4s, v0.s[2]\n"
    "fmla v8.4s, v18.4s, v1.s[2]\n"
    "fmla v9.4s, v19.4s, v1.s[2]\n"
    "fmla v10.4s, v18.4s, v2.s[2]\n"
    "fmla v11.4s, v19.4s, v2.s[2]\n"
    "fmla v12.4s, v18.4s, v3.s[2]\n"
    "fmla v13.4s, v19.4s, v3.s[2]\n"
    "fmla v14.4s, v18.4s, v4.s[2]\n"
    "fmla v15.4s, v19.4s, v4.s[2]\n"
    "fmla v16.4s, v18.4s, v5.s[2]\n"
    "fmla v17.4s, v19.4s, v5.s[2]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v6.4s, v20.4s, v0.s[3]\n"
    "fmla v7.4s, v21.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v8.4s, v20.4s, v1.s[3]\n"
    "fmla v9.4s, v21.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v10.4s, v20.4s, v2.s[3]\n"
    "fmla v11.4s, v21.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v12.4s, v20.4s, v3.s[3]\n"
    "fmla v13.4s, v21.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v14.4s, v20.4s, v4.s[3]\n"
    "fmla v15.4s, v21.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "fmla v16.4s, v20.4s, v5.s[3]\n"
    "fmla v17.4s, v21.4s, v5.s[3]\n"
    "ldr q5, [x10, #176]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v6.4s, v18.4s, v0.s[0]\n"
    "fmla v7.4s, v19.4s, v0.s[0]\n"
    "fmla v8.4s, v18.4s, v1.s[0]\n"
    "fmla v9.4s, v19.4s, v1.s[0]\n"
    "fmla v10.4s, v18.4s, v2.s[0]\n"
    "fmla v11.4s, v19.4s, v2.s[0]\n"
    "fmla v12.4s, v18.4s, v3.s[0]\n"
    "fmla v13.4s, v19.4s, v3.s[0]\n"
    "fmla v14.4s, v18.4s, v4.s[0]\n"
    "fmla v15.4s, v19.4s, v4.s[0]\n"
    "fmla v16.4s, v18.4s, v5.s[0]\n"
    "fmla v17.4s, v19.4s, v5.s[0]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v6.4s, v20.4s, v0.s[1]\n"
    "fmla v7.4s, v21.4s, v0.s[1]\n"
    "fmla v8.4s, v20.4s, v1.s[1]\n"
    "fmla v9.4s, v21.4s, v1.s[1]\n"
    "fmla v10.4s, v20.4s, v2.s[1]\n"
    "fmla v11.4s, v21.4s, v2.s[1]\n"
    "fmla v12.4s, v20.4s, v3.s[1]\n"
    "fmla v13.4s, v21.4s, v3.s[1]\n"
    "fmla v14.4s, v20.4s, v4.s[1]\n"
    "fmla v15.4s, v21.4s, v4.s[1]\n"
    "fmla v16.4s, v20.4s, v5.s[1]\n"
    "fmla v17.4s, v21.4s, v5.s[1]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v6.4s, v18.4s, v0.s[2]\n"
    "fmla v7.4s, v19.4s, v0.s[2]\n"
    "fmla v8.4s, v18.4s, v1.s[2]\n"
    "fmla v9.4s, v19.4s, v1.s[2]\n"
    "fmla v10.4s, v18.4s, v2.s[2]\n"
    "fmla v11.4s, v19.4s, v2.s[2]\n"
    "fmla v12.4s, v18.4s, v3.s[2]\n"
    "fmla v13.4s, v19.4s, v3.s[2]\n"
    "fmla v14.4s, v18.4s, v4.s[2]\n"
    "fmla v15.4s, v19.4s, v4.s[2]\n"
    "fmla v16.4s, v18.4s, v5.s[2]\n"
    "fmla v17.4s, v19.4s, v5.s[2]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v6.4s, v20.4s, v0.s[3]\n"
    "fmla v7.4s, v21.4s, v0.s[3]\n"
    "fmla v8.4s, v20.4s, v1.s[3]\n"
    "fmla v9.4s, v21.4s, v1.s[3]\n"
    "fmla v10.4s, v20.4s, v2.s[3]\n"
    "fmla v11.4s, v21.4s, v2.s[3]\n"
    "fmla v12.4s, v20.4s, v3.s[3]\n"
    "fmla v13.4s, v21.4s, v3.s[3]\n"
    "fmla v14.4s, v20.4s, v4.s[3]\n"
    "fmla v15.4s, v21.4s, v4.s[3]\n"
    "fmla v16.4s, v20.4s, v5.s[3]\n"
    "fmla v17.4s, v21.4s, v5.s[3]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
    "stp q6, q7, [%[out], #0]\n"
    "stp q8, q9, [%[out], #32]\n"
    "stp q10, q11, [%[out], #64]\n"
    "stp q12, q13, [%[out], #96]\n"
    "stp q14, q15, [%[out], #128]\n"
    "stp q16, q17, [%[out], #160]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [k] "r" (k), [inStr] "r" (inStride*sizeof(float))
    :   "x8", "x9", "x10", "x11",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
    );
    }
    #ifdef __DEBUG_PTMM_OFF
    printf ("Output After Kernel:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf("%6.3f\t", *(outputPtr + 8*5 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif
}
void kernel_1_5_8_1_1_0_0(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0], [1], [2], [3], [4]]
// Input registers required
// [0, 1, 2, 3, 4]
// Duplicate index: 0
// [1, 1, 1, 1, 1]
// Number of Input Regs: 5, Filter Regs: 2 Output Regs: 10
// Total number of registers required: 17
// In  - [0, 1, 2, 3, 4]
// Out - [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Fil - [15, 16, 15, 16]
// Register mapping diagram
//     0  1  2  3  4 
//
//15   5  7  9 11 13 
//16   6  8 10 12 14 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < (k/8); i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*2 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*3 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*4 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 1; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*1*8 + wf*8 + i));
            }
            printf("\n");
        }
    }
    printf ("Output:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif

    float* in = inputPtr + 0*8;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
    "add x8, %[in], %[inStr]\n"
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "prfm pldl1keep, [%[in], #64]\n"
    "prfm pldl1keep, [%[in], #128]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q5, q6, [%[out], #0]\n"
    "ldp q7, q8, [%[out], #32]\n"
    "ldp q9, q10, [%[out], #64]\n"
    "ldp q11, q12, [%[out], #96]\n"
    "ldp q13, q14, [%[out], #128]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #32]\n"
    "ldr q2, [x10, #64]\n"
    "ldr q3, [x10, #96]\n"
    "ldr q4, [x10, #128]\n"

    "sub w9, %w[k], #8\n"
    "LOOP_START%=:\n"
    "subs x9, x9, #8\n"
// K index 0
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[0]\n"
    "fmla v6.4s, v16.4s, v0.s[0]\n"
    "fmla v7.4s, v15.4s, v1.s[0]\n"
    "fmla v8.4s, v16.4s, v1.s[0]\n"
    "fmla v9.4s, v15.4s, v2.s[0]\n"
    "fmla v10.4s, v16.4s, v2.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v11.4s, v15.4s, v3.s[0]\n"
    "fmla v12.4s, v16.4s, v3.s[0]\n"
    "fmla v13.4s, v15.4s, v4.s[0]\n"
    "fmla v14.4s, v16.4s, v4.s[0]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[1]\n"
    "fmla v6.4s, v16.4s, v0.s[1]\n"
    "fmla v7.4s, v15.4s, v1.s[1]\n"
    "fmla v8.4s, v16.4s, v1.s[1]\n"
    "fmla v9.4s, v15.4s, v2.s[1]\n"
    "fmla v10.4s, v16.4s, v2.s[1]\n"
    "fmla v11.4s, v15.4s, v3.s[1]\n"
    "fmla v12.4s, v16.4s, v3.s[1]\n"
    "fmla v13.4s, v15.4s, v4.s[1]\n"
    "fmla v14.4s, v16.4s, v4.s[1]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[2]\n"
    "fmla v6.4s, v16.4s, v0.s[2]\n"
    "fmla v7.4s, v15.4s, v1.s[2]\n"
    "fmla v8.4s, v16.4s, v1.s[2]\n"
    "fmla v9.4s, v15.4s, v2.s[2]\n"
    "fmla v10.4s, v16.4s, v2.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v11.4s, v15.4s, v3.s[2]\n"
    "fmla v12.4s, v16.4s, v3.s[2]\n"
    "fmla v13.4s, v15.4s, v4.s[2]\n"
    "fmla v14.4s, v16.4s, v4.s[2]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[3]\n"
    "fmla v6.4s, v16.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v7.4s, v15.4s, v1.s[3]\n"
    "fmla v8.4s, v16.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v9.4s, v15.4s, v2.s[3]\n"
    "fmla v10.4s, v16.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v11.4s, v15.4s, v3.s[3]\n"
    "fmla v12.4s, v16.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v13.4s, v15.4s, v4.s[3]\n"
    "fmla v14.4s, v16.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[0]\n"
    "fmla v6.4s, v16.4s, v0.s[0]\n"
    "fmla v7.4s, v15.4s, v1.s[0]\n"
    "fmla v8.4s, v16.4s, v1.s[0]\n"
    "add x10, x10, %[inStr]\n"
    "add x8, x8, %[inStr]\n"
    "fmla v9.4s, v15.4s, v2.s[0]\n"
    "fmla v10.4s, v16.4s, v2.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v11.4s, v15.4s, v3.s[0]\n"
    "fmla v12.4s, v16.4s, v3.s[0]\n"
    "fmla v13.4s, v15.4s, v4.s[0]\n"
    "fmla v14.4s, v16.4s, v4.s[0]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[1]\n"
    "fmla v6.4s, v16.4s, v0.s[1]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "fmla v7.4s, v15.4s, v1.s[1]\n"
    "fmla v8.4s, v16.4s, v1.s[1]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "fmla v9.4s, v15.4s, v2.s[1]\n"
    "fmla v10.4s, v16.4s, v2.s[1]\n"
    "prfm pldl1keep, [x8, #128]\n"
    "fmla v11.4s, v15.4s, v3.s[1]\n"
    "fmla v12.4s, v16.4s, v3.s[1]\n"
    "fmla v13.4s, v15.4s, v4.s[1]\n"
    "fmla v14.4s, v16.4s, v4.s[1]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[2]\n"
    "fmla v6.4s, v16.4s, v0.s[2]\n"
    "fmla v7.4s, v15.4s, v1.s[2]\n"
    "fmla v8.4s, v16.4s, v1.s[2]\n"
    "fmla v9.4s, v15.4s, v2.s[2]\n"
    "fmla v10.4s, v16.4s, v2.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v11.4s, v15.4s, v3.s[2]\n"
    "fmla v12.4s, v16.4s, v3.s[2]\n"
    "fmla v13.4s, v15.4s, v4.s[2]\n"
    "fmla v14.4s, v16.4s, v4.s[2]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[3]\n"
    "fmla v6.4s, v16.4s, v0.s[3]\n"
    "ldr q0, [x10, #0]\n"
    "fmla v7.4s, v15.4s, v1.s[3]\n"
    "fmla v8.4s, v16.4s, v1.s[3]\n"
    "ldr q1, [x10, #32]\n"
    "fmla v9.4s, v15.4s, v2.s[3]\n"
    "fmla v10.4s, v16.4s, v2.s[3]\n"
    "ldr q2, [x10, #64]\n"
    "fmla v11.4s, v15.4s, v3.s[3]\n"
    "fmla v12.4s, v16.4s, v3.s[3]\n"
    "ldr q3, [x10, #96]\n"
    "fmla v13.4s, v15.4s, v4.s[3]\n"
    "fmla v14.4s, v16.4s, v4.s[3]\n"
    "ldr q4, [x10, #128]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
    "b.ne LOOP_START%=\n"
// Remaining 8 channels
// K index 0
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[0]\n"
    "fmla v6.4s, v16.4s, v0.s[0]\n"
    "fmla v7.4s, v15.4s, v1.s[0]\n"
    "fmla v8.4s, v16.4s, v1.s[0]\n"
    "fmla v9.4s, v15.4s, v2.s[0]\n"
    "fmla v10.4s, v16.4s, v2.s[0]\n"
    "fmla v11.4s, v15.4s, v3.s[0]\n"
    "fmla v12.4s, v16.4s, v3.s[0]\n"
    "fmla v13.4s, v15.4s, v4.s[0]\n"
    "fmla v14.4s, v16.4s, v4.s[0]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[1]\n"
    "fmla v6.4s, v16.4s, v0.s[1]\n"
    "fmla v7.4s, v15.4s, v1.s[1]\n"
    "fmla v8.4s, v16.4s, v1.s[1]\n"
    "fmla v9.4s, v15.4s, v2.s[1]\n"
    "fmla v10.4s, v16.4s, v2.s[1]\n"
    "fmla v11.4s, v15.4s, v3.s[1]\n"
    "fmla v12.4s, v16.4s, v3.s[1]\n"
    "fmla v13.4s, v15.4s, v4.s[1]\n"
    "fmla v14.4s, v16.4s, v4.s[1]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[2]\n"
    "fmla v6.4s, v16.4s, v0.s[2]\n"
    "fmla v7.4s, v15.4s, v1.s[2]\n"
    "fmla v8.4s, v16.4s, v1.s[2]\n"
    "fmla v9.4s, v15.4s, v2.s[2]\n"
    "fmla v10.4s, v16.4s, v2.s[2]\n"
    "fmla v11.4s, v15.4s, v3.s[2]\n"
    "fmla v12.4s, v16.4s, v3.s[2]\n"
    "fmla v13.4s, v15.4s, v4.s[2]\n"
    "fmla v14.4s, v16.4s, v4.s[2]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[3]\n"
    "fmla v6.4s, v16.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v7.4s, v15.4s, v1.s[3]\n"
    "fmla v8.4s, v16.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v9.4s, v15.4s, v2.s[3]\n"
    "fmla v10.4s, v16.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v11.4s, v15.4s, v3.s[3]\n"
    "fmla v12.4s, v16.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "fmla v13.4s, v15.4s, v4.s[3]\n"
    "fmla v14.4s, v16.4s, v4.s[3]\n"
    "ldr q4, [x10, #144]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[0]\n"
    "fmla v6.4s, v16.4s, v0.s[0]\n"
    "fmla v7.4s, v15.4s, v1.s[0]\n"
    "fmla v8.4s, v16.4s, v1.s[0]\n"
    "fmla v9.4s, v15.4s, v2.s[0]\n"
    "fmla v10.4s, v16.4s, v2.s[0]\n"
    "fmla v11.4s, v15.4s, v3.s[0]\n"
    "fmla v12.4s, v16.4s, v3.s[0]\n"
    "fmla v13.4s, v15.4s, v4.s[0]\n"
    "fmla v14.4s, v16.4s, v4.s[0]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[1]\n"
    "fmla v6.4s, v16.4s, v0.s[1]\n"
    "fmla v7.4s, v15.4s, v1.s[1]\n"
    "fmla v8.4s, v16.4s, v1.s[1]\n"
    "fmla v9.4s, v15.4s, v2.s[1]\n"
    "fmla v10.4s, v16.4s, v2.s[1]\n"
    "fmla v11.4s, v15.4s, v3.s[1]\n"
    "fmla v12.4s, v16.4s, v3.s[1]\n"
    "fmla v13.4s, v15.4s, v4.s[1]\n"
    "fmla v14.4s, v16.4s, v4.s[1]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[2]\n"
    "fmla v6.4s, v16.4s, v0.s[2]\n"
    "fmla v7.4s, v15.4s, v1.s[2]\n"
    "fmla v8.4s, v16.4s, v1.s[2]\n"
    "fmla v9.4s, v15.4s, v2.s[2]\n"
    "fmla v10.4s, v16.4s, v2.s[2]\n"
    "fmla v11.4s, v15.4s, v3.s[2]\n"
    "fmla v12.4s, v16.4s, v3.s[2]\n"
    "fmla v13.4s, v15.4s, v4.s[2]\n"
    "fmla v14.4s, v16.4s, v4.s[2]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v5.4s, v15.4s, v0.s[3]\n"
    "fmla v6.4s, v16.4s, v0.s[3]\n"
    "fmla v7.4s, v15.4s, v1.s[3]\n"
    "fmla v8.4s, v16.4s, v1.s[3]\n"
    "fmla v9.4s, v15.4s, v2.s[3]\n"
    "fmla v10.4s, v16.4s, v2.s[3]\n"
    "fmla v11.4s, v15.4s, v3.s[3]\n"
    "fmla v12.4s, v16.4s, v3.s[3]\n"
    "fmla v13.4s, v15.4s, v4.s[3]\n"
    "fmla v14.4s, v16.4s, v4.s[3]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
    "stp q5, q6, [%[out], #0]\n"
    "stp q7, q8, [%[out], #32]\n"
    "stp q9, q10, [%[out], #64]\n"
    "stp q11, q12, [%[out], #96]\n"
    "stp q13, q14, [%[out], #128]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [k] "r" (k), [inStr] "r" (inStride*sizeof(float))
    :   "x8", "x9", "x10", "x11",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
    );
    }
    #ifdef __DEBUG_PTMM_OFF
    printf ("Output After Kernel:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf("%6.3f\t", *(outputPtr + 8*4 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif
}
void kernel_1_4_8_1_1_0_0(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0], [1], [2], [3]]
// Input registers required
// [0, 1, 2, 3]
// Duplicate index: 0
// [1, 1, 1, 1]
// Number of Input Regs: 4, Filter Regs: 2 Output Regs: 8
// Total number of registers required: 14
// In  - [0, 1, 2, 3]
// Out - [4, 5, 6, 7, 8, 9, 10, 11]
// Fil - [12, 13, 12, 13]
// Register mapping diagram
//     0  1  2  3 
//
//12   4  6  8 10 
//13   5  7  9 11 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < (k/8); i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*2 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*3 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 1; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*1*8 + wf*8 + i));
            }
            printf("\n");
        }
    }
    printf ("Output:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif

    float* in = inputPtr + 0*8;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
    "add x8, %[in], %[inStr]\n"
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "prfm pldl1keep, [%[in], #64]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q4, q5, [%[out], #0]\n"
    "ldp q6, q7, [%[out], #32]\n"
    "ldp q8, q9, [%[out], #64]\n"
    "ldp q10, q11, [%[out], #96]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #32]\n"
    "ldr q2, [x10, #64]\n"
    "ldr q3, [x10, #96]\n"

    "sub w9, %w[k], #8\n"
    "LOOP_START%=:\n"
    "subs x9, x9, #8\n"
// K index 0
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[0]\n"
    "fmla v5.4s, v13.4s, v0.s[0]\n"
    "fmla v6.4s, v12.4s, v1.s[0]\n"
    "fmla v7.4s, v13.4s, v1.s[0]\n"
    "fmla v8.4s, v12.4s, v2.s[0]\n"
    "fmla v9.4s, v13.4s, v2.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v10.4s, v12.4s, v3.s[0]\n"
    "fmla v11.4s, v13.4s, v3.s[0]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[1]\n"
    "fmla v5.4s, v13.4s, v0.s[1]\n"
    "fmla v6.4s, v12.4s, v1.s[1]\n"
    "fmla v7.4s, v13.4s, v1.s[1]\n"
    "fmla v8.4s, v12.4s, v2.s[1]\n"
    "fmla v9.4s, v13.4s, v2.s[1]\n"
    "fmla v10.4s, v12.4s, v3.s[1]\n"
    "fmla v11.4s, v13.4s, v3.s[1]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[2]\n"
    "fmla v5.4s, v13.4s, v0.s[2]\n"
    "fmla v6.4s, v12.4s, v1.s[2]\n"
    "fmla v7.4s, v13.4s, v1.s[2]\n"
    "fmla v8.4s, v12.4s, v2.s[2]\n"
    "fmla v9.4s, v13.4s, v2.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v10.4s, v12.4s, v3.s[2]\n"
    "fmla v11.4s, v13.4s, v3.s[2]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[3]\n"
    "fmla v5.4s, v13.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v6.4s, v12.4s, v1.s[3]\n"
    "fmla v7.4s, v13.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v8.4s, v12.4s, v2.s[3]\n"
    "fmla v9.4s, v13.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v10.4s, v12.4s, v3.s[3]\n"
    "fmla v11.4s, v13.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[0]\n"
    "fmla v5.4s, v13.4s, v0.s[0]\n"
    "fmla v6.4s, v12.4s, v1.s[0]\n"
    "fmla v7.4s, v13.4s, v1.s[0]\n"
    "add x10, x10, %[inStr]\n"
    "add x8, x8, %[inStr]\n"
    "fmla v8.4s, v12.4s, v2.s[0]\n"
    "fmla v9.4s, v13.4s, v2.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v10.4s, v12.4s, v3.s[0]\n"
    "fmla v11.4s, v13.4s, v3.s[0]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[1]\n"
    "fmla v5.4s, v13.4s, v0.s[1]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "fmla v6.4s, v12.4s, v1.s[1]\n"
    "fmla v7.4s, v13.4s, v1.s[1]\n"
    "fmla v8.4s, v12.4s, v2.s[1]\n"
    "fmla v9.4s, v13.4s, v2.s[1]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "fmla v10.4s, v12.4s, v3.s[1]\n"
    "fmla v11.4s, v13.4s, v3.s[1]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[2]\n"
    "fmla v5.4s, v13.4s, v0.s[2]\n"
    "fmla v6.4s, v12.4s, v1.s[2]\n"
    "fmla v7.4s, v13.4s, v1.s[2]\n"
    "fmla v8.4s, v12.4s, v2.s[2]\n"
    "fmla v9.4s, v13.4s, v2.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v10.4s, v12.4s, v3.s[2]\n"
    "fmla v11.4s, v13.4s, v3.s[2]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[3]\n"
    "fmla v5.4s, v13.4s, v0.s[3]\n"
    "ldr q0, [x10, #0]\n"
    "fmla v6.4s, v12.4s, v1.s[3]\n"
    "fmla v7.4s, v13.4s, v1.s[3]\n"
    "ldr q1, [x10, #32]\n"
    "fmla v8.4s, v12.4s, v2.s[3]\n"
    "fmla v9.4s, v13.4s, v2.s[3]\n"
    "ldr q2, [x10, #64]\n"
    "fmla v10.4s, v12.4s, v3.s[3]\n"
    "fmla v11.4s, v13.4s, v3.s[3]\n"
    "ldr q3, [x10, #96]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
    "b.ne LOOP_START%=\n"
// Remaining 8 channels
// K index 0
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[0]\n"
    "fmla v5.4s, v13.4s, v0.s[0]\n"
    "fmla v6.4s, v12.4s, v1.s[0]\n"
    "fmla v7.4s, v13.4s, v1.s[0]\n"
    "fmla v8.4s, v12.4s, v2.s[0]\n"
    "fmla v9.4s, v13.4s, v2.s[0]\n"
    "fmla v10.4s, v12.4s, v3.s[0]\n"
    "fmla v11.4s, v13.4s, v3.s[0]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[1]\n"
    "fmla v5.4s, v13.4s, v0.s[1]\n"
    "fmla v6.4s, v12.4s, v1.s[1]\n"
    "fmla v7.4s, v13.4s, v1.s[1]\n"
    "fmla v8.4s, v12.4s, v2.s[1]\n"
    "fmla v9.4s, v13.4s, v2.s[1]\n"
    "fmla v10.4s, v12.4s, v3.s[1]\n"
    "fmla v11.4s, v13.4s, v3.s[1]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[2]\n"
    "fmla v5.4s, v13.4s, v0.s[2]\n"
    "fmla v6.4s, v12.4s, v1.s[2]\n"
    "fmla v7.4s, v13.4s, v1.s[2]\n"
    "fmla v8.4s, v12.4s, v2.s[2]\n"
    "fmla v9.4s, v13.4s, v2.s[2]\n"
    "fmla v10.4s, v12.4s, v3.s[2]\n"
    "fmla v11.4s, v13.4s, v3.s[2]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[3]\n"
    "fmla v5.4s, v13.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v6.4s, v12.4s, v1.s[3]\n"
    "fmla v7.4s, v13.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v8.4s, v12.4s, v2.s[3]\n"
    "fmla v9.4s, v13.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "fmla v10.4s, v12.4s, v3.s[3]\n"
    "fmla v11.4s, v13.4s, v3.s[3]\n"
    "ldr q3, [x10, #112]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[0]\n"
    "fmla v5.4s, v13.4s, v0.s[0]\n"
    "fmla v6.4s, v12.4s, v1.s[0]\n"
    "fmla v7.4s, v13.4s, v1.s[0]\n"
    "fmla v8.4s, v12.4s, v2.s[0]\n"
    "fmla v9.4s, v13.4s, v2.s[0]\n"
    "fmla v10.4s, v12.4s, v3.s[0]\n"
    "fmla v11.4s, v13.4s, v3.s[0]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[1]\n"
    "fmla v5.4s, v13.4s, v0.s[1]\n"
    "fmla v6.4s, v12.4s, v1.s[1]\n"
    "fmla v7.4s, v13.4s, v1.s[1]\n"
    "fmla v8.4s, v12.4s, v2.s[1]\n"
    "fmla v9.4s, v13.4s, v2.s[1]\n"
    "fmla v10.4s, v12.4s, v3.s[1]\n"
    "fmla v11.4s, v13.4s, v3.s[1]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[2]\n"
    "fmla v5.4s, v13.4s, v0.s[2]\n"
    "fmla v6.4s, v12.4s, v1.s[2]\n"
    "fmla v7.4s, v13.4s, v1.s[2]\n"
    "fmla v8.4s, v12.4s, v2.s[2]\n"
    "fmla v9.4s, v13.4s, v2.s[2]\n"
    "fmla v10.4s, v12.4s, v3.s[2]\n"
    "fmla v11.4s, v13.4s, v3.s[2]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v4.4s, v12.4s, v0.s[3]\n"
    "fmla v5.4s, v13.4s, v0.s[3]\n"
    "fmla v6.4s, v12.4s, v1.s[3]\n"
    "fmla v7.4s, v13.4s, v1.s[3]\n"
    "fmla v8.4s, v12.4s, v2.s[3]\n"
    "fmla v9.4s, v13.4s, v2.s[3]\n"
    "fmla v10.4s, v12.4s, v3.s[3]\n"
    "fmla v11.4s, v13.4s, v3.s[3]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
    "stp q4, q5, [%[out], #0]\n"
    "stp q6, q7, [%[out], #32]\n"
    "stp q8, q9, [%[out], #64]\n"
    "stp q10, q11, [%[out], #96]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [k] "r" (k), [inStr] "r" (inStride*sizeof(float))
    :   "x8", "x9", "x10", "x11",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
    );
    }
    #ifdef __DEBUG_PTMM_OFF
    printf ("Output After Kernel:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf("%6.3f\t", *(outputPtr + 8*3 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif
}
void kernel_1_3_8_1_1_0_0(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0], [1], [2]]
// Input registers required
// [0, 1, 2]
// Duplicate index: 0
// [1, 1, 1]
// Number of Input Regs: 3, Filter Regs: 2 Output Regs: 6
// Total number of registers required: 11
// In  - [0, 1, 2]
// Out - [3, 4, 5, 6, 7, 8]
// Fil - [9, 10, 9, 10]
// Register mapping diagram
//     0  1  2 
//
// 9   3  5  7 
//10   4  6  8 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < (k/8); i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*2 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 1; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*1*8 + wf*8 + i));
            }
            printf("\n");
        }
    }
    printf ("Output:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif

    float* in = inputPtr + 0*8;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
    "add x8, %[in], %[inStr]\n"
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "prfm pldl1keep, [%[in], #64]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q3, q4, [%[out], #0]\n"
    "ldp q5, q6, [%[out], #32]\n"
    "ldp q7, q8, [%[out], #64]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #32]\n"
    "ldr q2, [x10, #64]\n"

    "sub w9, %w[k], #8\n"
    "LOOP_START%=:\n"
    "subs x9, x9, #8\n"
// K index 0
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[0]\n"
    "fmla v4.4s, v10.4s, v0.s[0]\n"
    "fmla v5.4s, v9.4s, v1.s[0]\n"
    "fmla v6.4s, v10.4s, v1.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v7.4s, v9.4s, v2.s[0]\n"
    "fmla v8.4s, v10.4s, v2.s[0]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[1]\n"
    "fmla v4.4s, v10.4s, v0.s[1]\n"
    "fmla v5.4s, v9.4s, v1.s[1]\n"
    "fmla v6.4s, v10.4s, v1.s[1]\n"
    "fmla v7.4s, v9.4s, v2.s[1]\n"
    "fmla v8.4s, v10.4s, v2.s[1]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[2]\n"
    "fmla v4.4s, v10.4s, v0.s[2]\n"
    "fmla v5.4s, v9.4s, v1.s[2]\n"
    "fmla v6.4s, v10.4s, v1.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v7.4s, v9.4s, v2.s[2]\n"
    "fmla v8.4s, v10.4s, v2.s[2]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[3]\n"
    "fmla v4.4s, v10.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v5.4s, v9.4s, v1.s[3]\n"
    "fmla v6.4s, v10.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v7.4s, v9.4s, v2.s[3]\n"
    "fmla v8.4s, v10.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[0]\n"
    "fmla v4.4s, v10.4s, v0.s[0]\n"
    "fmla v5.4s, v9.4s, v1.s[0]\n"
    "fmla v6.4s, v10.4s, v1.s[0]\n"
    "add x10, x10, %[inStr]\n"
    "add x8, x8, %[inStr]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v7.4s, v9.4s, v2.s[0]\n"
    "fmla v8.4s, v10.4s, v2.s[0]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[1]\n"
    "fmla v4.4s, v10.4s, v0.s[1]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "fmla v5.4s, v9.4s, v1.s[1]\n"
    "fmla v6.4s, v10.4s, v1.s[1]\n"
    "prfm pldl1keep, [x8, #64]\n"
    "fmla v7.4s, v9.4s, v2.s[1]\n"
    "fmla v8.4s, v10.4s, v2.s[1]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[2]\n"
    "fmla v4.4s, v10.4s, v0.s[2]\n"
    "fmla v5.4s, v9.4s, v1.s[2]\n"
    "fmla v6.4s, v10.4s, v1.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "fmla v7.4s, v9.4s, v2.s[2]\n"
    "fmla v8.4s, v10.4s, v2.s[2]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[3]\n"
    "fmla v4.4s, v10.4s, v0.s[3]\n"
    "ldr q0, [x10, #0]\n"
    "fmla v5.4s, v9.4s, v1.s[3]\n"
    "fmla v6.4s, v10.4s, v1.s[3]\n"
    "ldr q1, [x10, #32]\n"
    "fmla v7.4s, v9.4s, v2.s[3]\n"
    "fmla v8.4s, v10.4s, v2.s[3]\n"
    "ldr q2, [x10, #64]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
    "b.ne LOOP_START%=\n"
// Remaining 8 channels
// K index 0
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[0]\n"
    "fmla v4.4s, v10.4s, v0.s[0]\n"
    "fmla v5.4s, v9.4s, v1.s[0]\n"
    "fmla v6.4s, v10.4s, v1.s[0]\n"
    "fmla v7.4s, v9.4s, v2.s[0]\n"
    "fmla v8.4s, v10.4s, v2.s[0]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[1]\n"
    "fmla v4.4s, v10.4s, v0.s[1]\n"
    "fmla v5.4s, v9.4s, v1.s[1]\n"
    "fmla v6.4s, v10.4s, v1.s[1]\n"
    "fmla v7.4s, v9.4s, v2.s[1]\n"
    "fmla v8.4s, v10.4s, v2.s[1]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[2]\n"
    "fmla v4.4s, v10.4s, v0.s[2]\n"
    "fmla v5.4s, v9.4s, v1.s[2]\n"
    "fmla v6.4s, v10.4s, v1.s[2]\n"
    "fmla v7.4s, v9.4s, v2.s[2]\n"
    "fmla v8.4s, v10.4s, v2.s[2]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[3]\n"
    "fmla v4.4s, v10.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v5.4s, v9.4s, v1.s[3]\n"
    "fmla v6.4s, v10.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "fmla v7.4s, v9.4s, v2.s[3]\n"
    "fmla v8.4s, v10.4s, v2.s[3]\n"
    "ldr q2, [x10, #80]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[0]\n"
    "fmla v4.4s, v10.4s, v0.s[0]\n"
    "fmla v5.4s, v9.4s, v1.s[0]\n"
    "fmla v6.4s, v10.4s, v1.s[0]\n"
    "fmla v7.4s, v9.4s, v2.s[0]\n"
    "fmla v8.4s, v10.4s, v2.s[0]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[1]\n"
    "fmla v4.4s, v10.4s, v0.s[1]\n"
    "fmla v5.4s, v9.4s, v1.s[1]\n"
    "fmla v6.4s, v10.4s, v1.s[1]\n"
    "fmla v7.4s, v9.4s, v2.s[1]\n"
    "fmla v8.4s, v10.4s, v2.s[1]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[2]\n"
    "fmla v4.4s, v10.4s, v0.s[2]\n"
    "fmla v5.4s, v9.4s, v1.s[2]\n"
    "fmla v6.4s, v10.4s, v1.s[2]\n"
    "fmla v7.4s, v9.4s, v2.s[2]\n"
    "fmla v8.4s, v10.4s, v2.s[2]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v3.4s, v9.4s, v0.s[3]\n"
    "fmla v4.4s, v10.4s, v0.s[3]\n"
    "fmla v5.4s, v9.4s, v1.s[3]\n"
    "fmla v6.4s, v10.4s, v1.s[3]\n"
    "fmla v7.4s, v9.4s, v2.s[3]\n"
    "fmla v8.4s, v10.4s, v2.s[3]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
    "stp q3, q4, [%[out], #0]\n"
    "stp q5, q6, [%[out], #32]\n"
    "stp q7, q8, [%[out], #64]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [k] "r" (k), [inStr] "r" (inStride*sizeof(float))
    :   "x8", "x9", "x10", "x11",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
    );
    }
    #ifdef __DEBUG_PTMM_OFF
    printf ("Output After Kernel:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf("%6.3f\t", *(outputPtr + 8*2 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif
}
void kernel_1_2_8_1_1_0_0(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0], [1]]
// Input registers required
// [0, 1]
// Duplicate index: 0
// [1, 1]
// Number of Input Regs: 2, Filter Regs: 2 Output Regs: 4
// Total number of registers required: 8
// In  - [0, 1]
// Out - [2, 3, 4, 5]
// Fil - [6, 7, 6, 7]
// Register mapping diagram
//     0  1 
//
// 6   2  4 
// 7   3  5 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < (k/8); i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*1 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 1; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*1*8 + wf*8 + i));
            }
            printf("\n");
        }
    }
    printf ("Output:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif

    float* in = inputPtr + 0*8;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
    "add x8, %[in], %[inStr]\n"
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q2, q3, [%[out], #0]\n"
    "ldp q4, q5, [%[out], #32]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #32]\n"

    "sub w9, %w[k], #8\n"
    "LOOP_START%=:\n"
    "subs x9, x9, #8\n"
// K index 0
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[0]\n"
    "fmla v3.4s, v7.4s, v0.s[0]\n"
    "fmla v4.4s, v6.4s, v1.s[0]\n"
    "fmla v5.4s, v7.4s, v1.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[1]\n"
    "fmla v3.4s, v7.4s, v0.s[1]\n"
    "fmla v4.4s, v6.4s, v1.s[1]\n"
    "fmla v5.4s, v7.4s, v1.s[1]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[2]\n"
    "fmla v3.4s, v7.4s, v0.s[2]\n"
    "fmla v4.4s, v6.4s, v1.s[2]\n"
    "fmla v5.4s, v7.4s, v1.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[3]\n"
    "fmla v3.4s, v7.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v4.4s, v6.4s, v1.s[3]\n"
    "fmla v5.4s, v7.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[0]\n"
    "fmla v3.4s, v7.4s, v0.s[0]\n"
    "add x10, x10, %[inStr]\n"
    "add x8, x8, %[inStr]\n"
    "fmla v4.4s, v6.4s, v1.s[0]\n"
    "fmla v5.4s, v7.4s, v1.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[1]\n"
    "fmla v3.4s, v7.4s, v0.s[1]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "fmla v4.4s, v6.4s, v1.s[1]\n"
    "fmla v5.4s, v7.4s, v1.s[1]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[2]\n"
    "fmla v3.4s, v7.4s, v0.s[2]\n"
    "fmla v4.4s, v6.4s, v1.s[2]\n"
    "fmla v5.4s, v7.4s, v1.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[3]\n"
    "fmla v3.4s, v7.4s, v0.s[3]\n"
    "ldr q0, [x10, #0]\n"
    "fmla v4.4s, v6.4s, v1.s[3]\n"
    "fmla v5.4s, v7.4s, v1.s[3]\n"
    "ldr q1, [x10, #32]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
    "b.ne LOOP_START%=\n"
// Remaining 8 channels
// K index 0
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[0]\n"
    "fmla v3.4s, v7.4s, v0.s[0]\n"
    "fmla v4.4s, v6.4s, v1.s[0]\n"
    "fmla v5.4s, v7.4s, v1.s[0]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[1]\n"
    "fmla v3.4s, v7.4s, v0.s[1]\n"
    "fmla v4.4s, v6.4s, v1.s[1]\n"
    "fmla v5.4s, v7.4s, v1.s[1]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[2]\n"
    "fmla v3.4s, v7.4s, v0.s[2]\n"
    "fmla v4.4s, v6.4s, v1.s[2]\n"
    "fmla v5.4s, v7.4s, v1.s[2]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[3]\n"
    "fmla v3.4s, v7.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "fmla v4.4s, v6.4s, v1.s[3]\n"
    "fmla v5.4s, v7.4s, v1.s[3]\n"
    "ldr q1, [x10, #48]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[0]\n"
    "fmla v3.4s, v7.4s, v0.s[0]\n"
    "fmla v4.4s, v6.4s, v1.s[0]\n"
    "fmla v5.4s, v7.4s, v1.s[0]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[1]\n"
    "fmla v3.4s, v7.4s, v0.s[1]\n"
    "fmla v4.4s, v6.4s, v1.s[1]\n"
    "fmla v5.4s, v7.4s, v1.s[1]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[2]\n"
    "fmla v3.4s, v7.4s, v0.s[2]\n"
    "fmla v4.4s, v6.4s, v1.s[2]\n"
    "fmla v5.4s, v7.4s, v1.s[2]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v2.4s, v6.4s, v0.s[3]\n"
    "fmla v3.4s, v7.4s, v0.s[3]\n"
    "fmla v4.4s, v6.4s, v1.s[3]\n"
    "fmla v5.4s, v7.4s, v1.s[3]\n"
    "ld1 {v6.4s - v7.4s}, [x11], #32\n"
    "stp q2, q3, [%[out], #0]\n"
    "stp q4, q5, [%[out], #32]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [k] "r" (k), [inStr] "r" (inStride*sizeof(float))
    :   "x8", "x9", "x10", "x11",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
    );
    }
    #ifdef __DEBUG_PTMM_OFF
    printf ("Output After Kernel:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf("%6.3f\t", *(outputPtr + 8*1 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif
}
void kernel_1_1_8_1_1_0_0(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0]]
// Input registers required
// [0]
// Duplicate index: 0
// [1]
// Number of Input Regs: 1, Filter Regs: 2 Output Regs: 2
// Total number of registers required: 5
// In  - [0]
// Out - [1, 2]
// Fil - [3, 4, 3, 4]
// Register mapping diagram
//     0 
//
// 3   1 
// 4   2 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < (k/8); i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 8*0 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 1; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*1*8 + wf*8 + i));
            }
            printf("\n");
        }
    }
    printf ("Output:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif

    float* in = inputPtr + 0*8;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
    "add x8, %[in], %[inStr]\n"
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q1, q2, [%[out], #0]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"

    "sub w9, %w[k], #8\n"
    "LOOP_START%=:\n"
    "subs x9, x9, #8\n"
// K index 0
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[0]\n"
    "fmla v2.4s, v4.4s, v0.s[0]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[1]\n"
    "fmla v2.4s, v4.4s, v0.s[1]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[2]\n"
    "fmla v2.4s, v4.4s, v0.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[3]\n"
    "fmla v2.4s, v4.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[0]\n"
    "fmla v2.4s, v4.4s, v0.s[0]\n"
    "add x10, x10, %[inStr]\n"
    "add x8, x8, %[inStr]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[1]\n"
    "fmla v2.4s, v4.4s, v0.s[1]\n"
    "prfm pldl1keep, [x8, #0]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[2]\n"
    "fmla v2.4s, v4.4s, v0.s[2]\n"
    "prfm pldl1keep, [x11, #256]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[3]\n"
    "fmla v2.4s, v4.4s, v0.s[3]\n"
    "ldr q0, [x10, #0]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
    "b.ne LOOP_START%=\n"
// Remaining 8 channels
// K index 0
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[0]\n"
    "fmla v2.4s, v4.4s, v0.s[0]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[1]\n"
    "fmla v2.4s, v4.4s, v0.s[1]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[2]\n"
    "fmla v2.4s, v4.4s, v0.s[2]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 3
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[3]\n"
    "fmla v2.4s, v4.4s, v0.s[3]\n"
    "ldr q0, [x10, #16]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 4
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[0]\n"
    "fmla v2.4s, v4.4s, v0.s[0]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 5
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[1]\n"
    "fmla v2.4s, v4.4s, v0.s[1]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 6
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[2]\n"
    "fmla v2.4s, v4.4s, v0.s[2]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
// K index 7
// Filter width idx 0
    "fmla v1.4s, v3.4s, v0.s[3]\n"
    "fmla v2.4s, v4.4s, v0.s[3]\n"
    "ld1 {v3.4s - v4.4s}, [x11], #32\n"
    "stp q1, q2, [%[out], #0]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [k] "r" (k), [inStr] "r" (inStride*sizeof(float))
    :   "x8", "x9", "x10", "x11",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
    );
    }
    #ifdef __DEBUG_PTMM_OFF
    printf ("Output After Kernel:\n");
    for (int i = 0; i < 8; i++)
    {
        printf("Row %d:\t", i);
        printf("%6.3f\t", *(outputPtr + 8*0 + i));
        printf ("\n");
    }
    printf ("\n");
    #endif
}
