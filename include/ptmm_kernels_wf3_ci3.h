void kernel_3_7_8_1_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Duplicate index: 12
// [1, 2, 3, 3, 3, 3, 3, 2, 1]
// Number of Input Regs: 9, Filter Regs: 4 Output Regs: 14
// Total number of registers required: 27
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Out - [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
// Fil - [23, 24, 25, 26, 23, 24, 25, 26, 23, 24]
// Register mapping diagram
//           0  1  2  3  4  5  6  7  8 
//
//23 25 23   9 11 13 15 17 19 21 
//24 26 24  10 12 14 16 18 20 22 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + 3*7 + j));
            printf("%6.3f\t", *(inputPtr + 3*8 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q9, q10, [%[out], #0]\n"
    "ldp q11, q12, [%[out], #32]\n"
    "ldp q13, q14, [%[out], #64]\n"
    "ldp q15, q16, [%[out], #96]\n"
    "ldp q17, q18, [%[out], #128]\n"
    "ldp q19, q20, [%[out], #160]\n"
    "ldp q21, q22, [%[out], #192]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"
    "ldr q5, [x10, #60]\n"
    "ldr q6, [x10, #72]\n"
    "ldr q7, [x10, #84]\n"
    "ldr q8, [x10, #96]\n"

// K index 0
// Filter width idx 0
    "fmla v9.4s, v23.4s, v0.s[0]\n"
    "fmla v10.4s, v24.4s, v0.s[0]\n"
    "fmla v11.4s, v23.4s, v1.s[0]\n"
    "fmla v12.4s, v24.4s, v1.s[0]\n"
    "fmla v13.4s, v23.4s, v2.s[0]\n"
    "fmla v14.4s, v24.4s, v2.s[0]\n"
    "fmla v15.4s, v23.4s, v3.s[0]\n"
    "fmla v16.4s, v24.4s, v3.s[0]\n"
    "fmla v17.4s, v23.4s, v4.s[0]\n"
    "fmla v18.4s, v24.4s, v4.s[0]\n"
    "fmla v19.4s, v23.4s, v5.s[0]\n"
    "fmla v20.4s, v24.4s, v5.s[0]\n"
    "fmla v21.4s, v23.4s, v6.s[0]\n"
    "fmla v22.4s, v24.4s, v6.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v9.4s, v25.4s, v1.s[0]\n"
    "fmla v10.4s, v26.4s, v1.s[0]\n"
    "fmla v11.4s, v25.4s, v2.s[0]\n"
    "fmla v12.4s, v26.4s, v2.s[0]\n"
    "fmla v13.4s, v25.4s, v3.s[0]\n"
    "fmla v14.4s, v26.4s, v3.s[0]\n"
    "fmla v15.4s, v25.4s, v4.s[0]\n"
    "fmla v16.4s, v26.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v5.s[0]\n"
    "fmla v18.4s, v26.4s, v5.s[0]\n"
    "fmla v19.4s, v25.4s, v6.s[0]\n"
    "fmla v20.4s, v26.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v7.s[0]\n"
    "fmla v22.4s, v26.4s, v7.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v9.4s, v23.4s, v2.s[0]\n"
    "fmla v10.4s, v24.4s, v2.s[0]\n"
    "fmla v11.4s, v23.4s, v3.s[0]\n"
    "fmla v12.4s, v24.4s, v3.s[0]\n"
    "fmla v13.4s, v23.4s, v4.s[0]\n"
    "fmla v14.4s, v24.4s, v4.s[0]\n"
    "fmla v15.4s, v23.4s, v5.s[0]\n"
    "fmla v16.4s, v24.4s, v5.s[0]\n"
    "fmla v17.4s, v23.4s, v6.s[0]\n"
    "fmla v18.4s, v24.4s, v6.s[0]\n"
    "fmla v19.4s, v23.4s, v7.s[0]\n"
    "fmla v20.4s, v24.4s, v7.s[0]\n"
    "fmla v21.4s, v23.4s, v8.s[0]\n"
    "fmla v22.4s, v24.4s, v8.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v9.4s, v25.4s, v0.s[1]\n"
    "fmla v10.4s, v26.4s, v0.s[1]\n"
    "fmla v11.4s, v25.4s, v1.s[1]\n"
    "fmla v12.4s, v26.4s, v1.s[1]\n"
    "fmla v13.4s, v25.4s, v2.s[1]\n"
    "fmla v14.4s, v26.4s, v2.s[1]\n"
    "fmla v15.4s, v25.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v3.s[1]\n"
    "fmla v17.4s, v25.4s, v4.s[1]\n"
    "fmla v18.4s, v26.4s, v4.s[1]\n"
    "fmla v19.4s, v25.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v5.s[1]\n"
    "fmla v21.4s, v25.4s, v6.s[1]\n"
    "fmla v22.4s, v26.4s, v6.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 1
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
    "fmla v21.4s, v23.4s, v7.s[1]\n"
    "fmla v22.4s, v24.4s, v7.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v9.4s, v25.4s, v2.s[1]\n"
    "fmla v10.4s, v26.4s, v2.s[1]\n"
    "fmla v11.4s, v25.4s, v3.s[1]\n"
    "fmla v12.4s, v26.4s, v3.s[1]\n"
    "fmla v13.4s, v25.4s, v4.s[1]\n"
    "fmla v14.4s, v26.4s, v4.s[1]\n"
    "fmla v15.4s, v25.4s, v5.s[1]\n"
    "fmla v16.4s, v26.4s, v5.s[1]\n"
    "fmla v17.4s, v25.4s, v6.s[1]\n"
    "fmla v18.4s, v26.4s, v6.s[1]\n"
    "fmla v19.4s, v25.4s, v7.s[1]\n"
    "fmla v20.4s, v26.4s, v7.s[1]\n"
    "fmla v21.4s, v25.4s, v8.s[1]\n"
    "fmla v22.4s, v26.4s, v8.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v9.4s, v23.4s, v0.s[2]\n"
    "fmla v10.4s, v24.4s, v0.s[2]\n"
    "fmla v11.4s, v23.4s, v1.s[2]\n"
    "fmla v12.4s, v24.4s, v1.s[2]\n"
    "fmla v13.4s, v23.4s, v2.s[2]\n"
    "fmla v14.4s, v24.4s, v2.s[2]\n"
    "fmla v15.4s, v23.4s, v3.s[2]\n"
    "fmla v16.4s, v24.4s, v3.s[2]\n"
    "fmla v17.4s, v23.4s, v4.s[2]\n"
    "fmla v18.4s, v24.4s, v4.s[2]\n"
    "fmla v19.4s, v23.4s, v5.s[2]\n"
    "fmla v20.4s, v24.4s, v5.s[2]\n"
    "fmla v21.4s, v23.4s, v6.s[2]\n"
    "fmla v22.4s, v24.4s, v6.s[2]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v9.4s, v25.4s, v1.s[2]\n"
    "fmla v10.4s, v26.4s, v1.s[2]\n"
    "fmla v11.4s, v25.4s, v2.s[2]\n"
    "fmla v12.4s, v26.4s, v2.s[2]\n"
    "fmla v13.4s, v25.4s, v3.s[2]\n"
    "fmla v14.4s, v26.4s, v3.s[2]\n"
    "fmla v15.4s, v25.4s, v4.s[2]\n"
    "fmla v16.4s, v26.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v5.s[2]\n"
    "fmla v18.4s, v26.4s, v5.s[2]\n"
    "fmla v19.4s, v25.4s, v6.s[2]\n"
    "fmla v20.4s, v26.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v7.s[2]\n"
    "fmla v22.4s, v26.4s, v7.s[2]\n"
// Filter width idx 2
    "fmla v9.4s, v23.4s, v2.s[2]\n"
    "fmla v10.4s, v24.4s, v2.s[2]\n"
    "fmla v11.4s, v23.4s, v3.s[2]\n"
    "fmla v12.4s, v24.4s, v3.s[2]\n"
    "fmla v13.4s, v23.4s, v4.s[2]\n"
    "fmla v14.4s, v24.4s, v4.s[2]\n"
    "fmla v15.4s, v23.4s, v5.s[2]\n"
    "fmla v16.4s, v24.4s, v5.s[2]\n"
    "fmla v17.4s, v23.4s, v6.s[2]\n"
    "fmla v18.4s, v24.4s, v6.s[2]\n"
    "fmla v19.4s, v23.4s, v7.s[2]\n"
    "fmla v20.4s, v24.4s, v7.s[2]\n"
    "fmla v21.4s, v23.4s, v8.s[2]\n"
    "fmla v22.4s, v24.4s, v8.s[2]\n"
    "stp q9, q10, [%[out], #0]\n"
    "stp q11, q12, [%[out], #32]\n"
    "stp q13, q14, [%[out], #64]\n"
    "stp q15, q16, [%[out], #96]\n"
    "stp q17, q18, [%[out], #128]\n"
    "stp q19, q20, [%[out], #160]\n"
    "stp q21, q22, [%[out], #192]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_7_8_1_1_1_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Duplicate index: 12
// [1, 2, 3, 3, 3, 3, 3, 2, 1]
// Number of Input Regs: 9, Filter Regs: 4 Output Regs: 14
// Total number of registers required: 27
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Out - [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
// Fil - [23, 24, 25, 26, 23, 24, 25, 26, 23, 24]
// Register mapping diagram
//           0  1  2  3  4  5  6  7  8 
//
//23 25 23   9 11 13 15 17 19 21 
//24 26 24  10 12 14 16 18 20 22 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + 3*7 + j));
            printf("%6.3f\t", *(inputPtr + 3*8 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 1*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q9, q10, [%[out], #0]\n"
    "ldp q11, q12, [%[out], #32]\n"
    "ldp q13, q14, [%[out], #64]\n"
    "ldp q15, q16, [%[out], #96]\n"
    "ldp q17, q18, [%[out], #128]\n"
    "ldp q19, q20, [%[out], #160]\n"
    "ldp q21, q22, [%[out], #192]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Load Input
    "ldr q1, [x10, #0]\n"
    "ldr q2, [x10, #12]\n"
    "ldr q3, [x10, #24]\n"
    "ldr q4, [x10, #36]\n"
    "ldr q5, [x10, #48]\n"
    "ldr q6, [x10, #60]\n"
    "ldr q7, [x10, #72]\n"
    "ldr q8, [x10, #84]\n"

// K index 0
// Filter width idx 0
    "fmla v11.4s, v23.4s, v1.s[0]\n"
    "fmla v12.4s, v24.4s, v1.s[0]\n"
    "fmla v13.4s, v23.4s, v2.s[0]\n"
    "fmla v14.4s, v24.4s, v2.s[0]\n"
    "fmla v15.4s, v23.4s, v3.s[0]\n"
    "fmla v16.4s, v24.4s, v3.s[0]\n"
    "fmla v17.4s, v23.4s, v4.s[0]\n"
    "fmla v18.4s, v24.4s, v4.s[0]\n"
    "fmla v19.4s, v23.4s, v5.s[0]\n"
    "fmla v20.4s, v24.4s, v5.s[0]\n"
    "fmla v21.4s, v23.4s, v6.s[0]\n"
    "fmla v22.4s, v24.4s, v6.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v9.4s, v25.4s, v1.s[0]\n"
    "fmla v10.4s, v26.4s, v1.s[0]\n"
    "fmla v11.4s, v25.4s, v2.s[0]\n"
    "fmla v12.4s, v26.4s, v2.s[0]\n"
    "fmla v13.4s, v25.4s, v3.s[0]\n"
    "fmla v14.4s, v26.4s, v3.s[0]\n"
    "fmla v15.4s, v25.4s, v4.s[0]\n"
    "fmla v16.4s, v26.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v5.s[0]\n"
    "fmla v18.4s, v26.4s, v5.s[0]\n"
    "fmla v19.4s, v25.4s, v6.s[0]\n"
    "fmla v20.4s, v26.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v7.s[0]\n"
    "fmla v22.4s, v26.4s, v7.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v9.4s, v23.4s, v2.s[0]\n"
    "fmla v10.4s, v24.4s, v2.s[0]\n"
    "fmla v11.4s, v23.4s, v3.s[0]\n"
    "fmla v12.4s, v24.4s, v3.s[0]\n"
    "fmla v13.4s, v23.4s, v4.s[0]\n"
    "fmla v14.4s, v24.4s, v4.s[0]\n"
    "fmla v15.4s, v23.4s, v5.s[0]\n"
    "fmla v16.4s, v24.4s, v5.s[0]\n"
    "fmla v17.4s, v23.4s, v6.s[0]\n"
    "fmla v18.4s, v24.4s, v6.s[0]\n"
    "fmla v19.4s, v23.4s, v7.s[0]\n"
    "fmla v20.4s, v24.4s, v7.s[0]\n"
    "fmla v21.4s, v23.4s, v8.s[0]\n"
    "fmla v22.4s, v24.4s, v8.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v11.4s, v25.4s, v1.s[1]\n"
    "fmla v12.4s, v26.4s, v1.s[1]\n"
    "fmla v13.4s, v25.4s, v2.s[1]\n"
    "fmla v14.4s, v26.4s, v2.s[1]\n"
    "fmla v15.4s, v25.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v3.s[1]\n"
    "fmla v17.4s, v25.4s, v4.s[1]\n"
    "fmla v18.4s, v26.4s, v4.s[1]\n"
    "fmla v19.4s, v25.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v5.s[1]\n"
    "fmla v21.4s, v25.4s, v6.s[1]\n"
    "fmla v22.4s, v26.4s, v6.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 1
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
    "fmla v21.4s, v23.4s, v7.s[1]\n"
    "fmla v22.4s, v24.4s, v7.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v9.4s, v25.4s, v2.s[1]\n"
    "fmla v10.4s, v26.4s, v2.s[1]\n"
    "fmla v11.4s, v25.4s, v3.s[1]\n"
    "fmla v12.4s, v26.4s, v3.s[1]\n"
    "fmla v13.4s, v25.4s, v4.s[1]\n"
    "fmla v14.4s, v26.4s, v4.s[1]\n"
    "fmla v15.4s, v25.4s, v5.s[1]\n"
    "fmla v16.4s, v26.4s, v5.s[1]\n"
    "fmla v17.4s, v25.4s, v6.s[1]\n"
    "fmla v18.4s, v26.4s, v6.s[1]\n"
    "fmla v19.4s, v25.4s, v7.s[1]\n"
    "fmla v20.4s, v26.4s, v7.s[1]\n"
    "fmla v21.4s, v25.4s, v8.s[1]\n"
    "fmla v22.4s, v26.4s, v8.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v11.4s, v23.4s, v1.s[2]\n"
    "fmla v12.4s, v24.4s, v1.s[2]\n"
    "fmla v13.4s, v23.4s, v2.s[2]\n"
    "fmla v14.4s, v24.4s, v2.s[2]\n"
    "fmla v15.4s, v23.4s, v3.s[2]\n"
    "fmla v16.4s, v24.4s, v3.s[2]\n"
    "fmla v17.4s, v23.4s, v4.s[2]\n"
    "fmla v18.4s, v24.4s, v4.s[2]\n"
    "fmla v19.4s, v23.4s, v5.s[2]\n"
    "fmla v20.4s, v24.4s, v5.s[2]\n"
    "fmla v21.4s, v23.4s, v6.s[2]\n"
    "fmla v22.4s, v24.4s, v6.s[2]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v9.4s, v25.4s, v1.s[2]\n"
    "fmla v10.4s, v26.4s, v1.s[2]\n"
    "fmla v11.4s, v25.4s, v2.s[2]\n"
    "fmla v12.4s, v26.4s, v2.s[2]\n"
    "fmla v13.4s, v25.4s, v3.s[2]\n"
    "fmla v14.4s, v26.4s, v3.s[2]\n"
    "fmla v15.4s, v25.4s, v4.s[2]\n"
    "fmla v16.4s, v26.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v5.s[2]\n"
    "fmla v18.4s, v26.4s, v5.s[2]\n"
    "fmla v19.4s, v25.4s, v6.s[2]\n"
    "fmla v20.4s, v26.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v7.s[2]\n"
    "fmla v22.4s, v26.4s, v7.s[2]\n"
// Filter width idx 2
    "fmla v9.4s, v23.4s, v2.s[2]\n"
    "fmla v10.4s, v24.4s, v2.s[2]\n"
    "fmla v11.4s, v23.4s, v3.s[2]\n"
    "fmla v12.4s, v24.4s, v3.s[2]\n"
    "fmla v13.4s, v23.4s, v4.s[2]\n"
    "fmla v14.4s, v24.4s, v4.s[2]\n"
    "fmla v15.4s, v23.4s, v5.s[2]\n"
    "fmla v16.4s, v24.4s, v5.s[2]\n"
    "fmla v17.4s, v23.4s, v6.s[2]\n"
    "fmla v18.4s, v24.4s, v6.s[2]\n"
    "fmla v19.4s, v23.4s, v7.s[2]\n"
    "fmla v20.4s, v24.4s, v7.s[2]\n"
    "fmla v21.4s, v23.4s, v8.s[2]\n"
    "fmla v22.4s, v24.4s, v8.s[2]\n"
    "stp q9, q10, [%[out], #0]\n"
    "stp q11, q12, [%[out], #32]\n"
    "stp q13, q14, [%[out], #64]\n"
    "stp q15, q16, [%[out], #96]\n"
    "stp q17, q18, [%[out], #128]\n"
    "stp q19, q20, [%[out], #160]\n"
    "stp q21, q22, [%[out], #192]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_7_8_1_1_0_1_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Duplicate index: 12
// [1, 2, 3, 3, 3, 3, 3, 2, 1]
// Number of Input Regs: 9, Filter Regs: 4 Output Regs: 14
// Total number of registers required: 27
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Out - [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
// Fil - [23, 24, 25, 26, 23, 24, 25, 26, 23, 24]
// Register mapping diagram
//           0  1  2  3  4  5  6  7  8 
//
//23 25 23   9 11 13 15 17 19 21 
//24 26 24  10 12 14 16 18 20 22 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + 3*7 + j));
            printf("%6.3f\t", *(inputPtr + 3*8 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q9, q10, [%[out], #0]\n"
    "ldp q11, q12, [%[out], #32]\n"
    "ldp q13, q14, [%[out], #64]\n"
    "ldp q15, q16, [%[out], #96]\n"
    "ldp q17, q18, [%[out], #128]\n"
    "ldp q19, q20, [%[out], #160]\n"
    "ldp q21, q22, [%[out], #192]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"
    "ldr q5, [x10, #60]\n"
    "ldr q6, [x10, #72]\n"
    "ldr q7, [x10, #84]\n"

// K index 0
// Filter width idx 0
    "fmla v9.4s, v23.4s, v0.s[0]\n"
    "fmla v10.4s, v24.4s, v0.s[0]\n"
    "fmla v11.4s, v23.4s, v1.s[0]\n"
    "fmla v12.4s, v24.4s, v1.s[0]\n"
    "fmla v13.4s, v23.4s, v2.s[0]\n"
    "fmla v14.4s, v24.4s, v2.s[0]\n"
    "fmla v15.4s, v23.4s, v3.s[0]\n"
    "fmla v16.4s, v24.4s, v3.s[0]\n"
    "fmla v17.4s, v23.4s, v4.s[0]\n"
    "fmla v18.4s, v24.4s, v4.s[0]\n"
    "fmla v19.4s, v23.4s, v5.s[0]\n"
    "fmla v20.4s, v24.4s, v5.s[0]\n"
    "fmla v21.4s, v23.4s, v6.s[0]\n"
    "fmla v22.4s, v24.4s, v6.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v9.4s, v25.4s, v1.s[0]\n"
    "fmla v10.4s, v26.4s, v1.s[0]\n"
    "fmla v11.4s, v25.4s, v2.s[0]\n"
    "fmla v12.4s, v26.4s, v2.s[0]\n"
    "fmla v13.4s, v25.4s, v3.s[0]\n"
    "fmla v14.4s, v26.4s, v3.s[0]\n"
    "fmla v15.4s, v25.4s, v4.s[0]\n"
    "fmla v16.4s, v26.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v5.s[0]\n"
    "fmla v18.4s, v26.4s, v5.s[0]\n"
    "fmla v19.4s, v25.4s, v6.s[0]\n"
    "fmla v20.4s, v26.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v7.s[0]\n"
    "fmla v22.4s, v26.4s, v7.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v9.4s, v23.4s, v2.s[0]\n"
    "fmla v10.4s, v24.4s, v2.s[0]\n"
    "fmla v11.4s, v23.4s, v3.s[0]\n"
    "fmla v12.4s, v24.4s, v3.s[0]\n"
    "fmla v13.4s, v23.4s, v4.s[0]\n"
    "fmla v14.4s, v24.4s, v4.s[0]\n"
    "fmla v15.4s, v23.4s, v5.s[0]\n"
    "fmla v16.4s, v24.4s, v5.s[0]\n"
    "fmla v17.4s, v23.4s, v6.s[0]\n"
    "fmla v18.4s, v24.4s, v6.s[0]\n"
    "fmla v19.4s, v23.4s, v7.s[0]\n"
    "fmla v20.4s, v24.4s, v7.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v9.4s, v25.4s, v0.s[1]\n"
    "fmla v10.4s, v26.4s, v0.s[1]\n"
    "fmla v11.4s, v25.4s, v1.s[1]\n"
    "fmla v12.4s, v26.4s, v1.s[1]\n"
    "fmla v13.4s, v25.4s, v2.s[1]\n"
    "fmla v14.4s, v26.4s, v2.s[1]\n"
    "fmla v15.4s, v25.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v3.s[1]\n"
    "fmla v17.4s, v25.4s, v4.s[1]\n"
    "fmla v18.4s, v26.4s, v4.s[1]\n"
    "fmla v19.4s, v25.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v5.s[1]\n"
    "fmla v21.4s, v25.4s, v6.s[1]\n"
    "fmla v22.4s, v26.4s, v6.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 1
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
    "fmla v21.4s, v23.4s, v7.s[1]\n"
    "fmla v22.4s, v24.4s, v7.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v9.4s, v25.4s, v2.s[1]\n"
    "fmla v10.4s, v26.4s, v2.s[1]\n"
    "fmla v11.4s, v25.4s, v3.s[1]\n"
    "fmla v12.4s, v26.4s, v3.s[1]\n"
    "fmla v13.4s, v25.4s, v4.s[1]\n"
    "fmla v14.4s, v26.4s, v4.s[1]\n"
    "fmla v15.4s, v25.4s, v5.s[1]\n"
    "fmla v16.4s, v26.4s, v5.s[1]\n"
    "fmla v17.4s, v25.4s, v6.s[1]\n"
    "fmla v18.4s, v26.4s, v6.s[1]\n"
    "fmla v19.4s, v25.4s, v7.s[1]\n"
    "fmla v20.4s, v26.4s, v7.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v9.4s, v23.4s, v0.s[2]\n"
    "fmla v10.4s, v24.4s, v0.s[2]\n"
    "fmla v11.4s, v23.4s, v1.s[2]\n"
    "fmla v12.4s, v24.4s, v1.s[2]\n"
    "fmla v13.4s, v23.4s, v2.s[2]\n"
    "fmla v14.4s, v24.4s, v2.s[2]\n"
    "fmla v15.4s, v23.4s, v3.s[2]\n"
    "fmla v16.4s, v24.4s, v3.s[2]\n"
    "fmla v17.4s, v23.4s, v4.s[2]\n"
    "fmla v18.4s, v24.4s, v4.s[2]\n"
    "fmla v19.4s, v23.4s, v5.s[2]\n"
    "fmla v20.4s, v24.4s, v5.s[2]\n"
    "fmla v21.4s, v23.4s, v6.s[2]\n"
    "fmla v22.4s, v24.4s, v6.s[2]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v9.4s, v25.4s, v1.s[2]\n"
    "fmla v10.4s, v26.4s, v1.s[2]\n"
    "fmla v11.4s, v25.4s, v2.s[2]\n"
    "fmla v12.4s, v26.4s, v2.s[2]\n"
    "fmla v13.4s, v25.4s, v3.s[2]\n"
    "fmla v14.4s, v26.4s, v3.s[2]\n"
    "fmla v15.4s, v25.4s, v4.s[2]\n"
    "fmla v16.4s, v26.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v5.s[2]\n"
    "fmla v18.4s, v26.4s, v5.s[2]\n"
    "fmla v19.4s, v25.4s, v6.s[2]\n"
    "fmla v20.4s, v26.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v7.s[2]\n"
    "fmla v22.4s, v26.4s, v7.s[2]\n"
// Filter width idx 2
    "fmla v9.4s, v23.4s, v2.s[2]\n"
    "fmla v10.4s, v24.4s, v2.s[2]\n"
    "fmla v11.4s, v23.4s, v3.s[2]\n"
    "fmla v12.4s, v24.4s, v3.s[2]\n"
    "fmla v13.4s, v23.4s, v4.s[2]\n"
    "fmla v14.4s, v24.4s, v4.s[2]\n"
    "fmla v15.4s, v23.4s, v5.s[2]\n"
    "fmla v16.4s, v24.4s, v5.s[2]\n"
    "fmla v17.4s, v23.4s, v6.s[2]\n"
    "fmla v18.4s, v24.4s, v6.s[2]\n"
    "fmla v19.4s, v23.4s, v7.s[2]\n"
    "fmla v20.4s, v24.4s, v7.s[2]\n"
    "stp q9, q10, [%[out], #0]\n"
    "stp q11, q12, [%[out], #32]\n"
    "stp q13, q14, [%[out], #64]\n"
    "stp q15, q16, [%[out], #96]\n"
    "stp q17, q18, [%[out], #128]\n"
    "stp q19, q20, [%[out], #160]\n"
    "stp q21, q22, [%[out], #192]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_8_8_1_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
// Duplicate index: 14
// [1, 2, 3, 3, 3, 3, 3, 3, 2, 1]
// Number of Input Regs: 10, Filter Regs: 4 Output Regs: 16
// Total number of registers required: 30
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
// Out - [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
// Fil - [26, 27, 28, 29, 26, 27, 28, 29, 26, 27]
// Register mapping diagram
//           0  1  2  3  4  5  6  7  8  9 
//
//26 28 26  10 12 14 16 18 20 22 24 
//27 29 27  11 13 15 17 19 21 23 25 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + 3*7 + j));
            printf("%6.3f\t", *(inputPtr + 3*8 + j));
            printf("%6.3f\t", *(inputPtr + 3*9 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q10, q11, [%[out], #0]\n"
    "ldp q12, q13, [%[out], #32]\n"
    "ldp q14, q15, [%[out], #64]\n"
    "ldp q16, q17, [%[out], #96]\n"
    "ldp q18, q19, [%[out], #128]\n"
    "ldp q20, q21, [%[out], #160]\n"
    "ldp q22, q23, [%[out], #192]\n"
    "ldp q24, q25, [%[out], #224]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"
    "ldr q5, [x10, #60]\n"
    "ldr q6, [x10, #72]\n"
    "ldr q7, [x10, #84]\n"
    "ldr q8, [x10, #96]\n"
    "ldr q9, [x10, #108]\n"

// K index 0
// Filter width idx 0
    "fmla v10.4s, v26.4s, v0.s[0]\n"
    "fmla v11.4s, v27.4s, v0.s[0]\n"
    "fmla v12.4s, v26.4s, v1.s[0]\n"
    "fmla v13.4s, v27.4s, v1.s[0]\n"
    "fmla v14.4s, v26.4s, v2.s[0]\n"
    "fmla v15.4s, v27.4s, v2.s[0]\n"
    "fmla v16.4s, v26.4s, v3.s[0]\n"
    "fmla v17.4s, v27.4s, v3.s[0]\n"
    "fmla v18.4s, v26.4s, v4.s[0]\n"
    "fmla v19.4s, v27.4s, v4.s[0]\n"
    "fmla v20.4s, v26.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v5.s[0]\n"
    "fmla v22.4s, v26.4s, v6.s[0]\n"
    "fmla v23.4s, v27.4s, v6.s[0]\n"
    "fmla v24.4s, v26.4s, v7.s[0]\n"
    "fmla v25.4s, v27.4s, v7.s[0]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v10.4s, v28.4s, v1.s[0]\n"
    "fmla v11.4s, v29.4s, v1.s[0]\n"
    "fmla v12.4s, v28.4s, v2.s[0]\n"
    "fmla v13.4s, v29.4s, v2.s[0]\n"
    "fmla v14.4s, v28.4s, v3.s[0]\n"
    "fmla v15.4s, v29.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v4.s[0]\n"
    "fmla v17.4s, v29.4s, v4.s[0]\n"
    "fmla v18.4s, v28.4s, v5.s[0]\n"
    "fmla v19.4s, v29.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v6.s[0]\n"
    "fmla v21.4s, v29.4s, v6.s[0]\n"
    "fmla v22.4s, v28.4s, v7.s[0]\n"
    "fmla v23.4s, v29.4s, v7.s[0]\n"
    "fmla v24.4s, v28.4s, v8.s[0]\n"
    "fmla v25.4s, v29.4s, v8.s[0]\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v10.4s, v26.4s, v2.s[0]\n"
    "fmla v11.4s, v27.4s, v2.s[0]\n"
    "fmla v12.4s, v26.4s, v3.s[0]\n"
    "fmla v13.4s, v27.4s, v3.s[0]\n"
    "fmla v14.4s, v26.4s, v4.s[0]\n"
    "fmla v15.4s, v27.4s, v4.s[0]\n"
    "fmla v16.4s, v26.4s, v5.s[0]\n"
    "fmla v17.4s, v27.4s, v5.s[0]\n"
    "fmla v18.4s, v26.4s, v6.s[0]\n"
    "fmla v19.4s, v27.4s, v6.s[0]\n"
    "fmla v20.4s, v26.4s, v7.s[0]\n"
    "fmla v21.4s, v27.4s, v7.s[0]\n"
    "fmla v22.4s, v26.4s, v8.s[0]\n"
    "fmla v23.4s, v27.4s, v8.s[0]\n"
    "fmla v24.4s, v26.4s, v9.s[0]\n"
    "fmla v25.4s, v27.4s, v9.s[0]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v10.4s, v28.4s, v0.s[1]\n"
    "fmla v11.4s, v29.4s, v0.s[1]\n"
    "fmla v12.4s, v28.4s, v1.s[1]\n"
    "fmla v13.4s, v29.4s, v1.s[1]\n"
    "fmla v14.4s, v28.4s, v2.s[1]\n"
    "fmla v15.4s, v29.4s, v2.s[1]\n"
    "fmla v16.4s, v28.4s, v3.s[1]\n"
    "fmla v17.4s, v29.4s, v3.s[1]\n"
    "fmla v18.4s, v28.4s, v4.s[1]\n"
    "fmla v19.4s, v29.4s, v4.s[1]\n"
    "fmla v20.4s, v28.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v5.s[1]\n"
    "fmla v22.4s, v28.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v6.s[1]\n"
    "fmla v24.4s, v28.4s, v7.s[1]\n"
    "fmla v25.4s, v29.4s, v7.s[1]\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// Filter width idx 1
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
    "fmla v24.4s, v26.4s, v8.s[1]\n"
    "fmla v25.4s, v27.4s, v8.s[1]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v10.4s, v28.4s, v2.s[1]\n"
    "fmla v11.4s, v29.4s, v2.s[1]\n"
    "fmla v12.4s, v28.4s, v3.s[1]\n"
    "fmla v13.4s, v29.4s, v3.s[1]\n"
    "fmla v14.4s, v28.4s, v4.s[1]\n"
    "fmla v15.4s, v29.4s, v4.s[1]\n"
    "fmla v16.4s, v28.4s, v5.s[1]\n"
    "fmla v17.4s, v29.4s, v5.s[1]\n"
    "fmla v18.4s, v28.4s, v6.s[1]\n"
    "fmla v19.4s, v29.4s, v6.s[1]\n"
    "fmla v20.4s, v28.4s, v7.s[1]\n"
    "fmla v21.4s, v29.4s, v7.s[1]\n"
    "fmla v22.4s, v28.4s, v8.s[1]\n"
    "fmla v23.4s, v29.4s, v8.s[1]\n"
    "fmla v24.4s, v28.4s, v9.s[1]\n"
    "fmla v25.4s, v29.4s, v9.s[1]\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v10.4s, v26.4s, v0.s[2]\n"
    "fmla v11.4s, v27.4s, v0.s[2]\n"
    "fmla v12.4s, v26.4s, v1.s[2]\n"
    "fmla v13.4s, v27.4s, v1.s[2]\n"
    "fmla v14.4s, v26.4s, v2.s[2]\n"
    "fmla v15.4s, v27.4s, v2.s[2]\n"
    "fmla v16.4s, v26.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v3.s[2]\n"
    "fmla v18.4s, v26.4s, v4.s[2]\n"
    "fmla v19.4s, v27.4s, v4.s[2]\n"
    "fmla v20.4s, v26.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v5.s[2]\n"
    "fmla v22.4s, v26.4s, v6.s[2]\n"
    "fmla v23.4s, v27.4s, v6.s[2]\n"
    "fmla v24.4s, v26.4s, v7.s[2]\n"
    "fmla v25.4s, v27.4s, v7.s[2]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v10.4s, v28.4s, v1.s[2]\n"
    "fmla v11.4s, v29.4s, v1.s[2]\n"
    "fmla v12.4s, v28.4s, v2.s[2]\n"
    "fmla v13.4s, v29.4s, v2.s[2]\n"
    "fmla v14.4s, v28.4s, v3.s[2]\n"
    "fmla v15.4s, v29.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v4.s[2]\n"
    "fmla v17.4s, v29.4s, v4.s[2]\n"
    "fmla v18.4s, v28.4s, v5.s[2]\n"
    "fmla v19.4s, v29.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v6.s[2]\n"
    "fmla v21.4s, v29.4s, v6.s[2]\n"
    "fmla v22.4s, v28.4s, v7.s[2]\n"
    "fmla v23.4s, v29.4s, v7.s[2]\n"
    "fmla v24.4s, v28.4s, v8.s[2]\n"
    "fmla v25.4s, v29.4s, v8.s[2]\n"
// Filter width idx 2
    "fmla v10.4s, v26.4s, v2.s[2]\n"
    "fmla v11.4s, v27.4s, v2.s[2]\n"
    "fmla v12.4s, v26.4s, v3.s[2]\n"
    "fmla v13.4s, v27.4s, v3.s[2]\n"
    "fmla v14.4s, v26.4s, v4.s[2]\n"
    "fmla v15.4s, v27.4s, v4.s[2]\n"
    "fmla v16.4s, v26.4s, v5.s[2]\n"
    "fmla v17.4s, v27.4s, v5.s[2]\n"
    "fmla v18.4s, v26.4s, v6.s[2]\n"
    "fmla v19.4s, v27.4s, v6.s[2]\n"
    "fmla v20.4s, v26.4s, v7.s[2]\n"
    "fmla v21.4s, v27.4s, v7.s[2]\n"
    "fmla v22.4s, v26.4s, v8.s[2]\n"
    "fmla v23.4s, v27.4s, v8.s[2]\n"
    "fmla v24.4s, v26.4s, v9.s[2]\n"
    "fmla v25.4s, v27.4s, v9.s[2]\n"
    "stp q10, q11, [%[out], #0]\n"
    "stp q12, q13, [%[out], #32]\n"
    "stp q14, q15, [%[out], #64]\n"
    "stp q16, q17, [%[out], #96]\n"
    "stp q18, q19, [%[out], #128]\n"
    "stp q20, q21, [%[out], #160]\n"
    "stp q22, q23, [%[out], #192]\n"
    "stp q24, q25, [%[out], #224]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_8_8_1_1_1_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
// Duplicate index: 14
// [1, 2, 3, 3, 3, 3, 3, 3, 2, 1]
// Number of Input Regs: 10, Filter Regs: 4 Output Regs: 16
// Total number of registers required: 30
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
// Out - [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
// Fil - [26, 27, 28, 29, 26, 27, 28, 29, 26, 27]
// Register mapping diagram
//           0  1  2  3  4  5  6  7  8  9 
//
//26 28 26  10 12 14 16 18 20 22 24 
//27 29 27  11 13 15 17 19 21 23 25 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + 3*7 + j));
            printf("%6.3f\t", *(inputPtr + 3*8 + j));
            printf("%6.3f\t", *(inputPtr + 3*9 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 1*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q10, q11, [%[out], #0]\n"
    "ldp q12, q13, [%[out], #32]\n"
    "ldp q14, q15, [%[out], #64]\n"
    "ldp q16, q17, [%[out], #96]\n"
    "ldp q18, q19, [%[out], #128]\n"
    "ldp q20, q21, [%[out], #160]\n"
    "ldp q22, q23, [%[out], #192]\n"
    "ldp q24, q25, [%[out], #224]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// Load Input
    "ldr q1, [x10, #0]\n"
    "ldr q2, [x10, #12]\n"
    "ldr q3, [x10, #24]\n"
    "ldr q4, [x10, #36]\n"
    "ldr q5, [x10, #48]\n"
    "ldr q6, [x10, #60]\n"
    "ldr q7, [x10, #72]\n"
    "ldr q8, [x10, #84]\n"
    "ldr q9, [x10, #96]\n"

// K index 0
// Filter width idx 0
    "fmla v12.4s, v26.4s, v1.s[0]\n"
    "fmla v13.4s, v27.4s, v1.s[0]\n"
    "fmla v14.4s, v26.4s, v2.s[0]\n"
    "fmla v15.4s, v27.4s, v2.s[0]\n"
    "fmla v16.4s, v26.4s, v3.s[0]\n"
    "fmla v17.4s, v27.4s, v3.s[0]\n"
    "fmla v18.4s, v26.4s, v4.s[0]\n"
    "fmla v19.4s, v27.4s, v4.s[0]\n"
    "fmla v20.4s, v26.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v5.s[0]\n"
    "fmla v22.4s, v26.4s, v6.s[0]\n"
    "fmla v23.4s, v27.4s, v6.s[0]\n"
    "fmla v24.4s, v26.4s, v7.s[0]\n"
    "fmla v25.4s, v27.4s, v7.s[0]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v10.4s, v28.4s, v1.s[0]\n"
    "fmla v11.4s, v29.4s, v1.s[0]\n"
    "fmla v12.4s, v28.4s, v2.s[0]\n"
    "fmla v13.4s, v29.4s, v2.s[0]\n"
    "fmla v14.4s, v28.4s, v3.s[0]\n"
    "fmla v15.4s, v29.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v4.s[0]\n"
    "fmla v17.4s, v29.4s, v4.s[0]\n"
    "fmla v18.4s, v28.4s, v5.s[0]\n"
    "fmla v19.4s, v29.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v6.s[0]\n"
    "fmla v21.4s, v29.4s, v6.s[0]\n"
    "fmla v22.4s, v28.4s, v7.s[0]\n"
    "fmla v23.4s, v29.4s, v7.s[0]\n"
    "fmla v24.4s, v28.4s, v8.s[0]\n"
    "fmla v25.4s, v29.4s, v8.s[0]\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v10.4s, v26.4s, v2.s[0]\n"
    "fmla v11.4s, v27.4s, v2.s[0]\n"
    "fmla v12.4s, v26.4s, v3.s[0]\n"
    "fmla v13.4s, v27.4s, v3.s[0]\n"
    "fmla v14.4s, v26.4s, v4.s[0]\n"
    "fmla v15.4s, v27.4s, v4.s[0]\n"
    "fmla v16.4s, v26.4s, v5.s[0]\n"
    "fmla v17.4s, v27.4s, v5.s[0]\n"
    "fmla v18.4s, v26.4s, v6.s[0]\n"
    "fmla v19.4s, v27.4s, v6.s[0]\n"
    "fmla v20.4s, v26.4s, v7.s[0]\n"
    "fmla v21.4s, v27.4s, v7.s[0]\n"
    "fmla v22.4s, v26.4s, v8.s[0]\n"
    "fmla v23.4s, v27.4s, v8.s[0]\n"
    "fmla v24.4s, v26.4s, v9.s[0]\n"
    "fmla v25.4s, v27.4s, v9.s[0]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v12.4s, v28.4s, v1.s[1]\n"
    "fmla v13.4s, v29.4s, v1.s[1]\n"
    "fmla v14.4s, v28.4s, v2.s[1]\n"
    "fmla v15.4s, v29.4s, v2.s[1]\n"
    "fmla v16.4s, v28.4s, v3.s[1]\n"
    "fmla v17.4s, v29.4s, v3.s[1]\n"
    "fmla v18.4s, v28.4s, v4.s[1]\n"
    "fmla v19.4s, v29.4s, v4.s[1]\n"
    "fmla v20.4s, v28.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v5.s[1]\n"
    "fmla v22.4s, v28.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v6.s[1]\n"
    "fmla v24.4s, v28.4s, v7.s[1]\n"
    "fmla v25.4s, v29.4s, v7.s[1]\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// Filter width idx 1
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
    "fmla v24.4s, v26.4s, v8.s[1]\n"
    "fmla v25.4s, v27.4s, v8.s[1]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v10.4s, v28.4s, v2.s[1]\n"
    "fmla v11.4s, v29.4s, v2.s[1]\n"
    "fmla v12.4s, v28.4s, v3.s[1]\n"
    "fmla v13.4s, v29.4s, v3.s[1]\n"
    "fmla v14.4s, v28.4s, v4.s[1]\n"
    "fmla v15.4s, v29.4s, v4.s[1]\n"
    "fmla v16.4s, v28.4s, v5.s[1]\n"
    "fmla v17.4s, v29.4s, v5.s[1]\n"
    "fmla v18.4s, v28.4s, v6.s[1]\n"
    "fmla v19.4s, v29.4s, v6.s[1]\n"
    "fmla v20.4s, v28.4s, v7.s[1]\n"
    "fmla v21.4s, v29.4s, v7.s[1]\n"
    "fmla v22.4s, v28.4s, v8.s[1]\n"
    "fmla v23.4s, v29.4s, v8.s[1]\n"
    "fmla v24.4s, v28.4s, v9.s[1]\n"
    "fmla v25.4s, v29.4s, v9.s[1]\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v12.4s, v26.4s, v1.s[2]\n"
    "fmla v13.4s, v27.4s, v1.s[2]\n"
    "fmla v14.4s, v26.4s, v2.s[2]\n"
    "fmla v15.4s, v27.4s, v2.s[2]\n"
    "fmla v16.4s, v26.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v3.s[2]\n"
    "fmla v18.4s, v26.4s, v4.s[2]\n"
    "fmla v19.4s, v27.4s, v4.s[2]\n"
    "fmla v20.4s, v26.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v5.s[2]\n"
    "fmla v22.4s, v26.4s, v6.s[2]\n"
    "fmla v23.4s, v27.4s, v6.s[2]\n"
    "fmla v24.4s, v26.4s, v7.s[2]\n"
    "fmla v25.4s, v27.4s, v7.s[2]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v10.4s, v28.4s, v1.s[2]\n"
    "fmla v11.4s, v29.4s, v1.s[2]\n"
    "fmla v12.4s, v28.4s, v2.s[2]\n"
    "fmla v13.4s, v29.4s, v2.s[2]\n"
    "fmla v14.4s, v28.4s, v3.s[2]\n"
    "fmla v15.4s, v29.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v4.s[2]\n"
    "fmla v17.4s, v29.4s, v4.s[2]\n"
    "fmla v18.4s, v28.4s, v5.s[2]\n"
    "fmla v19.4s, v29.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v6.s[2]\n"
    "fmla v21.4s, v29.4s, v6.s[2]\n"
    "fmla v22.4s, v28.4s, v7.s[2]\n"
    "fmla v23.4s, v29.4s, v7.s[2]\n"
    "fmla v24.4s, v28.4s, v8.s[2]\n"
    "fmla v25.4s, v29.4s, v8.s[2]\n"
// Filter width idx 2
    "fmla v10.4s, v26.4s, v2.s[2]\n"
    "fmla v11.4s, v27.4s, v2.s[2]\n"
    "fmla v12.4s, v26.4s, v3.s[2]\n"
    "fmla v13.4s, v27.4s, v3.s[2]\n"
    "fmla v14.4s, v26.4s, v4.s[2]\n"
    "fmla v15.4s, v27.4s, v4.s[2]\n"
    "fmla v16.4s, v26.4s, v5.s[2]\n"
    "fmla v17.4s, v27.4s, v5.s[2]\n"
    "fmla v18.4s, v26.4s, v6.s[2]\n"
    "fmla v19.4s, v27.4s, v6.s[2]\n"
    "fmla v20.4s, v26.4s, v7.s[2]\n"
    "fmla v21.4s, v27.4s, v7.s[2]\n"
    "fmla v22.4s, v26.4s, v8.s[2]\n"
    "fmla v23.4s, v27.4s, v8.s[2]\n"
    "fmla v24.4s, v26.4s, v9.s[2]\n"
    "fmla v25.4s, v27.4s, v9.s[2]\n"
    "stp q10, q11, [%[out], #0]\n"
    "stp q12, q13, [%[out], #32]\n"
    "stp q14, q15, [%[out], #64]\n"
    "stp q16, q17, [%[out], #96]\n"
    "stp q18, q19, [%[out], #128]\n"
    "stp q20, q21, [%[out], #160]\n"
    "stp q22, q23, [%[out], #192]\n"
    "stp q24, q25, [%[out], #224]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_8_8_1_1_0_1_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
// Duplicate index: 14
// [1, 2, 3, 3, 3, 3, 3, 3, 2, 1]
// Number of Input Regs: 10, Filter Regs: 4 Output Regs: 16
// Total number of registers required: 30
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
// Out - [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
// Fil - [26, 27, 28, 29, 26, 27, 28, 29, 26, 27]
// Register mapping diagram
//           0  1  2  3  4  5  6  7  8  9 
//
//26 28 26  10 12 14 16 18 20 22 24 
//27 29 27  11 13 15 17 19 21 23 25 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + 3*7 + j));
            printf("%6.3f\t", *(inputPtr + 3*8 + j));
            printf("%6.3f\t", *(inputPtr + 3*9 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q10, q11, [%[out], #0]\n"
    "ldp q12, q13, [%[out], #32]\n"
    "ldp q14, q15, [%[out], #64]\n"
    "ldp q16, q17, [%[out], #96]\n"
    "ldp q18, q19, [%[out], #128]\n"
    "ldp q20, q21, [%[out], #160]\n"
    "ldp q22, q23, [%[out], #192]\n"
    "ldp q24, q25, [%[out], #224]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"
    "ldr q5, [x10, #60]\n"
    "ldr q6, [x10, #72]\n"
    "ldr q7, [x10, #84]\n"
    "ldr q8, [x10, #96]\n"

// K index 0
// Filter width idx 0
    "fmla v10.4s, v26.4s, v0.s[0]\n"
    "fmla v11.4s, v27.4s, v0.s[0]\n"
    "fmla v12.4s, v26.4s, v1.s[0]\n"
    "fmla v13.4s, v27.4s, v1.s[0]\n"
    "fmla v14.4s, v26.4s, v2.s[0]\n"
    "fmla v15.4s, v27.4s, v2.s[0]\n"
    "fmla v16.4s, v26.4s, v3.s[0]\n"
    "fmla v17.4s, v27.4s, v3.s[0]\n"
    "fmla v18.4s, v26.4s, v4.s[0]\n"
    "fmla v19.4s, v27.4s, v4.s[0]\n"
    "fmla v20.4s, v26.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v5.s[0]\n"
    "fmla v22.4s, v26.4s, v6.s[0]\n"
    "fmla v23.4s, v27.4s, v6.s[0]\n"
    "fmla v24.4s, v26.4s, v7.s[0]\n"
    "fmla v25.4s, v27.4s, v7.s[0]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v10.4s, v28.4s, v1.s[0]\n"
    "fmla v11.4s, v29.4s, v1.s[0]\n"
    "fmla v12.4s, v28.4s, v2.s[0]\n"
    "fmla v13.4s, v29.4s, v2.s[0]\n"
    "fmla v14.4s, v28.4s, v3.s[0]\n"
    "fmla v15.4s, v29.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v4.s[0]\n"
    "fmla v17.4s, v29.4s, v4.s[0]\n"
    "fmla v18.4s, v28.4s, v5.s[0]\n"
    "fmla v19.4s, v29.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v6.s[0]\n"
    "fmla v21.4s, v29.4s, v6.s[0]\n"
    "fmla v22.4s, v28.4s, v7.s[0]\n"
    "fmla v23.4s, v29.4s, v7.s[0]\n"
    "fmla v24.4s, v28.4s, v8.s[0]\n"
    "fmla v25.4s, v29.4s, v8.s[0]\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v10.4s, v26.4s, v2.s[0]\n"
    "fmla v11.4s, v27.4s, v2.s[0]\n"
    "fmla v12.4s, v26.4s, v3.s[0]\n"
    "fmla v13.4s, v27.4s, v3.s[0]\n"
    "fmla v14.4s, v26.4s, v4.s[0]\n"
    "fmla v15.4s, v27.4s, v4.s[0]\n"
    "fmla v16.4s, v26.4s, v5.s[0]\n"
    "fmla v17.4s, v27.4s, v5.s[0]\n"
    "fmla v18.4s, v26.4s, v6.s[0]\n"
    "fmla v19.4s, v27.4s, v6.s[0]\n"
    "fmla v20.4s, v26.4s, v7.s[0]\n"
    "fmla v21.4s, v27.4s, v7.s[0]\n"
    "fmla v22.4s, v26.4s, v8.s[0]\n"
    "fmla v23.4s, v27.4s, v8.s[0]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v10.4s, v28.4s, v0.s[1]\n"
    "fmla v11.4s, v29.4s, v0.s[1]\n"
    "fmla v12.4s, v28.4s, v1.s[1]\n"
    "fmla v13.4s, v29.4s, v1.s[1]\n"
    "fmla v14.4s, v28.4s, v2.s[1]\n"
    "fmla v15.4s, v29.4s, v2.s[1]\n"
    "fmla v16.4s, v28.4s, v3.s[1]\n"
    "fmla v17.4s, v29.4s, v3.s[1]\n"
    "fmla v18.4s, v28.4s, v4.s[1]\n"
    "fmla v19.4s, v29.4s, v4.s[1]\n"
    "fmla v20.4s, v28.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v5.s[1]\n"
    "fmla v22.4s, v28.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v6.s[1]\n"
    "fmla v24.4s, v28.4s, v7.s[1]\n"
    "fmla v25.4s, v29.4s, v7.s[1]\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// Filter width idx 1
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
    "fmla v24.4s, v26.4s, v8.s[1]\n"
    "fmla v25.4s, v27.4s, v8.s[1]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v10.4s, v28.4s, v2.s[1]\n"
    "fmla v11.4s, v29.4s, v2.s[1]\n"
    "fmla v12.4s, v28.4s, v3.s[1]\n"
    "fmla v13.4s, v29.4s, v3.s[1]\n"
    "fmla v14.4s, v28.4s, v4.s[1]\n"
    "fmla v15.4s, v29.4s, v4.s[1]\n"
    "fmla v16.4s, v28.4s, v5.s[1]\n"
    "fmla v17.4s, v29.4s, v5.s[1]\n"
    "fmla v18.4s, v28.4s, v6.s[1]\n"
    "fmla v19.4s, v29.4s, v6.s[1]\n"
    "fmla v20.4s, v28.4s, v7.s[1]\n"
    "fmla v21.4s, v29.4s, v7.s[1]\n"
    "fmla v22.4s, v28.4s, v8.s[1]\n"
    "fmla v23.4s, v29.4s, v8.s[1]\n"
    "ld1 {v28.4s - v29.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v10.4s, v26.4s, v0.s[2]\n"
    "fmla v11.4s, v27.4s, v0.s[2]\n"
    "fmla v12.4s, v26.4s, v1.s[2]\n"
    "fmla v13.4s, v27.4s, v1.s[2]\n"
    "fmla v14.4s, v26.4s, v2.s[2]\n"
    "fmla v15.4s, v27.4s, v2.s[2]\n"
    "fmla v16.4s, v26.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v3.s[2]\n"
    "fmla v18.4s, v26.4s, v4.s[2]\n"
    "fmla v19.4s, v27.4s, v4.s[2]\n"
    "fmla v20.4s, v26.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v5.s[2]\n"
    "fmla v22.4s, v26.4s, v6.s[2]\n"
    "fmla v23.4s, v27.4s, v6.s[2]\n"
    "fmla v24.4s, v26.4s, v7.s[2]\n"
    "fmla v25.4s, v27.4s, v7.s[2]\n"
    "ld1 {v26.4s - v27.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v10.4s, v28.4s, v1.s[2]\n"
    "fmla v11.4s, v29.4s, v1.s[2]\n"
    "fmla v12.4s, v28.4s, v2.s[2]\n"
    "fmla v13.4s, v29.4s, v2.s[2]\n"
    "fmla v14.4s, v28.4s, v3.s[2]\n"
    "fmla v15.4s, v29.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v4.s[2]\n"
    "fmla v17.4s, v29.4s, v4.s[2]\n"
    "fmla v18.4s, v28.4s, v5.s[2]\n"
    "fmla v19.4s, v29.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v6.s[2]\n"
    "fmla v21.4s, v29.4s, v6.s[2]\n"
    "fmla v22.4s, v28.4s, v7.s[2]\n"
    "fmla v23.4s, v29.4s, v7.s[2]\n"
    "fmla v24.4s, v28.4s, v8.s[2]\n"
    "fmla v25.4s, v29.4s, v8.s[2]\n"
// Filter width idx 2
    "fmla v10.4s, v26.4s, v2.s[2]\n"
    "fmla v11.4s, v27.4s, v2.s[2]\n"
    "fmla v12.4s, v26.4s, v3.s[2]\n"
    "fmla v13.4s, v27.4s, v3.s[2]\n"
    "fmla v14.4s, v26.4s, v4.s[2]\n"
    "fmla v15.4s, v27.4s, v4.s[2]\n"
    "fmla v16.4s, v26.4s, v5.s[2]\n"
    "fmla v17.4s, v27.4s, v5.s[2]\n"
    "fmla v18.4s, v26.4s, v6.s[2]\n"
    "fmla v19.4s, v27.4s, v6.s[2]\n"
    "fmla v20.4s, v26.4s, v7.s[2]\n"
    "fmla v21.4s, v27.4s, v7.s[2]\n"
    "fmla v22.4s, v26.4s, v8.s[2]\n"
    "fmla v23.4s, v27.4s, v8.s[2]\n"
    "stp q10, q11, [%[out], #0]\n"
    "stp q12, q13, [%[out], #32]\n"
    "stp q14, q15, [%[out], #64]\n"
    "stp q16, q17, [%[out], #96]\n"
    "stp q18, q19, [%[out], #128]\n"
    "stp q20, q21, [%[out], #160]\n"
    "stp q22, q23, [%[out], #192]\n"
    "stp q24, q25, [%[out], #224]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_7_8_2_1_1_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 10], [10, 11, 12], [12, 13, 14]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Duplicate index: 6
// [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1]
// Number of Input Regs: 15, Filter Regs: 2 Output Regs: 14
// Total number of registers required: 31
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Out - [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
// Fil - [29, 30, 29, 30, 29, 30, 29, 30]
// Register mapping diagram
//           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 
//
//29 29 29  15    17    19    21    23    25    27    
//30 30 30  16    18    20    22    24    26    28    
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*7 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*8 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*9 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*10 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*11 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*12 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*13 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*14 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 1*3;
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
    "mov x11, %[fil]\n"
// Load Output
    "ldp q15, q16, [%[out], #0]\n"
    "ldp q17, q18, [%[out], #32]\n"
    "ldp q19, q20, [%[out], #64]\n"
    "ldp q21, q22, [%[out], #96]\n"
    "ldp q23, q24, [%[out], #128]\n"
    "ldp q25, q26, [%[out], #160]\n"
    "ldp q27, q28, [%[out], #192]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// Load Input
    "ldr q1, [x10, #0]\n"
    "ldr q2, [x10, #12]\n"
    "ldr q3, [x10, #24]\n"
    "ldr q4, [x10, #36]\n"
    "ldr q5, [x10, #48]\n"
    "ldr q6, [x10, #60]\n"
    "ldr q7, [x10, #72]\n"
    "ldr q8, [x10, #84]\n"
    "ldr q9, [x10, #96]\n"
    "ldr q10, [x10, #108]\n"
    "ldr q11, [x10, #120]\n"
    "ldr q12, [x10, #132]\n"
    "ldr q13, [x10, #144]\n"
    "ldr q14, [x10, #156]\n"

// Remaining 3 channels
// K index 0
// Filter width idx 0
    "fmla v17.4s, v29.4s, v2.s[0]\n"
    "fmla v19.4s, v29.4s, v4.s[0]\n"
    "fmla v21.4s, v29.4s, v6.s[0]\n"
    "fmla v23.4s, v29.4s, v8.s[0]\n"
    "fmla v25.4s, v29.4s, v10.s[0]\n"
    "fmla v27.4s, v29.4s, v12.s[0]\n"
    "ldr q29, [x11], #16\n"
    "fmla v18.4s, v30.4s, v2.s[0]\n"
    "fmla v20.4s, v30.4s, v4.s[0]\n"
    "fmla v22.4s, v30.4s, v6.s[0]\n"
    "fmla v24.4s, v30.4s, v8.s[0]\n"
    "fmla v26.4s, v30.4s, v10.s[0]\n"
    "fmla v28.4s, v30.4s, v12.s[0]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 1
    "fmla v15.4s, v29.4s, v1.s[0]\n"
    "fmla v17.4s, v29.4s, v3.s[0]\n"
    "fmla v19.4s, v29.4s, v5.s[0]\n"
    "fmla v21.4s, v29.4s, v7.s[0]\n"
    "fmla v23.4s, v29.4s, v9.s[0]\n"
    "fmla v25.4s, v29.4s, v11.s[0]\n"
    "fmla v27.4s, v29.4s, v13.s[0]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v1.s[0]\n"
    "fmla v18.4s, v30.4s, v3.s[0]\n"
    "fmla v20.4s, v30.4s, v5.s[0]\n"
    "fmla v22.4s, v30.4s, v7.s[0]\n"
    "fmla v24.4s, v30.4s, v9.s[0]\n"
    "fmla v26.4s, v30.4s, v11.s[0]\n"
    "fmla v28.4s, v30.4s, v13.s[0]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 2
    "fmla v15.4s, v29.4s, v2.s[0]\n"
    "fmla v17.4s, v29.4s, v4.s[0]\n"
    "fmla v19.4s, v29.4s, v6.s[0]\n"
    "fmla v21.4s, v29.4s, v8.s[0]\n"
    "fmla v23.4s, v29.4s, v10.s[0]\n"
    "fmla v25.4s, v29.4s, v12.s[0]\n"
    "fmla v27.4s, v29.4s, v14.s[0]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v2.s[0]\n"
    "fmla v18.4s, v30.4s, v4.s[0]\n"
    "fmla v20.4s, v30.4s, v6.s[0]\n"
    "fmla v22.4s, v30.4s, v8.s[0]\n"
    "fmla v24.4s, v30.4s, v10.s[0]\n"
    "fmla v26.4s, v30.4s, v12.s[0]\n"
    "fmla v28.4s, v30.4s, v14.s[0]\n"
    "ldr q30, [x11], #16\n"
// K index 1
// Filter width idx 0
    "fmla v17.4s, v29.4s, v2.s[1]\n"
    "fmla v19.4s, v29.4s, v4.s[1]\n"
    "fmla v21.4s, v29.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v8.s[1]\n"
    "fmla v25.4s, v29.4s, v10.s[1]\n"
    "fmla v27.4s, v29.4s, v12.s[1]\n"
    "ldr q29, [x11], #16\n"
    "fmla v18.4s, v30.4s, v2.s[1]\n"
    "fmla v20.4s, v30.4s, v4.s[1]\n"
    "fmla v22.4s, v30.4s, v6.s[1]\n"
    "fmla v24.4s, v30.4s, v8.s[1]\n"
    "fmla v26.4s, v30.4s, v10.s[1]\n"
    "fmla v28.4s, v30.4s, v12.s[1]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 1
    "fmla v15.4s, v29.4s, v1.s[1]\n"
    "fmla v17.4s, v29.4s, v3.s[1]\n"
    "fmla v19.4s, v29.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v7.s[1]\n"
    "fmla v23.4s, v29.4s, v9.s[1]\n"
    "fmla v25.4s, v29.4s, v11.s[1]\n"
    "fmla v27.4s, v29.4s, v13.s[1]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v1.s[1]\n"
    "fmla v18.4s, v30.4s, v3.s[1]\n"
    "fmla v20.4s, v30.4s, v5.s[1]\n"
    "fmla v22.4s, v30.4s, v7.s[1]\n"
    "fmla v24.4s, v30.4s, v9.s[1]\n"
    "fmla v26.4s, v30.4s, v11.s[1]\n"
    "fmla v28.4s, v30.4s, v13.s[1]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 2
    "fmla v15.4s, v29.4s, v2.s[1]\n"
    "fmla v17.4s, v29.4s, v4.s[1]\n"
    "fmla v19.4s, v29.4s, v6.s[1]\n"
    "fmla v21.4s, v29.4s, v8.s[1]\n"
    "fmla v23.4s, v29.4s, v10.s[1]\n"
    "fmla v25.4s, v29.4s, v12.s[1]\n"
    "fmla v27.4s, v29.4s, v14.s[1]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v2.s[1]\n"
    "fmla v18.4s, v30.4s, v4.s[1]\n"
    "fmla v20.4s, v30.4s, v6.s[1]\n"
    "fmla v22.4s, v30.4s, v8.s[1]\n"
    "fmla v24.4s, v30.4s, v10.s[1]\n"
    "fmla v26.4s, v30.4s, v12.s[1]\n"
    "fmla v28.4s, v30.4s, v14.s[1]\n"
    "ldr q30, [x11], #16\n"
// K index 2
// Filter width idx 0
    "fmla v17.4s, v29.4s, v2.s[2]\n"
    "fmla v19.4s, v29.4s, v4.s[2]\n"
    "fmla v21.4s, v29.4s, v6.s[2]\n"
    "fmla v23.4s, v29.4s, v8.s[2]\n"
    "fmla v25.4s, v29.4s, v10.s[2]\n"
    "fmla v27.4s, v29.4s, v12.s[2]\n"
    "ldr q29, [x11], #16\n"
    "fmla v18.4s, v30.4s, v2.s[2]\n"
    "fmla v20.4s, v30.4s, v4.s[2]\n"
    "fmla v22.4s, v30.4s, v6.s[2]\n"
    "fmla v24.4s, v30.4s, v8.s[2]\n"
    "fmla v26.4s, v30.4s, v10.s[2]\n"
    "fmla v28.4s, v30.4s, v12.s[2]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 1
    "fmla v15.4s, v29.4s, v1.s[2]\n"
    "fmla v17.4s, v29.4s, v3.s[2]\n"
    "fmla v19.4s, v29.4s, v5.s[2]\n"
    "fmla v21.4s, v29.4s, v7.s[2]\n"
    "fmla v23.4s, v29.4s, v9.s[2]\n"
    "fmla v25.4s, v29.4s, v11.s[2]\n"
    "fmla v27.4s, v29.4s, v13.s[2]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v1.s[2]\n"
    "fmla v18.4s, v30.4s, v3.s[2]\n"
    "fmla v20.4s, v30.4s, v5.s[2]\n"
    "fmla v22.4s, v30.4s, v7.s[2]\n"
    "fmla v24.4s, v30.4s, v9.s[2]\n"
    "fmla v26.4s, v30.4s, v11.s[2]\n"
    "fmla v28.4s, v30.4s, v13.s[2]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 2
    "fmla v15.4s, v29.4s, v2.s[2]\n"
    "fmla v17.4s, v29.4s, v4.s[2]\n"
    "fmla v19.4s, v29.4s, v6.s[2]\n"
    "fmla v21.4s, v29.4s, v8.s[2]\n"
    "fmla v23.4s, v29.4s, v10.s[2]\n"
    "fmla v25.4s, v29.4s, v12.s[2]\n"
    "fmla v27.4s, v29.4s, v14.s[2]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v2.s[2]\n"
    "fmla v18.4s, v30.4s, v4.s[2]\n"
    "fmla v20.4s, v30.4s, v6.s[2]\n"
    "fmla v22.4s, v30.4s, v8.s[2]\n"
    "fmla v24.4s, v30.4s, v10.s[2]\n"
    "fmla v26.4s, v30.4s, v12.s[2]\n"
    "fmla v28.4s, v30.4s, v14.s[2]\n"
    "ldr q30, [x11], #16\n"
    "stp q15, q16, [%[out], #0]\n"
    "stp q17, q18, [%[out], #32]\n"
    "stp q19, q20, [%[out], #64]\n"
    "stp q21, q22, [%[out], #96]\n"
    "stp q23, q24, [%[out], #128]\n"
    "stp q25, q26, [%[out], #160]\n"
    "stp q27, q28, [%[out], #192]\n"
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
void kernel_3_7_8_2_1_0_1_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 10], [10, 11, 12], [12, 13, 14]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Duplicate index: 6
// [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1]
// Number of Input Regs: 15, Filter Regs: 2 Output Regs: 14
// Total number of registers required: 31
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Out - [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
// Fil - [29, 30, 29, 30, 29, 30, 29, 30]
// Register mapping diagram
//           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 
//
//29 29 29  15    17    19    21    23    25    27    
//30 30 30  16    18    20    22    24    26    28    
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*7 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*8 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*9 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*10 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*11 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*12 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*13 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*14 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
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
    "mov x11, %[fil]\n"
// Load Output
    "ldp q15, q16, [%[out], #0]\n"
    "ldp q17, q18, [%[out], #32]\n"
    "ldp q19, q20, [%[out], #64]\n"
    "ldp q21, q22, [%[out], #96]\n"
    "ldp q23, q24, [%[out], #128]\n"
    "ldp q25, q26, [%[out], #160]\n"
    "ldp q27, q28, [%[out], #192]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"
    "ldr q5, [x10, #60]\n"
    "ldr q6, [x10, #72]\n"
    "ldr q7, [x10, #84]\n"
    "ldr q8, [x10, #96]\n"
    "ldr q9, [x10, #108]\n"
    "ldr q10, [x10, #120]\n"
    "ldr q11, [x10, #132]\n"
    "ldr q12, [x10, #144]\n"
    "ldr q13, [x10, #156]\n"

// Remaining 3 channels
// K index 0
// Filter width idx 0
    "fmla v15.4s, v29.4s, v0.s[0]\n"
    "fmla v17.4s, v29.4s, v2.s[0]\n"
    "fmla v19.4s, v29.4s, v4.s[0]\n"
    "fmla v21.4s, v29.4s, v6.s[0]\n"
    "fmla v23.4s, v29.4s, v8.s[0]\n"
    "fmla v25.4s, v29.4s, v10.s[0]\n"
    "fmla v27.4s, v29.4s, v12.s[0]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v0.s[0]\n"
    "fmla v18.4s, v30.4s, v2.s[0]\n"
    "fmla v20.4s, v30.4s, v4.s[0]\n"
    "fmla v22.4s, v30.4s, v6.s[0]\n"
    "fmla v24.4s, v30.4s, v8.s[0]\n"
    "fmla v26.4s, v30.4s, v10.s[0]\n"
    "fmla v28.4s, v30.4s, v12.s[0]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 1
    "fmla v15.4s, v29.4s, v1.s[0]\n"
    "fmla v17.4s, v29.4s, v3.s[0]\n"
    "fmla v19.4s, v29.4s, v5.s[0]\n"
    "fmla v21.4s, v29.4s, v7.s[0]\n"
    "fmla v23.4s, v29.4s, v9.s[0]\n"
    "fmla v25.4s, v29.4s, v11.s[0]\n"
    "fmla v27.4s, v29.4s, v13.s[0]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v1.s[0]\n"
    "fmla v18.4s, v30.4s, v3.s[0]\n"
    "fmla v20.4s, v30.4s, v5.s[0]\n"
    "fmla v22.4s, v30.4s, v7.s[0]\n"
    "fmla v24.4s, v30.4s, v9.s[0]\n"
    "fmla v26.4s, v30.4s, v11.s[0]\n"
    "fmla v28.4s, v30.4s, v13.s[0]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 2
    "fmla v15.4s, v29.4s, v2.s[0]\n"
    "fmla v17.4s, v29.4s, v4.s[0]\n"
    "fmla v19.4s, v29.4s, v6.s[0]\n"
    "fmla v21.4s, v29.4s, v8.s[0]\n"
    "fmla v23.4s, v29.4s, v10.s[0]\n"
    "fmla v25.4s, v29.4s, v12.s[0]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v2.s[0]\n"
    "fmla v18.4s, v30.4s, v4.s[0]\n"
    "fmla v20.4s, v30.4s, v6.s[0]\n"
    "fmla v22.4s, v30.4s, v8.s[0]\n"
    "fmla v24.4s, v30.4s, v10.s[0]\n"
    "fmla v26.4s, v30.4s, v12.s[0]\n"
    "ldr q30, [x11], #16\n"
// K index 1
// Filter width idx 0
    "fmla v15.4s, v29.4s, v0.s[1]\n"
    "fmla v17.4s, v29.4s, v2.s[1]\n"
    "fmla v19.4s, v29.4s, v4.s[1]\n"
    "fmla v21.4s, v29.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v8.s[1]\n"
    "fmla v25.4s, v29.4s, v10.s[1]\n"
    "fmla v27.4s, v29.4s, v12.s[1]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v0.s[1]\n"
    "fmla v18.4s, v30.4s, v2.s[1]\n"
    "fmla v20.4s, v30.4s, v4.s[1]\n"
    "fmla v22.4s, v30.4s, v6.s[1]\n"
    "fmla v24.4s, v30.4s, v8.s[1]\n"
    "fmla v26.4s, v30.4s, v10.s[1]\n"
    "fmla v28.4s, v30.4s, v12.s[1]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 1
    "fmla v15.4s, v29.4s, v1.s[1]\n"
    "fmla v17.4s, v29.4s, v3.s[1]\n"
    "fmla v19.4s, v29.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v7.s[1]\n"
    "fmla v23.4s, v29.4s, v9.s[1]\n"
    "fmla v25.4s, v29.4s, v11.s[1]\n"
    "fmla v27.4s, v29.4s, v13.s[1]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v1.s[1]\n"
    "fmla v18.4s, v30.4s, v3.s[1]\n"
    "fmla v20.4s, v30.4s, v5.s[1]\n"
    "fmla v22.4s, v30.4s, v7.s[1]\n"
    "fmla v24.4s, v30.4s, v9.s[1]\n"
    "fmla v26.4s, v30.4s, v11.s[1]\n"
    "fmla v28.4s, v30.4s, v13.s[1]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 2
    "fmla v15.4s, v29.4s, v2.s[1]\n"
    "fmla v17.4s, v29.4s, v4.s[1]\n"
    "fmla v19.4s, v29.4s, v6.s[1]\n"
    "fmla v21.4s, v29.4s, v8.s[1]\n"
    "fmla v23.4s, v29.4s, v10.s[1]\n"
    "fmla v25.4s, v29.4s, v12.s[1]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v2.s[1]\n"
    "fmla v18.4s, v30.4s, v4.s[1]\n"
    "fmla v20.4s, v30.4s, v6.s[1]\n"
    "fmla v22.4s, v30.4s, v8.s[1]\n"
    "fmla v24.4s, v30.4s, v10.s[1]\n"
    "fmla v26.4s, v30.4s, v12.s[1]\n"
    "ldr q30, [x11], #16\n"
// K index 2
// Filter width idx 0
    "fmla v15.4s, v29.4s, v0.s[2]\n"
    "fmla v17.4s, v29.4s, v2.s[2]\n"
    "fmla v19.4s, v29.4s, v4.s[2]\n"
    "fmla v21.4s, v29.4s, v6.s[2]\n"
    "fmla v23.4s, v29.4s, v8.s[2]\n"
    "fmla v25.4s, v29.4s, v10.s[2]\n"
    "fmla v27.4s, v29.4s, v12.s[2]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v0.s[2]\n"
    "fmla v18.4s, v30.4s, v2.s[2]\n"
    "fmla v20.4s, v30.4s, v4.s[2]\n"
    "fmla v22.4s, v30.4s, v6.s[2]\n"
    "fmla v24.4s, v30.4s, v8.s[2]\n"
    "fmla v26.4s, v30.4s, v10.s[2]\n"
    "fmla v28.4s, v30.4s, v12.s[2]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 1
    "fmla v15.4s, v29.4s, v1.s[2]\n"
    "fmla v17.4s, v29.4s, v3.s[2]\n"
    "fmla v19.4s, v29.4s, v5.s[2]\n"
    "fmla v21.4s, v29.4s, v7.s[2]\n"
    "fmla v23.4s, v29.4s, v9.s[2]\n"
    "fmla v25.4s, v29.4s, v11.s[2]\n"
    "fmla v27.4s, v29.4s, v13.s[2]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v1.s[2]\n"
    "fmla v18.4s, v30.4s, v3.s[2]\n"
    "fmla v20.4s, v30.4s, v5.s[2]\n"
    "fmla v22.4s, v30.4s, v7.s[2]\n"
    "fmla v24.4s, v30.4s, v9.s[2]\n"
    "fmla v26.4s, v30.4s, v11.s[2]\n"
    "fmla v28.4s, v30.4s, v13.s[2]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 2
    "fmla v15.4s, v29.4s, v2.s[2]\n"
    "fmla v17.4s, v29.4s, v4.s[2]\n"
    "fmla v19.4s, v29.4s, v6.s[2]\n"
    "fmla v21.4s, v29.4s, v8.s[2]\n"
    "fmla v23.4s, v29.4s, v10.s[2]\n"
    "fmla v25.4s, v29.4s, v12.s[2]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v2.s[2]\n"
    "fmla v18.4s, v30.4s, v4.s[2]\n"
    "fmla v20.4s, v30.4s, v6.s[2]\n"
    "fmla v22.4s, v30.4s, v8.s[2]\n"
    "fmla v24.4s, v30.4s, v10.s[2]\n"
    "fmla v26.4s, v30.4s, v12.s[2]\n"
    "ldr q30, [x11], #16\n"
    "stp q15, q16, [%[out], #0]\n"
    "stp q17, q18, [%[out], #32]\n"
    "stp q19, q20, [%[out], #64]\n"
    "stp q21, q22, [%[out], #96]\n"
    "stp q23, q24, [%[out], #128]\n"
    "stp q25, q26, [%[out], #160]\n"
    "stp q27, q28, [%[out], #192]\n"
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
void kernel_3_7_8_2_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 10], [10, 11, 12], [12, 13, 14]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Duplicate index: 6
// [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1]
// Number of Input Regs: 15, Filter Regs: 2 Output Regs: 14
// Total number of registers required: 31
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Out - [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
// Fil - [29, 30, 29, 30, 29, 30, 29, 30]
// Register mapping diagram
//           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 
//
//29 29 29  15    17    19    21    23    25    27    
//30 30 30  16    18    20    22    24    26    28    
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", i*8 + j);
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*7 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*8 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*9 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*10 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*11 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*12 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*13 + j));
            printf("%6.3f\t", *(inputPtr + i*inStride + 3*14 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
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
    "mov x11, %[fil]\n"
// Load Output
    "ldp q15, q16, [%[out], #0]\n"
    "ldp q17, q18, [%[out], #32]\n"
    "ldp q19, q20, [%[out], #64]\n"
    "ldp q21, q22, [%[out], #96]\n"
    "ldp q23, q24, [%[out], #128]\n"
    "ldp q25, q26, [%[out], #160]\n"
    "ldp q27, q28, [%[out], #192]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"
    "ldr q5, [x10, #60]\n"
    "ldr q6, [x10, #72]\n"
    "ldr q7, [x10, #84]\n"
    "ldr q8, [x10, #96]\n"
    "ldr q9, [x10, #108]\n"
    "ldr q10, [x10, #120]\n"
    "ldr q11, [x10, #132]\n"
    "ldr q12, [x10, #144]\n"
    "ldr q13, [x10, #156]\n"
    "ldr q14, [x10, #168]\n"

// Remaining 3 channels
// K index 0
// Filter width idx 0
    "fmla v15.4s, v29.4s, v0.s[0]\n"
    "fmla v17.4s, v29.4s, v2.s[0]\n"
    "fmla v19.4s, v29.4s, v4.s[0]\n"
    "fmla v21.4s, v29.4s, v6.s[0]\n"
    "fmla v23.4s, v29.4s, v8.s[0]\n"
    "fmla v25.4s, v29.4s, v10.s[0]\n"
    "fmla v27.4s, v29.4s, v12.s[0]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v0.s[0]\n"
    "fmla v18.4s, v30.4s, v2.s[0]\n"
    "fmla v20.4s, v30.4s, v4.s[0]\n"
    "fmla v22.4s, v30.4s, v6.s[0]\n"
    "fmla v24.4s, v30.4s, v8.s[0]\n"
    "fmla v26.4s, v30.4s, v10.s[0]\n"
    "fmla v28.4s, v30.4s, v12.s[0]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 1
    "fmla v15.4s, v29.4s, v1.s[0]\n"
    "fmla v17.4s, v29.4s, v3.s[0]\n"
    "fmla v19.4s, v29.4s, v5.s[0]\n"
    "fmla v21.4s, v29.4s, v7.s[0]\n"
    "fmla v23.4s, v29.4s, v9.s[0]\n"
    "fmla v25.4s, v29.4s, v11.s[0]\n"
    "fmla v27.4s, v29.4s, v13.s[0]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v1.s[0]\n"
    "fmla v18.4s, v30.4s, v3.s[0]\n"
    "fmla v20.4s, v30.4s, v5.s[0]\n"
    "fmla v22.4s, v30.4s, v7.s[0]\n"
    "fmla v24.4s, v30.4s, v9.s[0]\n"
    "fmla v26.4s, v30.4s, v11.s[0]\n"
    "fmla v28.4s, v30.4s, v13.s[0]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 2
    "fmla v15.4s, v29.4s, v2.s[0]\n"
    "fmla v17.4s, v29.4s, v4.s[0]\n"
    "fmla v19.4s, v29.4s, v6.s[0]\n"
    "fmla v21.4s, v29.4s, v8.s[0]\n"
    "fmla v23.4s, v29.4s, v10.s[0]\n"
    "fmla v25.4s, v29.4s, v12.s[0]\n"
    "fmla v27.4s, v29.4s, v14.s[0]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v2.s[0]\n"
    "fmla v18.4s, v30.4s, v4.s[0]\n"
    "fmla v20.4s, v30.4s, v6.s[0]\n"
    "fmla v22.4s, v30.4s, v8.s[0]\n"
    "fmla v24.4s, v30.4s, v10.s[0]\n"
    "fmla v26.4s, v30.4s, v12.s[0]\n"
    "fmla v28.4s, v30.4s, v14.s[0]\n"
    "ldr q30, [x11], #16\n"
// K index 1
// Filter width idx 0
    "fmla v15.4s, v29.4s, v0.s[1]\n"
    "fmla v17.4s, v29.4s, v2.s[1]\n"
    "fmla v19.4s, v29.4s, v4.s[1]\n"
    "fmla v21.4s, v29.4s, v6.s[1]\n"
    "fmla v23.4s, v29.4s, v8.s[1]\n"
    "fmla v25.4s, v29.4s, v10.s[1]\n"
    "fmla v27.4s, v29.4s, v12.s[1]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v0.s[1]\n"
    "fmla v18.4s, v30.4s, v2.s[1]\n"
    "fmla v20.4s, v30.4s, v4.s[1]\n"
    "fmla v22.4s, v30.4s, v6.s[1]\n"
    "fmla v24.4s, v30.4s, v8.s[1]\n"
    "fmla v26.4s, v30.4s, v10.s[1]\n"
    "fmla v28.4s, v30.4s, v12.s[1]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 1
    "fmla v15.4s, v29.4s, v1.s[1]\n"
    "fmla v17.4s, v29.4s, v3.s[1]\n"
    "fmla v19.4s, v29.4s, v5.s[1]\n"
    "fmla v21.4s, v29.4s, v7.s[1]\n"
    "fmla v23.4s, v29.4s, v9.s[1]\n"
    "fmla v25.4s, v29.4s, v11.s[1]\n"
    "fmla v27.4s, v29.4s, v13.s[1]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v1.s[1]\n"
    "fmla v18.4s, v30.4s, v3.s[1]\n"
    "fmla v20.4s, v30.4s, v5.s[1]\n"
    "fmla v22.4s, v30.4s, v7.s[1]\n"
    "fmla v24.4s, v30.4s, v9.s[1]\n"
    "fmla v26.4s, v30.4s, v11.s[1]\n"
    "fmla v28.4s, v30.4s, v13.s[1]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 2
    "fmla v15.4s, v29.4s, v2.s[1]\n"
    "fmla v17.4s, v29.4s, v4.s[1]\n"
    "fmla v19.4s, v29.4s, v6.s[1]\n"
    "fmla v21.4s, v29.4s, v8.s[1]\n"
    "fmla v23.4s, v29.4s, v10.s[1]\n"
    "fmla v25.4s, v29.4s, v12.s[1]\n"
    "fmla v27.4s, v29.4s, v14.s[1]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v2.s[1]\n"
    "fmla v18.4s, v30.4s, v4.s[1]\n"
    "fmla v20.4s, v30.4s, v6.s[1]\n"
    "fmla v22.4s, v30.4s, v8.s[1]\n"
    "fmla v24.4s, v30.4s, v10.s[1]\n"
    "fmla v26.4s, v30.4s, v12.s[1]\n"
    "fmla v28.4s, v30.4s, v14.s[1]\n"
    "ldr q30, [x11], #16\n"
// K index 2
// Filter width idx 0
    "fmla v15.4s, v29.4s, v0.s[2]\n"
    "fmla v17.4s, v29.4s, v2.s[2]\n"
    "fmla v19.4s, v29.4s, v4.s[2]\n"
    "fmla v21.4s, v29.4s, v6.s[2]\n"
    "fmla v23.4s, v29.4s, v8.s[2]\n"
    "fmla v25.4s, v29.4s, v10.s[2]\n"
    "fmla v27.4s, v29.4s, v12.s[2]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v0.s[2]\n"
    "fmla v18.4s, v30.4s, v2.s[2]\n"
    "fmla v20.4s, v30.4s, v4.s[2]\n"
    "fmla v22.4s, v30.4s, v6.s[2]\n"
    "fmla v24.4s, v30.4s, v8.s[2]\n"
    "fmla v26.4s, v30.4s, v10.s[2]\n"
    "fmla v28.4s, v30.4s, v12.s[2]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 1
    "fmla v15.4s, v29.4s, v1.s[2]\n"
    "fmla v17.4s, v29.4s, v3.s[2]\n"
    "fmla v19.4s, v29.4s, v5.s[2]\n"
    "fmla v21.4s, v29.4s, v7.s[2]\n"
    "fmla v23.4s, v29.4s, v9.s[2]\n"
    "fmla v25.4s, v29.4s, v11.s[2]\n"
    "fmla v27.4s, v29.4s, v13.s[2]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v1.s[2]\n"
    "fmla v18.4s, v30.4s, v3.s[2]\n"
    "fmla v20.4s, v30.4s, v5.s[2]\n"
    "fmla v22.4s, v30.4s, v7.s[2]\n"
    "fmla v24.4s, v30.4s, v9.s[2]\n"
    "fmla v26.4s, v30.4s, v11.s[2]\n"
    "fmla v28.4s, v30.4s, v13.s[2]\n"
    "ldr q30, [x11], #16\n"
// Filter width idx 2
    "fmla v15.4s, v29.4s, v2.s[2]\n"
    "fmla v17.4s, v29.4s, v4.s[2]\n"
    "fmla v19.4s, v29.4s, v6.s[2]\n"
    "fmla v21.4s, v29.4s, v8.s[2]\n"
    "fmla v23.4s, v29.4s, v10.s[2]\n"
    "fmla v25.4s, v29.4s, v12.s[2]\n"
    "fmla v27.4s, v29.4s, v14.s[2]\n"
    "ldr q29, [x11], #16\n"
    "fmla v16.4s, v30.4s, v2.s[2]\n"
    "fmla v18.4s, v30.4s, v4.s[2]\n"
    "fmla v20.4s, v30.4s, v6.s[2]\n"
    "fmla v22.4s, v30.4s, v8.s[2]\n"
    "fmla v24.4s, v30.4s, v10.s[2]\n"
    "fmla v26.4s, v30.4s, v12.s[2]\n"
    "fmla v28.4s, v30.4s, v14.s[2]\n"
    "ldr q30, [x11], #16\n"
    "stp q15, q16, [%[out], #0]\n"
    "stp q17, q18, [%[out], #32]\n"
    "stp q19, q20, [%[out], #64]\n"
    "stp q21, q22, [%[out], #96]\n"
    "stp q23, q24, [%[out], #128]\n"
    "stp q25, q26, [%[out], #160]\n"
    "stp q27, q28, [%[out], #192]\n"
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
void kernel_3_6_8_1_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7]
// Duplicate index: 10
// [1, 2, 3, 3, 3, 3, 2, 1]
// Number of Input Regs: 8, Filter Regs: 4 Output Regs: 12
// Total number of registers required: 24
// In  - [0, 1, 2, 3, 4, 5, 6, 7]
// Out - [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
// Fil - [20, 21, 22, 23, 20, 21, 22, 23, 20, 21]
// Register mapping diagram
//           0  1  2  3  4  5  6  7 
//
//20 22 20   8 10 12 14 16 18 
//21 23 21   9 11 13 15 17 19 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + 3*6 + j));
            printf("%6.3f\t", *(inputPtr + 3*7 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "prfm pldl1keep, [%[in], #0]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q8, q9, [%[out], #0]\n"
    "ldp q10, q11, [%[out], #32]\n"
    "ldp q12, q13, [%[out], #64]\n"
    "ldp q14, q15, [%[out], #96]\n"
    "ldp q16, q17, [%[out], #128]\n"
    "ldp q18, q19, [%[out], #160]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
    "ld1 {v22.4s - v23.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"
    "ldr q5, [x10, #60]\n"
    "ldr q6, [x10, #72]\n"
    "ldr q7, [x10, #84]\n"

// K index 0
// Filter width idx 0
    "fmla v8.4s, v20.4s, v0.s[0]\n"
    "fmla v9.4s, v21.4s, v0.s[0]\n"
    "fmla v10.4s, v20.4s, v1.s[0]\n"
    "fmla v11.4s, v21.4s, v1.s[0]\n"
    "fmla v12.4s, v20.4s, v2.s[0]\n"
    "fmla v13.4s, v21.4s, v2.s[0]\n"
    "fmla v14.4s, v20.4s, v3.s[0]\n"
    "fmla v15.4s, v21.4s, v3.s[0]\n"
    "fmla v16.4s, v20.4s, v4.s[0]\n"
    "fmla v17.4s, v21.4s, v4.s[0]\n"
    "fmla v18.4s, v20.4s, v5.s[0]\n"
    "fmla v19.4s, v21.4s, v5.s[0]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v8.4s, v22.4s, v1.s[0]\n"
    "fmla v9.4s, v23.4s, v1.s[0]\n"
    "fmla v10.4s, v22.4s, v2.s[0]\n"
    "fmla v11.4s, v23.4s, v2.s[0]\n"
    "fmla v12.4s, v22.4s, v3.s[0]\n"
    "fmla v13.4s, v23.4s, v3.s[0]\n"
    "fmla v14.4s, v22.4s, v4.s[0]\n"
    "fmla v15.4s, v23.4s, v4.s[0]\n"
    "fmla v16.4s, v22.4s, v5.s[0]\n"
    "fmla v17.4s, v23.4s, v5.s[0]\n"
    "fmla v18.4s, v22.4s, v6.s[0]\n"
    "fmla v19.4s, v23.4s, v6.s[0]\n"
    "ld1 {v22.4s - v23.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v8.4s, v20.4s, v2.s[0]\n"
    "fmla v9.4s, v21.4s, v2.s[0]\n"
    "fmla v10.4s, v20.4s, v3.s[0]\n"
    "fmla v11.4s, v21.4s, v3.s[0]\n"
    "fmla v12.4s, v20.4s, v4.s[0]\n"
    "fmla v13.4s, v21.4s, v4.s[0]\n"
    "fmla v14.4s, v20.4s, v5.s[0]\n"
    "fmla v15.4s, v21.4s, v5.s[0]\n"
    "fmla v16.4s, v20.4s, v6.s[0]\n"
    "fmla v17.4s, v21.4s, v6.s[0]\n"
    "fmla v18.4s, v20.4s, v7.s[0]\n"
    "fmla v19.4s, v21.4s, v7.s[0]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v8.4s, v22.4s, v0.s[1]\n"
    "fmla v9.4s, v23.4s, v0.s[1]\n"
    "fmla v10.4s, v22.4s, v1.s[1]\n"
    "fmla v11.4s, v23.4s, v1.s[1]\n"
    "fmla v12.4s, v22.4s, v2.s[1]\n"
    "fmla v13.4s, v23.4s, v2.s[1]\n"
    "fmla v14.4s, v22.4s, v3.s[1]\n"
    "fmla v15.4s, v23.4s, v3.s[1]\n"
    "fmla v16.4s, v22.4s, v4.s[1]\n"
    "fmla v17.4s, v23.4s, v4.s[1]\n"
    "fmla v18.4s, v22.4s, v5.s[1]\n"
    "fmla v19.4s, v23.4s, v5.s[1]\n"
    "ld1 {v22.4s - v23.4s}, [x11], #32\n"
// Filter width idx 1
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
    "fmla v18.4s, v20.4s, v6.s[1]\n"
    "fmla v19.4s, v21.4s, v6.s[1]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v8.4s, v22.4s, v2.s[1]\n"
    "fmla v9.4s, v23.4s, v2.s[1]\n"
    "fmla v10.4s, v22.4s, v3.s[1]\n"
    "fmla v11.4s, v23.4s, v3.s[1]\n"
    "fmla v12.4s, v22.4s, v4.s[1]\n"
    "fmla v13.4s, v23.4s, v4.s[1]\n"
    "fmla v14.4s, v22.4s, v5.s[1]\n"
    "fmla v15.4s, v23.4s, v5.s[1]\n"
    "fmla v16.4s, v22.4s, v6.s[1]\n"
    "fmla v17.4s, v23.4s, v6.s[1]\n"
    "fmla v18.4s, v22.4s, v7.s[1]\n"
    "fmla v19.4s, v23.4s, v7.s[1]\n"
    "ld1 {v22.4s - v23.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v8.4s, v20.4s, v0.s[2]\n"
    "fmla v9.4s, v21.4s, v0.s[2]\n"
    "fmla v10.4s, v20.4s, v1.s[2]\n"
    "fmla v11.4s, v21.4s, v1.s[2]\n"
    "fmla v12.4s, v20.4s, v2.s[2]\n"
    "fmla v13.4s, v21.4s, v2.s[2]\n"
    "fmla v14.4s, v20.4s, v3.s[2]\n"
    "fmla v15.4s, v21.4s, v3.s[2]\n"
    "fmla v16.4s, v20.4s, v4.s[2]\n"
    "fmla v17.4s, v21.4s, v4.s[2]\n"
    "fmla v18.4s, v20.4s, v5.s[2]\n"
    "fmla v19.4s, v21.4s, v5.s[2]\n"
    "ld1 {v20.4s - v21.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v8.4s, v22.4s, v1.s[2]\n"
    "fmla v9.4s, v23.4s, v1.s[2]\n"
    "fmla v10.4s, v22.4s, v2.s[2]\n"
    "fmla v11.4s, v23.4s, v2.s[2]\n"
    "fmla v12.4s, v22.4s, v3.s[2]\n"
    "fmla v13.4s, v23.4s, v3.s[2]\n"
    "fmla v14.4s, v22.4s, v4.s[2]\n"
    "fmla v15.4s, v23.4s, v4.s[2]\n"
    "fmla v16.4s, v22.4s, v5.s[2]\n"
    "fmla v17.4s, v23.4s, v5.s[2]\n"
    "fmla v18.4s, v22.4s, v6.s[2]\n"
    "fmla v19.4s, v23.4s, v6.s[2]\n"
// Filter width idx 2
    "fmla v8.4s, v20.4s, v2.s[2]\n"
    "fmla v9.4s, v21.4s, v2.s[2]\n"
    "fmla v10.4s, v20.4s, v3.s[2]\n"
    "fmla v11.4s, v21.4s, v3.s[2]\n"
    "fmla v12.4s, v20.4s, v4.s[2]\n"
    "fmla v13.4s, v21.4s, v4.s[2]\n"
    "fmla v14.4s, v20.4s, v5.s[2]\n"
    "fmla v15.4s, v21.4s, v5.s[2]\n"
    "fmla v16.4s, v20.4s, v6.s[2]\n"
    "fmla v17.4s, v21.4s, v6.s[2]\n"
    "fmla v18.4s, v20.4s, v7.s[2]\n"
    "fmla v19.4s, v21.4s, v7.s[2]\n"
    "stp q8, q9, [%[out], #0]\n"
    "stp q10, q11, [%[out], #32]\n"
    "stp q12, q13, [%[out], #64]\n"
    "stp q14, q15, [%[out], #96]\n"
    "stp q16, q17, [%[out], #128]\n"
    "stp q18, q19, [%[out], #160]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_5_8_1_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6]
// Duplicate index: 8
// [1, 2, 3, 3, 3, 2, 1]
// Number of Input Regs: 7, Filter Regs: 6 Output Regs: 10
// Total number of registers required: 23
// In  - [0, 1, 2, 3, 4, 5, 6]
// Out - [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
// Fil - [17, 18, 19, 20, 21, 22, 17, 18, 19, 20, 21, 22]
// Register mapping diagram
//           0  1  2  3  4  5  6 
//
//17 19 21   7  9 11 13 15 
//18 20 22   8 10 12 14 16 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + 3*5 + j));
            printf("%6.3f\t", *(inputPtr + 3*6 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q7, q8, [%[out], #0]\n"
    "ldp q9, q10, [%[out], #32]\n"
    "ldp q11, q12, [%[out], #64]\n"
    "ldp q13, q14, [%[out], #96]\n"
    "ldp q15, q16, [%[out], #128]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"
    "ldr q5, [x10, #60]\n"
    "ldr q6, [x10, #72]\n"

// K index 0
// Filter width idx 0
    "fmla v7.4s, v17.4s, v0.s[0]\n"
    "fmla v8.4s, v18.4s, v0.s[0]\n"
    "fmla v9.4s, v17.4s, v1.s[0]\n"
    "fmla v10.4s, v18.4s, v1.s[0]\n"
    "fmla v11.4s, v17.4s, v2.s[0]\n"
    "fmla v12.4s, v18.4s, v2.s[0]\n"
    "fmla v13.4s, v17.4s, v3.s[0]\n"
    "fmla v14.4s, v18.4s, v3.s[0]\n"
    "fmla v15.4s, v17.4s, v4.s[0]\n"
    "fmla v16.4s, v18.4s, v4.s[0]\n"
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v7.4s, v19.4s, v1.s[0]\n"
    "fmla v8.4s, v20.4s, v1.s[0]\n"
    "fmla v9.4s, v19.4s, v2.s[0]\n"
    "fmla v10.4s, v20.4s, v2.s[0]\n"
    "fmla v11.4s, v19.4s, v3.s[0]\n"
    "fmla v12.4s, v20.4s, v3.s[0]\n"
    "fmla v13.4s, v19.4s, v4.s[0]\n"
    "fmla v14.4s, v20.4s, v4.s[0]\n"
    "fmla v15.4s, v19.4s, v5.s[0]\n"
    "fmla v16.4s, v20.4s, v5.s[0]\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v7.4s, v21.4s, v2.s[0]\n"
    "fmla v8.4s, v22.4s, v2.s[0]\n"
    "fmla v9.4s, v21.4s, v3.s[0]\n"
    "fmla v10.4s, v22.4s, v3.s[0]\n"
    "fmla v11.4s, v21.4s, v4.s[0]\n"
    "fmla v12.4s, v22.4s, v4.s[0]\n"
    "fmla v13.4s, v21.4s, v5.s[0]\n"
    "fmla v14.4s, v22.4s, v5.s[0]\n"
    "fmla v15.4s, v21.4s, v6.s[0]\n"
    "fmla v16.4s, v22.4s, v6.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v7.4s, v17.4s, v0.s[1]\n"
    "fmla v8.4s, v18.4s, v0.s[1]\n"
    "fmla v9.4s, v17.4s, v1.s[1]\n"
    "fmla v10.4s, v18.4s, v1.s[1]\n"
    "fmla v11.4s, v17.4s, v2.s[1]\n"
    "fmla v12.4s, v18.4s, v2.s[1]\n"
    "fmla v13.4s, v17.4s, v3.s[1]\n"
    "fmla v14.4s, v18.4s, v3.s[1]\n"
    "fmla v15.4s, v17.4s, v4.s[1]\n"
    "fmla v16.4s, v18.4s, v4.s[1]\n"
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v7.4s, v19.4s, v1.s[1]\n"
    "fmla v8.4s, v20.4s, v1.s[1]\n"
    "fmla v9.4s, v19.4s, v2.s[1]\n"
    "fmla v10.4s, v20.4s, v2.s[1]\n"
    "fmla v11.4s, v19.4s, v3.s[1]\n"
    "fmla v12.4s, v20.4s, v3.s[1]\n"
    "fmla v13.4s, v19.4s, v4.s[1]\n"
    "fmla v14.4s, v20.4s, v4.s[1]\n"
    "fmla v15.4s, v19.4s, v5.s[1]\n"
    "fmla v16.4s, v20.4s, v5.s[1]\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v7.4s, v21.4s, v2.s[1]\n"
    "fmla v8.4s, v22.4s, v2.s[1]\n"
    "fmla v9.4s, v21.4s, v3.s[1]\n"
    "fmla v10.4s, v22.4s, v3.s[1]\n"
    "fmla v11.4s, v21.4s, v4.s[1]\n"
    "fmla v12.4s, v22.4s, v4.s[1]\n"
    "fmla v13.4s, v21.4s, v5.s[1]\n"
    "fmla v14.4s, v22.4s, v5.s[1]\n"
    "fmla v15.4s, v21.4s, v6.s[1]\n"
    "fmla v16.4s, v22.4s, v6.s[1]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v7.4s, v17.4s, v0.s[2]\n"
    "fmla v8.4s, v18.4s, v0.s[2]\n"
    "fmla v9.4s, v17.4s, v1.s[2]\n"
    "fmla v10.4s, v18.4s, v1.s[2]\n"
    "fmla v11.4s, v17.4s, v2.s[2]\n"
    "fmla v12.4s, v18.4s, v2.s[2]\n"
    "fmla v13.4s, v17.4s, v3.s[2]\n"
    "fmla v14.4s, v18.4s, v3.s[2]\n"
    "fmla v15.4s, v17.4s, v4.s[2]\n"
    "fmla v16.4s, v18.4s, v4.s[2]\n"
// Filter width idx 1
    "fmla v7.4s, v19.4s, v1.s[2]\n"
    "fmla v8.4s, v20.4s, v1.s[2]\n"
    "fmla v9.4s, v19.4s, v2.s[2]\n"
    "fmla v10.4s, v20.4s, v2.s[2]\n"
    "fmla v11.4s, v19.4s, v3.s[2]\n"
    "fmla v12.4s, v20.4s, v3.s[2]\n"
    "fmla v13.4s, v19.4s, v4.s[2]\n"
    "fmla v14.4s, v20.4s, v4.s[2]\n"
    "fmla v15.4s, v19.4s, v5.s[2]\n"
    "fmla v16.4s, v20.4s, v5.s[2]\n"
// Filter width idx 2
    "fmla v7.4s, v21.4s, v2.s[2]\n"
    "fmla v8.4s, v22.4s, v2.s[2]\n"
    "fmla v9.4s, v21.4s, v3.s[2]\n"
    "fmla v10.4s, v22.4s, v3.s[2]\n"
    "fmla v11.4s, v21.4s, v4.s[2]\n"
    "fmla v12.4s, v22.4s, v4.s[2]\n"
    "fmla v13.4s, v21.4s, v5.s[2]\n"
    "fmla v14.4s, v22.4s, v5.s[2]\n"
    "fmla v15.4s, v21.4s, v6.s[2]\n"
    "fmla v16.4s, v22.4s, v6.s[2]\n"
    "stp q7, q8, [%[out], #0]\n"
    "stp q9, q10, [%[out], #32]\n"
    "stp q11, q12, [%[out], #64]\n"
    "stp q13, q14, [%[out], #96]\n"
    "stp q15, q16, [%[out], #128]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_4_8_1_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
// Input registers required
// [0, 1, 2, 3, 4, 5]
// Duplicate index: 6
// [1, 2, 3, 3, 2, 1]
// Number of Input Regs: 6, Filter Regs: 6 Output Regs: 8
// Total number of registers required: 20
// In  - [0, 1, 2, 3, 4, 5]
// Out - [6, 7, 8, 9, 10, 11, 12, 13]
// Fil - [14, 15, 16, 17, 18, 19, 14, 15, 16, 17, 18, 19]
// Register mapping diagram
//           0  1  2  3  4  5 
//
//14 16 18   6  8 10 12 
//15 17 19   7  9 11 13 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf("%6.3f\t", *(inputPtr + 3*5 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q6, q7, [%[out], #0]\n"
    "ldp q8, q9, [%[out], #32]\n"
    "ldp q10, q11, [%[out], #64]\n"
    "ldp q12, q13, [%[out], #96]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v14.4s - v15.4s}, [x11], #32\n"
    "ld1 {v16.4s - v17.4s}, [x11], #32\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"
    "ldr q5, [x10, #60]\n"

// K index 0
// Filter width idx 0
    "fmla v6.4s, v14.4s, v0.s[0]\n"
    "fmla v7.4s, v15.4s, v0.s[0]\n"
    "fmla v8.4s, v14.4s, v1.s[0]\n"
    "fmla v9.4s, v15.4s, v1.s[0]\n"
    "fmla v10.4s, v14.4s, v2.s[0]\n"
    "fmla v11.4s, v15.4s, v2.s[0]\n"
    "fmla v12.4s, v14.4s, v3.s[0]\n"
    "fmla v13.4s, v15.4s, v3.s[0]\n"
    "ld1 {v14.4s - v15.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v6.4s, v16.4s, v1.s[0]\n"
    "fmla v7.4s, v17.4s, v1.s[0]\n"
    "fmla v8.4s, v16.4s, v2.s[0]\n"
    "fmla v9.4s, v17.4s, v2.s[0]\n"
    "fmla v10.4s, v16.4s, v3.s[0]\n"
    "fmla v11.4s, v17.4s, v3.s[0]\n"
    "fmla v12.4s, v16.4s, v4.s[0]\n"
    "fmla v13.4s, v17.4s, v4.s[0]\n"
    "ld1 {v16.4s - v17.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v6.4s, v18.4s, v2.s[0]\n"
    "fmla v7.4s, v19.4s, v2.s[0]\n"
    "fmla v8.4s, v18.4s, v3.s[0]\n"
    "fmla v9.4s, v19.4s, v3.s[0]\n"
    "fmla v10.4s, v18.4s, v4.s[0]\n"
    "fmla v11.4s, v19.4s, v4.s[0]\n"
    "fmla v12.4s, v18.4s, v5.s[0]\n"
    "fmla v13.4s, v19.4s, v5.s[0]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v6.4s, v14.4s, v0.s[1]\n"
    "fmla v7.4s, v15.4s, v0.s[1]\n"
    "fmla v8.4s, v14.4s, v1.s[1]\n"
    "fmla v9.4s, v15.4s, v1.s[1]\n"
    "fmla v10.4s, v14.4s, v2.s[1]\n"
    "fmla v11.4s, v15.4s, v2.s[1]\n"
    "fmla v12.4s, v14.4s, v3.s[1]\n"
    "fmla v13.4s, v15.4s, v3.s[1]\n"
    "ld1 {v14.4s - v15.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v6.4s, v16.4s, v1.s[1]\n"
    "fmla v7.4s, v17.4s, v1.s[1]\n"
    "fmla v8.4s, v16.4s, v2.s[1]\n"
    "fmla v9.4s, v17.4s, v2.s[1]\n"
    "fmla v10.4s, v16.4s, v3.s[1]\n"
    "fmla v11.4s, v17.4s, v3.s[1]\n"
    "fmla v12.4s, v16.4s, v4.s[1]\n"
    "fmla v13.4s, v17.4s, v4.s[1]\n"
    "ld1 {v16.4s - v17.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v6.4s, v18.4s, v2.s[1]\n"
    "fmla v7.4s, v19.4s, v2.s[1]\n"
    "fmla v8.4s, v18.4s, v3.s[1]\n"
    "fmla v9.4s, v19.4s, v3.s[1]\n"
    "fmla v10.4s, v18.4s, v4.s[1]\n"
    "fmla v11.4s, v19.4s, v4.s[1]\n"
    "fmla v12.4s, v18.4s, v5.s[1]\n"
    "fmla v13.4s, v19.4s, v5.s[1]\n"
    "ld1 {v18.4s - v19.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v6.4s, v14.4s, v0.s[2]\n"
    "fmla v7.4s, v15.4s, v0.s[2]\n"
    "fmla v8.4s, v14.4s, v1.s[2]\n"
    "fmla v9.4s, v15.4s, v1.s[2]\n"
    "fmla v10.4s, v14.4s, v2.s[2]\n"
    "fmla v11.4s, v15.4s, v2.s[2]\n"
    "fmla v12.4s, v14.4s, v3.s[2]\n"
    "fmla v13.4s, v15.4s, v3.s[2]\n"
// Filter width idx 1
    "fmla v6.4s, v16.4s, v1.s[2]\n"
    "fmla v7.4s, v17.4s, v1.s[2]\n"
    "fmla v8.4s, v16.4s, v2.s[2]\n"
    "fmla v9.4s, v17.4s, v2.s[2]\n"
    "fmla v10.4s, v16.4s, v3.s[2]\n"
    "fmla v11.4s, v17.4s, v3.s[2]\n"
    "fmla v12.4s, v16.4s, v4.s[2]\n"
    "fmla v13.4s, v17.4s, v4.s[2]\n"
// Filter width idx 2
    "fmla v6.4s, v18.4s, v2.s[2]\n"
    "fmla v7.4s, v19.4s, v2.s[2]\n"
    "fmla v8.4s, v18.4s, v3.s[2]\n"
    "fmla v9.4s, v19.4s, v3.s[2]\n"
    "fmla v10.4s, v18.4s, v4.s[2]\n"
    "fmla v11.4s, v19.4s, v4.s[2]\n"
    "fmla v12.4s, v18.4s, v5.s[2]\n"
    "fmla v13.4s, v19.4s, v5.s[2]\n"
    "stp q6, q7, [%[out], #0]\n"
    "stp q8, q9, [%[out], #32]\n"
    "stp q10, q11, [%[out], #64]\n"
    "stp q12, q13, [%[out], #96]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_3_8_1_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
// Input registers required
// [0, 1, 2, 3, 4]
// Duplicate index: 4
// [1, 2, 3, 2, 1]
// Number of Input Regs: 5, Filter Regs: 6 Output Regs: 6
// Total number of registers required: 17
// In  - [0, 1, 2, 3, 4]
// Out - [5, 6, 7, 8, 9, 10]
// Fil - [11, 12, 13, 14, 15, 16, 11, 12, 13, 14, 15, 16]
// Register mapping diagram
//           0  1  2  3  4 
//
//11 13 15   5  7  9 
//12 14 16   6  8 10 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf("%6.3f\t", *(inputPtr + 3*4 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q5, q6, [%[out], #0]\n"
    "ldp q7, q8, [%[out], #32]\n"
    "ldp q9, q10, [%[out], #64]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v11.4s - v12.4s}, [x11], #32\n"
    "ld1 {v13.4s - v14.4s}, [x11], #32\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"
    "ldr q4, [x10, #48]\n"

// K index 0
// Filter width idx 0
    "fmla v5.4s, v11.4s, v0.s[0]\n"
    "fmla v6.4s, v12.4s, v0.s[0]\n"
    "fmla v7.4s, v11.4s, v1.s[0]\n"
    "fmla v8.4s, v12.4s, v1.s[0]\n"
    "fmla v9.4s, v11.4s, v2.s[0]\n"
    "fmla v10.4s, v12.4s, v2.s[0]\n"
    "ld1 {v11.4s - v12.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v5.4s, v13.4s, v1.s[0]\n"
    "fmla v6.4s, v14.4s, v1.s[0]\n"
    "fmla v7.4s, v13.4s, v2.s[0]\n"
    "fmla v8.4s, v14.4s, v2.s[0]\n"
    "fmla v9.4s, v13.4s, v3.s[0]\n"
    "fmla v10.4s, v14.4s, v3.s[0]\n"
    "ld1 {v13.4s - v14.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v5.4s, v15.4s, v2.s[0]\n"
    "fmla v6.4s, v16.4s, v2.s[0]\n"
    "fmla v7.4s, v15.4s, v3.s[0]\n"
    "fmla v8.4s, v16.4s, v3.s[0]\n"
    "fmla v9.4s, v15.4s, v4.s[0]\n"
    "fmla v10.4s, v16.4s, v4.s[0]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v5.4s, v11.4s, v0.s[1]\n"
    "fmla v6.4s, v12.4s, v0.s[1]\n"
    "fmla v7.4s, v11.4s, v1.s[1]\n"
    "fmla v8.4s, v12.4s, v1.s[1]\n"
    "fmla v9.4s, v11.4s, v2.s[1]\n"
    "fmla v10.4s, v12.4s, v2.s[1]\n"
    "ld1 {v11.4s - v12.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v5.4s, v13.4s, v1.s[1]\n"
    "fmla v6.4s, v14.4s, v1.s[1]\n"
    "fmla v7.4s, v13.4s, v2.s[1]\n"
    "fmla v8.4s, v14.4s, v2.s[1]\n"
    "fmla v9.4s, v13.4s, v3.s[1]\n"
    "fmla v10.4s, v14.4s, v3.s[1]\n"
    "ld1 {v13.4s - v14.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v5.4s, v15.4s, v2.s[1]\n"
    "fmla v6.4s, v16.4s, v2.s[1]\n"
    "fmla v7.4s, v15.4s, v3.s[1]\n"
    "fmla v8.4s, v16.4s, v3.s[1]\n"
    "fmla v9.4s, v15.4s, v4.s[1]\n"
    "fmla v10.4s, v16.4s, v4.s[1]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v5.4s, v11.4s, v0.s[2]\n"
    "fmla v6.4s, v12.4s, v0.s[2]\n"
    "fmla v7.4s, v11.4s, v1.s[2]\n"
    "fmla v8.4s, v12.4s, v1.s[2]\n"
    "fmla v9.4s, v11.4s, v2.s[2]\n"
    "fmla v10.4s, v12.4s, v2.s[2]\n"
// Filter width idx 1
    "fmla v5.4s, v13.4s, v1.s[2]\n"
    "fmla v6.4s, v14.4s, v1.s[2]\n"
    "fmla v7.4s, v13.4s, v2.s[2]\n"
    "fmla v8.4s, v14.4s, v2.s[2]\n"
    "fmla v9.4s, v13.4s, v3.s[2]\n"
    "fmla v10.4s, v14.4s, v3.s[2]\n"
// Filter width idx 2
    "fmla v5.4s, v15.4s, v2.s[2]\n"
    "fmla v6.4s, v16.4s, v2.s[2]\n"
    "fmla v7.4s, v15.4s, v3.s[2]\n"
    "fmla v8.4s, v16.4s, v3.s[2]\n"
    "fmla v9.4s, v15.4s, v4.s[2]\n"
    "fmla v10.4s, v16.4s, v4.s[2]\n"
    "stp q5, q6, [%[out], #0]\n"
    "stp q7, q8, [%[out], #32]\n"
    "stp q9, q10, [%[out], #64]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_2_8_1_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2], [1, 2, 3]]
// Input registers required
// [0, 1, 2, 3]
// Duplicate index: 2
// [1, 2, 2, 1]
// Number of Input Regs: 4, Filter Regs: 6 Output Regs: 4
// Total number of registers required: 14
// In  - [0, 1, 2, 3]
// Out - [4, 5, 6, 7]
// Fil - [8, 9, 10, 11, 12, 13, 8, 9, 10, 11, 8, 9]
// Register mapping diagram
//           0  1  2  3 
//
// 8 10 12   4  6 
// 9 11 13   5  7 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf("%6.3f\t", *(inputPtr + 3*3 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q4, q5, [%[out], #0]\n"
    "ldp q6, q7, [%[out], #32]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v8.4s - v9.4s}, [x11], #32\n"
    "ld1 {v10.4s - v11.4s}, [x11], #32\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"
    "ldr q3, [x10, #36]\n"

// K index 0
// Filter width idx 0
    "fmla v4.4s, v8.4s, v0.s[0]\n"
    "fmla v5.4s, v9.4s, v0.s[0]\n"
    "fmla v6.4s, v8.4s, v1.s[0]\n"
    "fmla v7.4s, v9.4s, v1.s[0]\n"
    "ld1 {v8.4s - v9.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v4.4s, v10.4s, v1.s[0]\n"
    "fmla v5.4s, v11.4s, v1.s[0]\n"
    "fmla v6.4s, v10.4s, v2.s[0]\n"
    "fmla v7.4s, v11.4s, v2.s[0]\n"
    "ld1 {v10.4s - v11.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v4.4s, v12.4s, v2.s[0]\n"
    "fmla v5.4s, v13.4s, v2.s[0]\n"
    "fmla v6.4s, v12.4s, v3.s[0]\n"
    "fmla v7.4s, v13.4s, v3.s[0]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v4.4s, v8.4s, v0.s[1]\n"
    "fmla v5.4s, v9.4s, v0.s[1]\n"
    "fmla v6.4s, v8.4s, v1.s[1]\n"
    "fmla v7.4s, v9.4s, v1.s[1]\n"
    "ld1 {v8.4s - v9.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v4.4s, v10.4s, v1.s[1]\n"
    "fmla v5.4s, v11.4s, v1.s[1]\n"
    "fmla v6.4s, v10.4s, v2.s[1]\n"
    "fmla v7.4s, v11.4s, v2.s[1]\n"
    "ld1 {v10.4s - v11.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v4.4s, v12.4s, v2.s[1]\n"
    "fmla v5.4s, v13.4s, v2.s[1]\n"
    "fmla v6.4s, v12.4s, v3.s[1]\n"
    "fmla v7.4s, v13.4s, v3.s[1]\n"
    "ld1 {v12.4s - v13.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v4.4s, v8.4s, v0.s[2]\n"
    "fmla v5.4s, v9.4s, v0.s[2]\n"
    "fmla v6.4s, v8.4s, v1.s[2]\n"
    "fmla v7.4s, v9.4s, v1.s[2]\n"
// Filter width idx 1
    "fmla v4.4s, v10.4s, v1.s[2]\n"
    "fmla v5.4s, v11.4s, v1.s[2]\n"
    "fmla v6.4s, v10.4s, v2.s[2]\n"
    "fmla v7.4s, v11.4s, v2.s[2]\n"
// Filter width idx 2
    "fmla v4.4s, v12.4s, v2.s[2]\n"
    "fmla v5.4s, v13.4s, v2.s[2]\n"
    "fmla v6.4s, v12.4s, v3.s[2]\n"
    "fmla v7.4s, v13.4s, v3.s[2]\n"
    "stp q4, q5, [%[out], #0]\n"
    "stp q6, q7, [%[out], #32]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
void kernel_3_1_8_1_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2]]
// Input registers required
// [0, 1, 2]
// Duplicate index: 0
// [1, 1, 1]
// Number of Input Regs: 3, Filter Regs: 6 Output Regs: 2
// Total number of registers required: 11
// In  - [0, 1, 2]
// Out - [3, 4]
// Fil - [5, 6, 7, 8, 9, 10, 5, 6, 5, 6, 5, 6]
// Register mapping diagram
//           0  1  2 
//
// 5  7  9   3 
// 6  8 10   4 
//
    #ifdef __DEBUG_PTMM_OFF
    printf ("Input:\n");
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Row %d:\t", j);
            printf("%6.3f\t", *(inputPtr + 3*0 + j));
            printf("%6.3f\t", *(inputPtr + 3*1 + j));
            printf("%6.3f\t", *(inputPtr + 3*2 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 3; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*3*8 + wf*8 + i));
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

    float* in = inputPtr + 0*3;
    float* fil = filterPtr;
    float* out = outputPtr;
    {
    __asm __volatile (
// Prefetch input and filter
    "prfm pldl1keep, [%[fil], #0]\n"
    "prfm pldl1keep, [%[fil], #64]\n"
    "prfm pldl1keep, [%[fil], #128]\n"
    "prfm pldl1keep, [%[fil], #192]\n"
    "mov x11, %[fil]\n"
// Load Output
    "ldp q3, q4, [%[out], #0]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v5.4s - v6.4s}, [x11], #32\n"
    "ld1 {v7.4s - v8.4s}, [x11], #32\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// Load Input
    "ldr q0, [x10, #0]\n"
    "ldr q1, [x10, #12]\n"
    "ldr q2, [x10, #24]\n"

// K index 0
// Filter width idx 0
    "fmla v3.4s, v5.4s, v0.s[0]\n"
    "fmla v4.4s, v6.4s, v0.s[0]\n"
    "ld1 {v5.4s - v6.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v3.4s, v7.4s, v1.s[0]\n"
    "fmla v4.4s, v8.4s, v1.s[0]\n"
    "ld1 {v7.4s - v8.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v3.4s, v9.4s, v2.s[0]\n"
    "fmla v4.4s, v10.4s, v2.s[0]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v3.4s, v5.4s, v0.s[1]\n"
    "fmla v4.4s, v6.4s, v0.s[1]\n"
    "ld1 {v5.4s - v6.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v3.4s, v7.4s, v1.s[1]\n"
    "fmla v4.4s, v8.4s, v1.s[1]\n"
    "ld1 {v7.4s - v8.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v3.4s, v9.4s, v2.s[1]\n"
    "fmla v4.4s, v10.4s, v2.s[1]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v3.4s, v5.4s, v0.s[2]\n"
    "fmla v4.4s, v6.4s, v0.s[2]\n"
// Filter width idx 1
    "fmla v3.4s, v7.4s, v1.s[2]\n"
    "fmla v4.4s, v8.4s, v1.s[2]\n"
// Filter width idx 2
    "fmla v3.4s, v9.4s, v2.s[2]\n"
    "fmla v4.4s, v10.4s, v2.s[2]\n"
    "stp q3, q4, [%[out], #0]\n"
    :
    :           [in] "r" (in), [fil] "r" (fil), [out] "r" (out), [inStr] "r" (inStride*sizeof(float))
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
