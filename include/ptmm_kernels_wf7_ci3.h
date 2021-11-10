void kernel_7_5_8_2_1_3_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11, 12], [8, 9, 10, 11, 12, 13, 14]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Duplicate index: 20
// [1, 1, 2, 2, 3, 3, 4, 3, 4, 3, 3, 2, 2, 1, 1]
// Number of Input Regs: 15, Filter Regs: 4 Output Regs: 10
// Total number of registers required: 29
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Out - [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
// Fil - [25, 26, 27, 28, 25, 26, 27, 28, 25, 26, 27, 28, 25, 26, 25, 26, 27, 28]
// Register mapping diagram
//                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 
//
//25 27 25 27 25 27 25  15    17    19    21    23    
//26 28 26 28 26 28 26  16    18    20    22    24    
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
            printf("%6.3f\t", *(inputPtr + 3*10 + j));
            printf("%6.3f\t", *(inputPtr + 3*11 + j));
            printf("%6.3f\t", *(inputPtr + 3*12 + j));
            printf("%6.3f\t", *(inputPtr + 3*13 + j));
            printf("%6.3f\t", *(inputPtr + 3*14 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 7; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*7*8 + wf*8 + i));
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

    float* in = inputPtr + 3*3;
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
    "ldp q15, q16, [%[out], #0]\n"
    "ldp q17, q18, [%[out], #32]\n"
    "ldp q19, q20, [%[out], #64]\n"
    "ldp q21, q22, [%[out], #96]\n"
    "ldp q23, q24, [%[out], #128]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Load Input
    "ldr q3, [x10, #0]\n"
    "ldr q4, [x10, #12]\n"
    "ldr q5, [x10, #24]\n"
    "ldr q6, [x10, #36]\n"
    "ldr q7, [x10, #48]\n"
    "ldr q8, [x10, #60]\n"
    "ldr q9, [x10, #72]\n"
    "ldr q10, [x10, #84]\n"
    "ldr q11, [x10, #96]\n"
    "ldr q12, [x10, #108]\n"
    "ldr q13, [x10, #120]\n"
    "ldr q14, [x10, #132]\n"

// K index 0
// Filter width idx 0
    "fmla v19.4s, v25.4s, v4.s[0]\n"
    "fmla v20.4s, v26.4s, v4.s[0]\n"
    "fmla v21.4s, v25.4s, v6.s[0]\n"
    "fmla v22.4s, v26.4s, v6.s[0]\n"
    "fmla v23.4s, v25.4s, v8.s[0]\n"
    "fmla v24.4s, v26.4s, v8.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v17.4s, v27.4s, v3.s[0]\n"
    "fmla v18.4s, v28.4s, v3.s[0]\n"
    "fmla v19.4s, v27.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v7.s[0]\n"
    "fmla v22.4s, v28.4s, v7.s[0]\n"
    "fmla v23.4s, v27.4s, v9.s[0]\n"
    "fmla v24.4s, v28.4s, v9.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v17.4s, v25.4s, v4.s[0]\n"
    "fmla v18.4s, v26.4s, v4.s[0]\n"
    "fmla v19.4s, v25.4s, v6.s[0]\n"
    "fmla v20.4s, v26.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v8.s[0]\n"
    "fmla v22.4s, v26.4s, v8.s[0]\n"
    "fmla v23.4s, v25.4s, v10.s[0]\n"
    "fmla v24.4s, v26.4s, v10.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v15.4s, v27.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v3.s[0]\n"
    "fmla v17.4s, v27.4s, v5.s[0]\n"
    "fmla v18.4s, v28.4s, v5.s[0]\n"
    "fmla v19.4s, v27.4s, v7.s[0]\n"
    "fmla v20.4s, v28.4s, v7.s[0]\n"
    "fmla v21.4s, v27.4s, v9.s[0]\n"
    "fmla v22.4s, v28.4s, v9.s[0]\n"
    "fmla v23.4s, v27.4s, v11.s[0]\n"
    "fmla v24.4s, v28.4s, v11.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v15.4s, v25.4s, v4.s[0]\n"
    "fmla v16.4s, v26.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v6.s[0]\n"
    "fmla v18.4s, v26.4s, v6.s[0]\n"
    "fmla v19.4s, v25.4s, v8.s[0]\n"
    "fmla v20.4s, v26.4s, v8.s[0]\n"
    "fmla v21.4s, v25.4s, v10.s[0]\n"
    "fmla v22.4s, v26.4s, v10.s[0]\n"
    "fmla v23.4s, v25.4s, v12.s[0]\n"
    "fmla v24.4s, v26.4s, v12.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v15.4s, v27.4s, v5.s[0]\n"
    "fmla v16.4s, v28.4s, v5.s[0]\n"
    "fmla v17.4s, v27.4s, v7.s[0]\n"
    "fmla v18.4s, v28.4s, v7.s[0]\n"
    "fmla v19.4s, v27.4s, v9.s[0]\n"
    "fmla v20.4s, v28.4s, v9.s[0]\n"
    "fmla v21.4s, v27.4s, v11.s[0]\n"
    "fmla v22.4s, v28.4s, v11.s[0]\n"
    "fmla v23.4s, v27.4s, v13.s[0]\n"
    "fmla v24.4s, v28.4s, v13.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v15.4s, v25.4s, v6.s[0]\n"
    "fmla v16.4s, v26.4s, v6.s[0]\n"
    "fmla v17.4s, v25.4s, v8.s[0]\n"
    "fmla v18.4s, v26.4s, v8.s[0]\n"
    "fmla v19.4s, v25.4s, v10.s[0]\n"
    "fmla v20.4s, v26.4s, v10.s[0]\n"
    "fmla v21.4s, v25.4s, v12.s[0]\n"
    "fmla v22.4s, v26.4s, v12.s[0]\n"
    "fmla v23.4s, v25.4s, v14.s[0]\n"
    "fmla v24.4s, v26.4s, v14.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v19.4s, v27.4s, v4.s[1]\n"
    "fmla v20.4s, v28.4s, v4.s[1]\n"
    "fmla v21.4s, v27.4s, v6.s[1]\n"
    "fmla v22.4s, v28.4s, v6.s[1]\n"
    "fmla v23.4s, v27.4s, v8.s[1]\n"
    "fmla v24.4s, v28.4s, v8.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v17.4s, v25.4s, v3.s[1]\n"
    "fmla v18.4s, v26.4s, v3.s[1]\n"
    "fmla v19.4s, v25.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v5.s[1]\n"
    "fmla v21.4s, v25.4s, v7.s[1]\n"
    "fmla v22.4s, v26.4s, v7.s[1]\n"
    "fmla v23.4s, v25.4s, v9.s[1]\n"
    "fmla v24.4s, v26.4s, v9.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v17.4s, v27.4s, v4.s[1]\n"
    "fmla v18.4s, v28.4s, v4.s[1]\n"
    "fmla v19.4s, v27.4s, v6.s[1]\n"
    "fmla v20.4s, v28.4s, v6.s[1]\n"
    "fmla v21.4s, v27.4s, v8.s[1]\n"
    "fmla v22.4s, v28.4s, v8.s[1]\n"
    "fmla v23.4s, v27.4s, v10.s[1]\n"
    "fmla v24.4s, v28.4s, v10.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v15.4s, v25.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v3.s[1]\n"
    "fmla v17.4s, v25.4s, v5.s[1]\n"
    "fmla v18.4s, v26.4s, v5.s[1]\n"
    "fmla v19.4s, v25.4s, v7.s[1]\n"
    "fmla v20.4s, v26.4s, v7.s[1]\n"
    "fmla v21.4s, v25.4s, v9.s[1]\n"
    "fmla v22.4s, v26.4s, v9.s[1]\n"
    "fmla v23.4s, v25.4s, v11.s[1]\n"
    "fmla v24.4s, v26.4s, v11.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v15.4s, v27.4s, v4.s[1]\n"
    "fmla v16.4s, v28.4s, v4.s[1]\n"
    "fmla v17.4s, v27.4s, v6.s[1]\n"
    "fmla v18.4s, v28.4s, v6.s[1]\n"
    "fmla v19.4s, v27.4s, v8.s[1]\n"
    "fmla v20.4s, v28.4s, v8.s[1]\n"
    "fmla v21.4s, v27.4s, v10.s[1]\n"
    "fmla v22.4s, v28.4s, v10.s[1]\n"
    "fmla v23.4s, v27.4s, v12.s[1]\n"
    "fmla v24.4s, v28.4s, v12.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v15.4s, v25.4s, v5.s[1]\n"
    "fmla v16.4s, v26.4s, v5.s[1]\n"
    "fmla v17.4s, v25.4s, v7.s[1]\n"
    "fmla v18.4s, v26.4s, v7.s[1]\n"
    "fmla v19.4s, v25.4s, v9.s[1]\n"
    "fmla v20.4s, v26.4s, v9.s[1]\n"
    "fmla v21.4s, v25.4s, v11.s[1]\n"
    "fmla v22.4s, v26.4s, v11.s[1]\n"
    "fmla v23.4s, v25.4s, v13.s[1]\n"
    "fmla v24.4s, v26.4s, v13.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v15.4s, v27.4s, v6.s[1]\n"
    "fmla v16.4s, v28.4s, v6.s[1]\n"
    "fmla v17.4s, v27.4s, v8.s[1]\n"
    "fmla v18.4s, v28.4s, v8.s[1]\n"
    "fmla v19.4s, v27.4s, v10.s[1]\n"
    "fmla v20.4s, v28.4s, v10.s[1]\n"
    "fmla v21.4s, v27.4s, v12.s[1]\n"
    "fmla v22.4s, v28.4s, v12.s[1]\n"
    "fmla v23.4s, v27.4s, v14.s[1]\n"
    "fmla v24.4s, v28.4s, v14.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v19.4s, v25.4s, v4.s[2]\n"
    "fmla v20.4s, v26.4s, v4.s[2]\n"
    "fmla v21.4s, v25.4s, v6.s[2]\n"
    "fmla v22.4s, v26.4s, v6.s[2]\n"
    "fmla v23.4s, v25.4s, v8.s[2]\n"
    "fmla v24.4s, v26.4s, v8.s[2]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v17.4s, v27.4s, v3.s[2]\n"
    "fmla v18.4s, v28.4s, v3.s[2]\n"
    "fmla v19.4s, v27.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v7.s[2]\n"
    "fmla v22.4s, v28.4s, v7.s[2]\n"
    "fmla v23.4s, v27.4s, v9.s[2]\n"
    "fmla v24.4s, v28.4s, v9.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v17.4s, v25.4s, v4.s[2]\n"
    "fmla v18.4s, v26.4s, v4.s[2]\n"
    "fmla v19.4s, v25.4s, v6.s[2]\n"
    "fmla v20.4s, v26.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v8.s[2]\n"
    "fmla v22.4s, v26.4s, v8.s[2]\n"
    "fmla v23.4s, v25.4s, v10.s[2]\n"
    "fmla v24.4s, v26.4s, v10.s[2]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v15.4s, v27.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v5.s[2]\n"
    "fmla v18.4s, v28.4s, v5.s[2]\n"
    "fmla v19.4s, v27.4s, v7.s[2]\n"
    "fmla v20.4s, v28.4s, v7.s[2]\n"
    "fmla v21.4s, v27.4s, v9.s[2]\n"
    "fmla v22.4s, v28.4s, v9.s[2]\n"
    "fmla v23.4s, v27.4s, v11.s[2]\n"
    "fmla v24.4s, v28.4s, v11.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v15.4s, v25.4s, v4.s[2]\n"
    "fmla v16.4s, v26.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v6.s[2]\n"
    "fmla v18.4s, v26.4s, v6.s[2]\n"
    "fmla v19.4s, v25.4s, v8.s[2]\n"
    "fmla v20.4s, v26.4s, v8.s[2]\n"
    "fmla v21.4s, v25.4s, v10.s[2]\n"
    "fmla v22.4s, v26.4s, v10.s[2]\n"
    "fmla v23.4s, v25.4s, v12.s[2]\n"
    "fmla v24.4s, v26.4s, v12.s[2]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v15.4s, v27.4s, v5.s[2]\n"
    "fmla v16.4s, v28.4s, v5.s[2]\n"
    "fmla v17.4s, v27.4s, v7.s[2]\n"
    "fmla v18.4s, v28.4s, v7.s[2]\n"
    "fmla v19.4s, v27.4s, v9.s[2]\n"
    "fmla v20.4s, v28.4s, v9.s[2]\n"
    "fmla v21.4s, v27.4s, v11.s[2]\n"
    "fmla v22.4s, v28.4s, v11.s[2]\n"
    "fmla v23.4s, v27.4s, v13.s[2]\n"
    "fmla v24.4s, v28.4s, v13.s[2]\n"
// Filter width idx 6
    "fmla v15.4s, v25.4s, v6.s[2]\n"
    "fmla v16.4s, v26.4s, v6.s[2]\n"
    "fmla v17.4s, v25.4s, v8.s[2]\n"
    "fmla v18.4s, v26.4s, v8.s[2]\n"
    "fmla v19.4s, v25.4s, v10.s[2]\n"
    "fmla v20.4s, v26.4s, v10.s[2]\n"
    "fmla v21.4s, v25.4s, v12.s[2]\n"
    "fmla v22.4s, v26.4s, v12.s[2]\n"
    "fmla v23.4s, v25.4s, v14.s[2]\n"
    "fmla v24.4s, v26.4s, v14.s[2]\n"
    "stp q15, q16, [%[out], #0]\n"
    "stp q17, q18, [%[out], #32]\n"
    "stp q19, q20, [%[out], #64]\n"
    "stp q21, q22, [%[out], #96]\n"
    "stp q23, q24, [%[out], #128]\n"
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
void kernel_7_5_8_2_1_0_2_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11, 12], [8, 9, 10, 11, 12, 13, 14]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Duplicate index: 20
// [1, 1, 2, 2, 3, 3, 4, 3, 4, 3, 3, 2, 2, 1, 1]
// Number of Input Regs: 15, Filter Regs: 4 Output Regs: 10
// Total number of registers required: 29
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Out - [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
// Fil - [25, 26, 27, 28, 25, 26, 27, 28, 25, 26, 27, 28, 25, 26, 25, 26, 27, 28]
// Register mapping diagram
//                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 
//
//25 27 25 27 25 27 25  15    17    19    21    23    
//26 28 26 28 26 28 26  16    18    20    22    24    
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
            printf("%6.3f\t", *(inputPtr + 3*10 + j));
            printf("%6.3f\t", *(inputPtr + 3*11 + j));
            printf("%6.3f\t", *(inputPtr + 3*12 + j));
            printf("%6.3f\t", *(inputPtr + 3*13 + j));
            printf("%6.3f\t", *(inputPtr + 3*14 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 7; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*7*8 + wf*8 + i));
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
    "ldp q15, q16, [%[out], #0]\n"
    "ldp q17, q18, [%[out], #32]\n"
    "ldp q19, q20, [%[out], #64]\n"
    "ldp q21, q22, [%[out], #96]\n"
    "ldp q23, q24, [%[out], #128]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
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

// K index 0
// Filter width idx 0
    "fmla v15.4s, v25.4s, v0.s[0]\n"
    "fmla v16.4s, v26.4s, v0.s[0]\n"
    "fmla v17.4s, v25.4s, v2.s[0]\n"
    "fmla v18.4s, v26.4s, v2.s[0]\n"
    "fmla v19.4s, v25.4s, v4.s[0]\n"
    "fmla v20.4s, v26.4s, v4.s[0]\n"
    "fmla v21.4s, v25.4s, v6.s[0]\n"
    "fmla v22.4s, v26.4s, v6.s[0]\n"
    "fmla v23.4s, v25.4s, v8.s[0]\n"
    "fmla v24.4s, v26.4s, v8.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v15.4s, v27.4s, v1.s[0]\n"
    "fmla v16.4s, v28.4s, v1.s[0]\n"
    "fmla v17.4s, v27.4s, v3.s[0]\n"
    "fmla v18.4s, v28.4s, v3.s[0]\n"
    "fmla v19.4s, v27.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v7.s[0]\n"
    "fmla v22.4s, v28.4s, v7.s[0]\n"
    "fmla v23.4s, v27.4s, v9.s[0]\n"
    "fmla v24.4s, v28.4s, v9.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v15.4s, v25.4s, v2.s[0]\n"
    "fmla v16.4s, v26.4s, v2.s[0]\n"
    "fmla v17.4s, v25.4s, v4.s[0]\n"
    "fmla v18.4s, v26.4s, v4.s[0]\n"
    "fmla v19.4s, v25.4s, v6.s[0]\n"
    "fmla v20.4s, v26.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v8.s[0]\n"
    "fmla v22.4s, v26.4s, v8.s[0]\n"
    "fmla v23.4s, v25.4s, v10.s[0]\n"
    "fmla v24.4s, v26.4s, v10.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v15.4s, v27.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v3.s[0]\n"
    "fmla v17.4s, v27.4s, v5.s[0]\n"
    "fmla v18.4s, v28.4s, v5.s[0]\n"
    "fmla v19.4s, v27.4s, v7.s[0]\n"
    "fmla v20.4s, v28.4s, v7.s[0]\n"
    "fmla v21.4s, v27.4s, v9.s[0]\n"
    "fmla v22.4s, v28.4s, v9.s[0]\n"
    "fmla v23.4s, v27.4s, v11.s[0]\n"
    "fmla v24.4s, v28.4s, v11.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v15.4s, v25.4s, v4.s[0]\n"
    "fmla v16.4s, v26.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v6.s[0]\n"
    "fmla v18.4s, v26.4s, v6.s[0]\n"
    "fmla v19.4s, v25.4s, v8.s[0]\n"
    "fmla v20.4s, v26.4s, v8.s[0]\n"
    "fmla v21.4s, v25.4s, v10.s[0]\n"
    "fmla v22.4s, v26.4s, v10.s[0]\n"
    "fmla v23.4s, v25.4s, v12.s[0]\n"
    "fmla v24.4s, v26.4s, v12.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v15.4s, v27.4s, v5.s[0]\n"
    "fmla v16.4s, v28.4s, v5.s[0]\n"
    "fmla v17.4s, v27.4s, v7.s[0]\n"
    "fmla v18.4s, v28.4s, v7.s[0]\n"
    "fmla v19.4s, v27.4s, v9.s[0]\n"
    "fmla v20.4s, v28.4s, v9.s[0]\n"
    "fmla v21.4s, v27.4s, v11.s[0]\n"
    "fmla v22.4s, v28.4s, v11.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v15.4s, v25.4s, v6.s[0]\n"
    "fmla v16.4s, v26.4s, v6.s[0]\n"
    "fmla v17.4s, v25.4s, v8.s[0]\n"
    "fmla v18.4s, v26.4s, v8.s[0]\n"
    "fmla v19.4s, v25.4s, v10.s[0]\n"
    "fmla v20.4s, v26.4s, v10.s[0]\n"
    "fmla v21.4s, v25.4s, v12.s[0]\n"
    "fmla v22.4s, v26.4s, v12.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v15.4s, v27.4s, v0.s[1]\n"
    "fmla v16.4s, v28.4s, v0.s[1]\n"
    "fmla v17.4s, v27.4s, v2.s[1]\n"
    "fmla v18.4s, v28.4s, v2.s[1]\n"
    "fmla v19.4s, v27.4s, v4.s[1]\n"
    "fmla v20.4s, v28.4s, v4.s[1]\n"
    "fmla v21.4s, v27.4s, v6.s[1]\n"
    "fmla v22.4s, v28.4s, v6.s[1]\n"
    "fmla v23.4s, v27.4s, v8.s[1]\n"
    "fmla v24.4s, v28.4s, v8.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v15.4s, v25.4s, v1.s[1]\n"
    "fmla v16.4s, v26.4s, v1.s[1]\n"
    "fmla v17.4s, v25.4s, v3.s[1]\n"
    "fmla v18.4s, v26.4s, v3.s[1]\n"
    "fmla v19.4s, v25.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v5.s[1]\n"
    "fmla v21.4s, v25.4s, v7.s[1]\n"
    "fmla v22.4s, v26.4s, v7.s[1]\n"
    "fmla v23.4s, v25.4s, v9.s[1]\n"
    "fmla v24.4s, v26.4s, v9.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v15.4s, v27.4s, v2.s[1]\n"
    "fmla v16.4s, v28.4s, v2.s[1]\n"
    "fmla v17.4s, v27.4s, v4.s[1]\n"
    "fmla v18.4s, v28.4s, v4.s[1]\n"
    "fmla v19.4s, v27.4s, v6.s[1]\n"
    "fmla v20.4s, v28.4s, v6.s[1]\n"
    "fmla v21.4s, v27.4s, v8.s[1]\n"
    "fmla v22.4s, v28.4s, v8.s[1]\n"
    "fmla v23.4s, v27.4s, v10.s[1]\n"
    "fmla v24.4s, v28.4s, v10.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v15.4s, v25.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v3.s[1]\n"
    "fmla v17.4s, v25.4s, v5.s[1]\n"
    "fmla v18.4s, v26.4s, v5.s[1]\n"
    "fmla v19.4s, v25.4s, v7.s[1]\n"
    "fmla v20.4s, v26.4s, v7.s[1]\n"
    "fmla v21.4s, v25.4s, v9.s[1]\n"
    "fmla v22.4s, v26.4s, v9.s[1]\n"
    "fmla v23.4s, v25.4s, v11.s[1]\n"
    "fmla v24.4s, v26.4s, v11.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v15.4s, v27.4s, v4.s[1]\n"
    "fmla v16.4s, v28.4s, v4.s[1]\n"
    "fmla v17.4s, v27.4s, v6.s[1]\n"
    "fmla v18.4s, v28.4s, v6.s[1]\n"
    "fmla v19.4s, v27.4s, v8.s[1]\n"
    "fmla v20.4s, v28.4s, v8.s[1]\n"
    "fmla v21.4s, v27.4s, v10.s[1]\n"
    "fmla v22.4s, v28.4s, v10.s[1]\n"
    "fmla v23.4s, v27.4s, v12.s[1]\n"
    "fmla v24.4s, v28.4s, v12.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v15.4s, v25.4s, v5.s[1]\n"
    "fmla v16.4s, v26.4s, v5.s[1]\n"
    "fmla v17.4s, v25.4s, v7.s[1]\n"
    "fmla v18.4s, v26.4s, v7.s[1]\n"
    "fmla v19.4s, v25.4s, v9.s[1]\n"
    "fmla v20.4s, v26.4s, v9.s[1]\n"
    "fmla v21.4s, v25.4s, v11.s[1]\n"
    "fmla v22.4s, v26.4s, v11.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v15.4s, v27.4s, v6.s[1]\n"
    "fmla v16.4s, v28.4s, v6.s[1]\n"
    "fmla v17.4s, v27.4s, v8.s[1]\n"
    "fmla v18.4s, v28.4s, v8.s[1]\n"
    "fmla v19.4s, v27.4s, v10.s[1]\n"
    "fmla v20.4s, v28.4s, v10.s[1]\n"
    "fmla v21.4s, v27.4s, v12.s[1]\n"
    "fmla v22.4s, v28.4s, v12.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v15.4s, v25.4s, v0.s[2]\n"
    "fmla v16.4s, v26.4s, v0.s[2]\n"
    "fmla v17.4s, v25.4s, v2.s[2]\n"
    "fmla v18.4s, v26.4s, v2.s[2]\n"
    "fmla v19.4s, v25.4s, v4.s[2]\n"
    "fmla v20.4s, v26.4s, v4.s[2]\n"
    "fmla v21.4s, v25.4s, v6.s[2]\n"
    "fmla v22.4s, v26.4s, v6.s[2]\n"
    "fmla v23.4s, v25.4s, v8.s[2]\n"
    "fmla v24.4s, v26.4s, v8.s[2]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v15.4s, v27.4s, v1.s[2]\n"
    "fmla v16.4s, v28.4s, v1.s[2]\n"
    "fmla v17.4s, v27.4s, v3.s[2]\n"
    "fmla v18.4s, v28.4s, v3.s[2]\n"
    "fmla v19.4s, v27.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v7.s[2]\n"
    "fmla v22.4s, v28.4s, v7.s[2]\n"
    "fmla v23.4s, v27.4s, v9.s[2]\n"
    "fmla v24.4s, v28.4s, v9.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v15.4s, v25.4s, v2.s[2]\n"
    "fmla v16.4s, v26.4s, v2.s[2]\n"
    "fmla v17.4s, v25.4s, v4.s[2]\n"
    "fmla v18.4s, v26.4s, v4.s[2]\n"
    "fmla v19.4s, v25.4s, v6.s[2]\n"
    "fmla v20.4s, v26.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v8.s[2]\n"
    "fmla v22.4s, v26.4s, v8.s[2]\n"
    "fmla v23.4s, v25.4s, v10.s[2]\n"
    "fmla v24.4s, v26.4s, v10.s[2]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v15.4s, v27.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v5.s[2]\n"
    "fmla v18.4s, v28.4s, v5.s[2]\n"
    "fmla v19.4s, v27.4s, v7.s[2]\n"
    "fmla v20.4s, v28.4s, v7.s[2]\n"
    "fmla v21.4s, v27.4s, v9.s[2]\n"
    "fmla v22.4s, v28.4s, v9.s[2]\n"
    "fmla v23.4s, v27.4s, v11.s[2]\n"
    "fmla v24.4s, v28.4s, v11.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v15.4s, v25.4s, v4.s[2]\n"
    "fmla v16.4s, v26.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v6.s[2]\n"
    "fmla v18.4s, v26.4s, v6.s[2]\n"
    "fmla v19.4s, v25.4s, v8.s[2]\n"
    "fmla v20.4s, v26.4s, v8.s[2]\n"
    "fmla v21.4s, v25.4s, v10.s[2]\n"
    "fmla v22.4s, v26.4s, v10.s[2]\n"
    "fmla v23.4s, v25.4s, v12.s[2]\n"
    "fmla v24.4s, v26.4s, v12.s[2]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v15.4s, v27.4s, v5.s[2]\n"
    "fmla v16.4s, v28.4s, v5.s[2]\n"
    "fmla v17.4s, v27.4s, v7.s[2]\n"
    "fmla v18.4s, v28.4s, v7.s[2]\n"
    "fmla v19.4s, v27.4s, v9.s[2]\n"
    "fmla v20.4s, v28.4s, v9.s[2]\n"
    "fmla v21.4s, v27.4s, v11.s[2]\n"
    "fmla v22.4s, v28.4s, v11.s[2]\n"
// Filter width idx 6
    "fmla v15.4s, v25.4s, v6.s[2]\n"
    "fmla v16.4s, v26.4s, v6.s[2]\n"
    "fmla v17.4s, v25.4s, v8.s[2]\n"
    "fmla v18.4s, v26.4s, v8.s[2]\n"
    "fmla v19.4s, v25.4s, v10.s[2]\n"
    "fmla v20.4s, v26.4s, v10.s[2]\n"
    "fmla v21.4s, v25.4s, v12.s[2]\n"
    "fmla v22.4s, v26.4s, v12.s[2]\n"
    "stp q15, q16, [%[out], #0]\n"
    "stp q17, q18, [%[out], #32]\n"
    "stp q19, q20, [%[out], #64]\n"
    "stp q21, q22, [%[out], #96]\n"
    "stp q23, q24, [%[out], #128]\n"
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
void kernel_7_5_8_2_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11, 12], [8, 9, 10, 11, 12, 13, 14]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Duplicate index: 20
// [1, 1, 2, 2, 3, 3, 4, 3, 4, 3, 3, 2, 2, 1, 1]
// Number of Input Regs: 15, Filter Regs: 4 Output Regs: 10
// Total number of registers required: 29
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Out - [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
// Fil - [25, 26, 27, 28, 25, 26, 27, 28, 25, 26, 27, 28, 25, 26, 25, 26, 27, 28]
// Register mapping diagram
//                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 
//
//25 27 25 27 25 27 25  15    17    19    21    23    
//26 28 26 28 26 28 26  16    18    20    22    24    
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
            printf("%6.3f\t", *(inputPtr + 3*10 + j));
            printf("%6.3f\t", *(inputPtr + 3*11 + j));
            printf("%6.3f\t", *(inputPtr + 3*12 + j));
            printf("%6.3f\t", *(inputPtr + 3*13 + j));
            printf("%6.3f\t", *(inputPtr + 3*14 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 7; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*7*8 + wf*8 + i));
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
    "ldp q15, q16, [%[out], #0]\n"
    "ldp q17, q18, [%[out], #32]\n"
    "ldp q19, q20, [%[out], #64]\n"
    "ldp q21, q22, [%[out], #96]\n"
    "ldp q23, q24, [%[out], #128]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
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

// K index 0
// Filter width idx 0
    "fmla v15.4s, v25.4s, v0.s[0]\n"
    "fmla v16.4s, v26.4s, v0.s[0]\n"
    "fmla v17.4s, v25.4s, v2.s[0]\n"
    "fmla v18.4s, v26.4s, v2.s[0]\n"
    "fmla v19.4s, v25.4s, v4.s[0]\n"
    "fmla v20.4s, v26.4s, v4.s[0]\n"
    "fmla v21.4s, v25.4s, v6.s[0]\n"
    "fmla v22.4s, v26.4s, v6.s[0]\n"
    "fmla v23.4s, v25.4s, v8.s[0]\n"
    "fmla v24.4s, v26.4s, v8.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v15.4s, v27.4s, v1.s[0]\n"
    "fmla v16.4s, v28.4s, v1.s[0]\n"
    "fmla v17.4s, v27.4s, v3.s[0]\n"
    "fmla v18.4s, v28.4s, v3.s[0]\n"
    "fmla v19.4s, v27.4s, v5.s[0]\n"
    "fmla v20.4s, v28.4s, v5.s[0]\n"
    "fmla v21.4s, v27.4s, v7.s[0]\n"
    "fmla v22.4s, v28.4s, v7.s[0]\n"
    "fmla v23.4s, v27.4s, v9.s[0]\n"
    "fmla v24.4s, v28.4s, v9.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v15.4s, v25.4s, v2.s[0]\n"
    "fmla v16.4s, v26.4s, v2.s[0]\n"
    "fmla v17.4s, v25.4s, v4.s[0]\n"
    "fmla v18.4s, v26.4s, v4.s[0]\n"
    "fmla v19.4s, v25.4s, v6.s[0]\n"
    "fmla v20.4s, v26.4s, v6.s[0]\n"
    "fmla v21.4s, v25.4s, v8.s[0]\n"
    "fmla v22.4s, v26.4s, v8.s[0]\n"
    "fmla v23.4s, v25.4s, v10.s[0]\n"
    "fmla v24.4s, v26.4s, v10.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v15.4s, v27.4s, v3.s[0]\n"
    "fmla v16.4s, v28.4s, v3.s[0]\n"
    "fmla v17.4s, v27.4s, v5.s[0]\n"
    "fmla v18.4s, v28.4s, v5.s[0]\n"
    "fmla v19.4s, v27.4s, v7.s[0]\n"
    "fmla v20.4s, v28.4s, v7.s[0]\n"
    "fmla v21.4s, v27.4s, v9.s[0]\n"
    "fmla v22.4s, v28.4s, v9.s[0]\n"
    "fmla v23.4s, v27.4s, v11.s[0]\n"
    "fmla v24.4s, v28.4s, v11.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v15.4s, v25.4s, v4.s[0]\n"
    "fmla v16.4s, v26.4s, v4.s[0]\n"
    "fmla v17.4s, v25.4s, v6.s[0]\n"
    "fmla v18.4s, v26.4s, v6.s[0]\n"
    "fmla v19.4s, v25.4s, v8.s[0]\n"
    "fmla v20.4s, v26.4s, v8.s[0]\n"
    "fmla v21.4s, v25.4s, v10.s[0]\n"
    "fmla v22.4s, v26.4s, v10.s[0]\n"
    "fmla v23.4s, v25.4s, v12.s[0]\n"
    "fmla v24.4s, v26.4s, v12.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v15.4s, v27.4s, v5.s[0]\n"
    "fmla v16.4s, v28.4s, v5.s[0]\n"
    "fmla v17.4s, v27.4s, v7.s[0]\n"
    "fmla v18.4s, v28.4s, v7.s[0]\n"
    "fmla v19.4s, v27.4s, v9.s[0]\n"
    "fmla v20.4s, v28.4s, v9.s[0]\n"
    "fmla v21.4s, v27.4s, v11.s[0]\n"
    "fmla v22.4s, v28.4s, v11.s[0]\n"
    "fmla v23.4s, v27.4s, v13.s[0]\n"
    "fmla v24.4s, v28.4s, v13.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v15.4s, v25.4s, v6.s[0]\n"
    "fmla v16.4s, v26.4s, v6.s[0]\n"
    "fmla v17.4s, v25.4s, v8.s[0]\n"
    "fmla v18.4s, v26.4s, v8.s[0]\n"
    "fmla v19.4s, v25.4s, v10.s[0]\n"
    "fmla v20.4s, v26.4s, v10.s[0]\n"
    "fmla v21.4s, v25.4s, v12.s[0]\n"
    "fmla v22.4s, v26.4s, v12.s[0]\n"
    "fmla v23.4s, v25.4s, v14.s[0]\n"
    "fmla v24.4s, v26.4s, v14.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v15.4s, v27.4s, v0.s[1]\n"
    "fmla v16.4s, v28.4s, v0.s[1]\n"
    "fmla v17.4s, v27.4s, v2.s[1]\n"
    "fmla v18.4s, v28.4s, v2.s[1]\n"
    "fmla v19.4s, v27.4s, v4.s[1]\n"
    "fmla v20.4s, v28.4s, v4.s[1]\n"
    "fmla v21.4s, v27.4s, v6.s[1]\n"
    "fmla v22.4s, v28.4s, v6.s[1]\n"
    "fmla v23.4s, v27.4s, v8.s[1]\n"
    "fmla v24.4s, v28.4s, v8.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v15.4s, v25.4s, v1.s[1]\n"
    "fmla v16.4s, v26.4s, v1.s[1]\n"
    "fmla v17.4s, v25.4s, v3.s[1]\n"
    "fmla v18.4s, v26.4s, v3.s[1]\n"
    "fmla v19.4s, v25.4s, v5.s[1]\n"
    "fmla v20.4s, v26.4s, v5.s[1]\n"
    "fmla v21.4s, v25.4s, v7.s[1]\n"
    "fmla v22.4s, v26.4s, v7.s[1]\n"
    "fmla v23.4s, v25.4s, v9.s[1]\n"
    "fmla v24.4s, v26.4s, v9.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v15.4s, v27.4s, v2.s[1]\n"
    "fmla v16.4s, v28.4s, v2.s[1]\n"
    "fmla v17.4s, v27.4s, v4.s[1]\n"
    "fmla v18.4s, v28.4s, v4.s[1]\n"
    "fmla v19.4s, v27.4s, v6.s[1]\n"
    "fmla v20.4s, v28.4s, v6.s[1]\n"
    "fmla v21.4s, v27.4s, v8.s[1]\n"
    "fmla v22.4s, v28.4s, v8.s[1]\n"
    "fmla v23.4s, v27.4s, v10.s[1]\n"
    "fmla v24.4s, v28.4s, v10.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v15.4s, v25.4s, v3.s[1]\n"
    "fmla v16.4s, v26.4s, v3.s[1]\n"
    "fmla v17.4s, v25.4s, v5.s[1]\n"
    "fmla v18.4s, v26.4s, v5.s[1]\n"
    "fmla v19.4s, v25.4s, v7.s[1]\n"
    "fmla v20.4s, v26.4s, v7.s[1]\n"
    "fmla v21.4s, v25.4s, v9.s[1]\n"
    "fmla v22.4s, v26.4s, v9.s[1]\n"
    "fmla v23.4s, v25.4s, v11.s[1]\n"
    "fmla v24.4s, v26.4s, v11.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v15.4s, v27.4s, v4.s[1]\n"
    "fmla v16.4s, v28.4s, v4.s[1]\n"
    "fmla v17.4s, v27.4s, v6.s[1]\n"
    "fmla v18.4s, v28.4s, v6.s[1]\n"
    "fmla v19.4s, v27.4s, v8.s[1]\n"
    "fmla v20.4s, v28.4s, v8.s[1]\n"
    "fmla v21.4s, v27.4s, v10.s[1]\n"
    "fmla v22.4s, v28.4s, v10.s[1]\n"
    "fmla v23.4s, v27.4s, v12.s[1]\n"
    "fmla v24.4s, v28.4s, v12.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v15.4s, v25.4s, v5.s[1]\n"
    "fmla v16.4s, v26.4s, v5.s[1]\n"
    "fmla v17.4s, v25.4s, v7.s[1]\n"
    "fmla v18.4s, v26.4s, v7.s[1]\n"
    "fmla v19.4s, v25.4s, v9.s[1]\n"
    "fmla v20.4s, v26.4s, v9.s[1]\n"
    "fmla v21.4s, v25.4s, v11.s[1]\n"
    "fmla v22.4s, v26.4s, v11.s[1]\n"
    "fmla v23.4s, v25.4s, v13.s[1]\n"
    "fmla v24.4s, v26.4s, v13.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v15.4s, v27.4s, v6.s[1]\n"
    "fmla v16.4s, v28.4s, v6.s[1]\n"
    "fmla v17.4s, v27.4s, v8.s[1]\n"
    "fmla v18.4s, v28.4s, v8.s[1]\n"
    "fmla v19.4s, v27.4s, v10.s[1]\n"
    "fmla v20.4s, v28.4s, v10.s[1]\n"
    "fmla v21.4s, v27.4s, v12.s[1]\n"
    "fmla v22.4s, v28.4s, v12.s[1]\n"
    "fmla v23.4s, v27.4s, v14.s[1]\n"
    "fmla v24.4s, v28.4s, v14.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v15.4s, v25.4s, v0.s[2]\n"
    "fmla v16.4s, v26.4s, v0.s[2]\n"
    "fmla v17.4s, v25.4s, v2.s[2]\n"
    "fmla v18.4s, v26.4s, v2.s[2]\n"
    "fmla v19.4s, v25.4s, v4.s[2]\n"
    "fmla v20.4s, v26.4s, v4.s[2]\n"
    "fmla v21.4s, v25.4s, v6.s[2]\n"
    "fmla v22.4s, v26.4s, v6.s[2]\n"
    "fmla v23.4s, v25.4s, v8.s[2]\n"
    "fmla v24.4s, v26.4s, v8.s[2]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v15.4s, v27.4s, v1.s[2]\n"
    "fmla v16.4s, v28.4s, v1.s[2]\n"
    "fmla v17.4s, v27.4s, v3.s[2]\n"
    "fmla v18.4s, v28.4s, v3.s[2]\n"
    "fmla v19.4s, v27.4s, v5.s[2]\n"
    "fmla v20.4s, v28.4s, v5.s[2]\n"
    "fmla v21.4s, v27.4s, v7.s[2]\n"
    "fmla v22.4s, v28.4s, v7.s[2]\n"
    "fmla v23.4s, v27.4s, v9.s[2]\n"
    "fmla v24.4s, v28.4s, v9.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v15.4s, v25.4s, v2.s[2]\n"
    "fmla v16.4s, v26.4s, v2.s[2]\n"
    "fmla v17.4s, v25.4s, v4.s[2]\n"
    "fmla v18.4s, v26.4s, v4.s[2]\n"
    "fmla v19.4s, v25.4s, v6.s[2]\n"
    "fmla v20.4s, v26.4s, v6.s[2]\n"
    "fmla v21.4s, v25.4s, v8.s[2]\n"
    "fmla v22.4s, v26.4s, v8.s[2]\n"
    "fmla v23.4s, v25.4s, v10.s[2]\n"
    "fmla v24.4s, v26.4s, v10.s[2]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v15.4s, v27.4s, v3.s[2]\n"
    "fmla v16.4s, v28.4s, v3.s[2]\n"
    "fmla v17.4s, v27.4s, v5.s[2]\n"
    "fmla v18.4s, v28.4s, v5.s[2]\n"
    "fmla v19.4s, v27.4s, v7.s[2]\n"
    "fmla v20.4s, v28.4s, v7.s[2]\n"
    "fmla v21.4s, v27.4s, v9.s[2]\n"
    "fmla v22.4s, v28.4s, v9.s[2]\n"
    "fmla v23.4s, v27.4s, v11.s[2]\n"
    "fmla v24.4s, v28.4s, v11.s[2]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v15.4s, v25.4s, v4.s[2]\n"
    "fmla v16.4s, v26.4s, v4.s[2]\n"
    "fmla v17.4s, v25.4s, v6.s[2]\n"
    "fmla v18.4s, v26.4s, v6.s[2]\n"
    "fmla v19.4s, v25.4s, v8.s[2]\n"
    "fmla v20.4s, v26.4s, v8.s[2]\n"
    "fmla v21.4s, v25.4s, v10.s[2]\n"
    "fmla v22.4s, v26.4s, v10.s[2]\n"
    "fmla v23.4s, v25.4s, v12.s[2]\n"
    "fmla v24.4s, v26.4s, v12.s[2]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v15.4s, v27.4s, v5.s[2]\n"
    "fmla v16.4s, v28.4s, v5.s[2]\n"
    "fmla v17.4s, v27.4s, v7.s[2]\n"
    "fmla v18.4s, v28.4s, v7.s[2]\n"
    "fmla v19.4s, v27.4s, v9.s[2]\n"
    "fmla v20.4s, v28.4s, v9.s[2]\n"
    "fmla v21.4s, v27.4s, v11.s[2]\n"
    "fmla v22.4s, v28.4s, v11.s[2]\n"
    "fmla v23.4s, v27.4s, v13.s[2]\n"
    "fmla v24.4s, v28.4s, v13.s[2]\n"
// Filter width idx 6
    "fmla v15.4s, v25.4s, v6.s[2]\n"
    "fmla v16.4s, v26.4s, v6.s[2]\n"
    "fmla v17.4s, v25.4s, v8.s[2]\n"
    "fmla v18.4s, v26.4s, v8.s[2]\n"
    "fmla v19.4s, v25.4s, v10.s[2]\n"
    "fmla v20.4s, v26.4s, v10.s[2]\n"
    "fmla v21.4s, v25.4s, v12.s[2]\n"
    "fmla v22.4s, v26.4s, v12.s[2]\n"
    "fmla v23.4s, v25.4s, v14.s[2]\n"
    "fmla v24.4s, v26.4s, v14.s[2]\n"
    "stp q15, q16, [%[out], #0]\n"
    "stp q17, q18, [%[out], #32]\n"
    "stp q19, q20, [%[out], #64]\n"
    "stp q21, q22, [%[out], #96]\n"
    "stp q23, q24, [%[out], #128]\n"
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
void kernel_7_4_8_2_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11, 12]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
// Duplicate index: 15
// [1, 1, 2, 2, 3, 3, 4, 3, 3, 2, 2, 1, 1]
// Number of Input Regs: 13, Filter Regs: 4 Output Regs: 8
// Total number of registers required: 25
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
// Out - [13, 14, 15, 16, 17, 18, 19, 20]
// Fil - [21, 22, 23, 24, 21, 22, 23, 24, 21, 22, 23, 24, 21, 22, 23, 24, 21, 22]
// Register mapping diagram
//                       0  1  2  3  4  5  6  7  8  9 10 11 12 
//
//21 23 21 23 21 23 21  13    15    17    19    
//22 24 22 24 22 24 22  14    16    18    20    
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
            printf("%6.3f\t", *(inputPtr + 3*10 + j));
            printf("%6.3f\t", *(inputPtr + 3*11 + j));
            printf("%6.3f\t", *(inputPtr + 3*12 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 7; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*7*8 + wf*8 + i));
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
    "ldp q13, q14, [%[out], #0]\n"
    "ldp q15, q16, [%[out], #32]\n"
    "ldp q17, q18, [%[out], #64]\n"
    "ldp q19, q20, [%[out], #96]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
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

// K index 0
// Filter width idx 0
    "fmla v13.4s, v21.4s, v0.s[0]\n"
    "fmla v14.4s, v22.4s, v0.s[0]\n"
    "fmla v15.4s, v21.4s, v2.s[0]\n"
    "fmla v16.4s, v22.4s, v2.s[0]\n"
    "fmla v17.4s, v21.4s, v4.s[0]\n"
    "fmla v18.4s, v22.4s, v4.s[0]\n"
    "fmla v19.4s, v21.4s, v6.s[0]\n"
    "fmla v20.4s, v22.4s, v6.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v13.4s, v23.4s, v1.s[0]\n"
    "fmla v14.4s, v24.4s, v1.s[0]\n"
    "fmla v15.4s, v23.4s, v3.s[0]\n"
    "fmla v16.4s, v24.4s, v3.s[0]\n"
    "fmla v17.4s, v23.4s, v5.s[0]\n"
    "fmla v18.4s, v24.4s, v5.s[0]\n"
    "fmla v19.4s, v23.4s, v7.s[0]\n"
    "fmla v20.4s, v24.4s, v7.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v13.4s, v21.4s, v2.s[0]\n"
    "fmla v14.4s, v22.4s, v2.s[0]\n"
    "fmla v15.4s, v21.4s, v4.s[0]\n"
    "fmla v16.4s, v22.4s, v4.s[0]\n"
    "fmla v17.4s, v21.4s, v6.s[0]\n"
    "fmla v18.4s, v22.4s, v6.s[0]\n"
    "fmla v19.4s, v21.4s, v8.s[0]\n"
    "fmla v20.4s, v22.4s, v8.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v13.4s, v23.4s, v3.s[0]\n"
    "fmla v14.4s, v24.4s, v3.s[0]\n"
    "fmla v15.4s, v23.4s, v5.s[0]\n"
    "fmla v16.4s, v24.4s, v5.s[0]\n"
    "fmla v17.4s, v23.4s, v7.s[0]\n"
    "fmla v18.4s, v24.4s, v7.s[0]\n"
    "fmla v19.4s, v23.4s, v9.s[0]\n"
    "fmla v20.4s, v24.4s, v9.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v13.4s, v21.4s, v4.s[0]\n"
    "fmla v14.4s, v22.4s, v4.s[0]\n"
    "fmla v15.4s, v21.4s, v6.s[0]\n"
    "fmla v16.4s, v22.4s, v6.s[0]\n"
    "fmla v17.4s, v21.4s, v8.s[0]\n"
    "fmla v18.4s, v22.4s, v8.s[0]\n"
    "fmla v19.4s, v21.4s, v10.s[0]\n"
    "fmla v20.4s, v22.4s, v10.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v13.4s, v23.4s, v5.s[0]\n"
    "fmla v14.4s, v24.4s, v5.s[0]\n"
    "fmla v15.4s, v23.4s, v7.s[0]\n"
    "fmla v16.4s, v24.4s, v7.s[0]\n"
    "fmla v17.4s, v23.4s, v9.s[0]\n"
    "fmla v18.4s, v24.4s, v9.s[0]\n"
    "fmla v19.4s, v23.4s, v11.s[0]\n"
    "fmla v20.4s, v24.4s, v11.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v13.4s, v21.4s, v6.s[0]\n"
    "fmla v14.4s, v22.4s, v6.s[0]\n"
    "fmla v15.4s, v21.4s, v8.s[0]\n"
    "fmla v16.4s, v22.4s, v8.s[0]\n"
    "fmla v17.4s, v21.4s, v10.s[0]\n"
    "fmla v18.4s, v22.4s, v10.s[0]\n"
    "fmla v19.4s, v21.4s, v12.s[0]\n"
    "fmla v20.4s, v22.4s, v12.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v13.4s, v23.4s, v0.s[1]\n"
    "fmla v14.4s, v24.4s, v0.s[1]\n"
    "fmla v15.4s, v23.4s, v2.s[1]\n"
    "fmla v16.4s, v24.4s, v2.s[1]\n"
    "fmla v17.4s, v23.4s, v4.s[1]\n"
    "fmla v18.4s, v24.4s, v4.s[1]\n"
    "fmla v19.4s, v23.4s, v6.s[1]\n"
    "fmla v20.4s, v24.4s, v6.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v13.4s, v21.4s, v1.s[1]\n"
    "fmla v14.4s, v22.4s, v1.s[1]\n"
    "fmla v15.4s, v21.4s, v3.s[1]\n"
    "fmla v16.4s, v22.4s, v3.s[1]\n"
    "fmla v17.4s, v21.4s, v5.s[1]\n"
    "fmla v18.4s, v22.4s, v5.s[1]\n"
    "fmla v19.4s, v21.4s, v7.s[1]\n"
    "fmla v20.4s, v22.4s, v7.s[1]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v13.4s, v23.4s, v2.s[1]\n"
    "fmla v14.4s, v24.4s, v2.s[1]\n"
    "fmla v15.4s, v23.4s, v4.s[1]\n"
    "fmla v16.4s, v24.4s, v4.s[1]\n"
    "fmla v17.4s, v23.4s, v6.s[1]\n"
    "fmla v18.4s, v24.4s, v6.s[1]\n"
    "fmla v19.4s, v23.4s, v8.s[1]\n"
    "fmla v20.4s, v24.4s, v8.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v13.4s, v21.4s, v3.s[1]\n"
    "fmla v14.4s, v22.4s, v3.s[1]\n"
    "fmla v15.4s, v21.4s, v5.s[1]\n"
    "fmla v16.4s, v22.4s, v5.s[1]\n"
    "fmla v17.4s, v21.4s, v7.s[1]\n"
    "fmla v18.4s, v22.4s, v7.s[1]\n"
    "fmla v19.4s, v21.4s, v9.s[1]\n"
    "fmla v20.4s, v22.4s, v9.s[1]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v13.4s, v23.4s, v4.s[1]\n"
    "fmla v14.4s, v24.4s, v4.s[1]\n"
    "fmla v15.4s, v23.4s, v6.s[1]\n"
    "fmla v16.4s, v24.4s, v6.s[1]\n"
    "fmla v17.4s, v23.4s, v8.s[1]\n"
    "fmla v18.4s, v24.4s, v8.s[1]\n"
    "fmla v19.4s, v23.4s, v10.s[1]\n"
    "fmla v20.4s, v24.4s, v10.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v13.4s, v21.4s, v5.s[1]\n"
    "fmla v14.4s, v22.4s, v5.s[1]\n"
    "fmla v15.4s, v21.4s, v7.s[1]\n"
    "fmla v16.4s, v22.4s, v7.s[1]\n"
    "fmla v17.4s, v21.4s, v9.s[1]\n"
    "fmla v18.4s, v22.4s, v9.s[1]\n"
    "fmla v19.4s, v21.4s, v11.s[1]\n"
    "fmla v20.4s, v22.4s, v11.s[1]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v13.4s, v23.4s, v6.s[1]\n"
    "fmla v14.4s, v24.4s, v6.s[1]\n"
    "fmla v15.4s, v23.4s, v8.s[1]\n"
    "fmla v16.4s, v24.4s, v8.s[1]\n"
    "fmla v17.4s, v23.4s, v10.s[1]\n"
    "fmla v18.4s, v24.4s, v10.s[1]\n"
    "fmla v19.4s, v23.4s, v12.s[1]\n"
    "fmla v20.4s, v24.4s, v12.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v13.4s, v21.4s, v0.s[2]\n"
    "fmla v14.4s, v22.4s, v0.s[2]\n"
    "fmla v15.4s, v21.4s, v2.s[2]\n"
    "fmla v16.4s, v22.4s, v2.s[2]\n"
    "fmla v17.4s, v21.4s, v4.s[2]\n"
    "fmla v18.4s, v22.4s, v4.s[2]\n"
    "fmla v19.4s, v21.4s, v6.s[2]\n"
    "fmla v20.4s, v22.4s, v6.s[2]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v13.4s, v23.4s, v1.s[2]\n"
    "fmla v14.4s, v24.4s, v1.s[2]\n"
    "fmla v15.4s, v23.4s, v3.s[2]\n"
    "fmla v16.4s, v24.4s, v3.s[2]\n"
    "fmla v17.4s, v23.4s, v5.s[2]\n"
    "fmla v18.4s, v24.4s, v5.s[2]\n"
    "fmla v19.4s, v23.4s, v7.s[2]\n"
    "fmla v20.4s, v24.4s, v7.s[2]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v13.4s, v21.4s, v2.s[2]\n"
    "fmla v14.4s, v22.4s, v2.s[2]\n"
    "fmla v15.4s, v21.4s, v4.s[2]\n"
    "fmla v16.4s, v22.4s, v4.s[2]\n"
    "fmla v17.4s, v21.4s, v6.s[2]\n"
    "fmla v18.4s, v22.4s, v6.s[2]\n"
    "fmla v19.4s, v21.4s, v8.s[2]\n"
    "fmla v20.4s, v22.4s, v8.s[2]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v13.4s, v23.4s, v3.s[2]\n"
    "fmla v14.4s, v24.4s, v3.s[2]\n"
    "fmla v15.4s, v23.4s, v5.s[2]\n"
    "fmla v16.4s, v24.4s, v5.s[2]\n"
    "fmla v17.4s, v23.4s, v7.s[2]\n"
    "fmla v18.4s, v24.4s, v7.s[2]\n"
    "fmla v19.4s, v23.4s, v9.s[2]\n"
    "fmla v20.4s, v24.4s, v9.s[2]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v13.4s, v21.4s, v4.s[2]\n"
    "fmla v14.4s, v22.4s, v4.s[2]\n"
    "fmla v15.4s, v21.4s, v6.s[2]\n"
    "fmla v16.4s, v22.4s, v6.s[2]\n"
    "fmla v17.4s, v21.4s, v8.s[2]\n"
    "fmla v18.4s, v22.4s, v8.s[2]\n"
    "fmla v19.4s, v21.4s, v10.s[2]\n"
    "fmla v20.4s, v22.4s, v10.s[2]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v13.4s, v23.4s, v5.s[2]\n"
    "fmla v14.4s, v24.4s, v5.s[2]\n"
    "fmla v15.4s, v23.4s, v7.s[2]\n"
    "fmla v16.4s, v24.4s, v7.s[2]\n"
    "fmla v17.4s, v23.4s, v9.s[2]\n"
    "fmla v18.4s, v24.4s, v9.s[2]\n"
    "fmla v19.4s, v23.4s, v11.s[2]\n"
    "fmla v20.4s, v24.4s, v11.s[2]\n"
// Filter width idx 6
    "fmla v13.4s, v21.4s, v6.s[2]\n"
    "fmla v14.4s, v22.4s, v6.s[2]\n"
    "fmla v15.4s, v21.4s, v8.s[2]\n"
    "fmla v16.4s, v22.4s, v8.s[2]\n"
    "fmla v17.4s, v21.4s, v10.s[2]\n"
    "fmla v18.4s, v22.4s, v10.s[2]\n"
    "fmla v19.4s, v21.4s, v12.s[2]\n"
    "fmla v20.4s, v22.4s, v12.s[2]\n"
    "stp q13, q14, [%[out], #0]\n"
    "stp q15, q16, [%[out], #32]\n"
    "stp q17, q18, [%[out], #64]\n"
    "stp q19, q20, [%[out], #96]\n"
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
void kernel_7_3_8_2_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// Duplicate index: 10
// [1, 1, 2, 2, 3, 3, 3, 2, 2, 1, 1]
// Number of Input Regs: 11, Filter Regs: 14 Output Regs: 6
// Total number of registers required: 31
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// Out - [11, 12, 13, 14, 15, 16]
// Fil - [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 17, 18, 19, 20, 21, 22, 17, 18, 19, 20, 21, 22, 17, 18]
// Register mapping diagram
//                       0  1  2  3  4  5  6  7  8  9 10 
//
//17 19 21 23 25 27 29  11    13    15    
//18 20 22 24 26 28 30  12    14    16    
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
            printf("%6.3f\t", *(inputPtr + 3*10 + j));
            printf ("\n");
        }
    }
    printf ("Filter:\n");
    for (int wf = 0; wf < 7; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*7*8 + wf*8 + i));
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
    "ldp q11, q12, [%[out], #0]\n"
    "ldp q13, q14, [%[out], #32]\n"
    "ldp q15, q16, [%[out], #64]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
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

// K index 0
// Filter width idx 0
    "fmla v11.4s, v17.4s, v0.s[0]\n"
    "fmla v12.4s, v18.4s, v0.s[0]\n"
    "fmla v13.4s, v17.4s, v2.s[0]\n"
    "fmla v14.4s, v18.4s, v2.s[0]\n"
    "fmla v15.4s, v17.4s, v4.s[0]\n"
    "fmla v16.4s, v18.4s, v4.s[0]\n"
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v11.4s, v19.4s, v1.s[0]\n"
    "fmla v12.4s, v20.4s, v1.s[0]\n"
    "fmla v13.4s, v19.4s, v3.s[0]\n"
    "fmla v14.4s, v20.4s, v3.s[0]\n"
    "fmla v15.4s, v19.4s, v5.s[0]\n"
    "fmla v16.4s, v20.4s, v5.s[0]\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v11.4s, v21.4s, v2.s[0]\n"
    "fmla v12.4s, v22.4s, v2.s[0]\n"
    "fmla v13.4s, v21.4s, v4.s[0]\n"
    "fmla v14.4s, v22.4s, v4.s[0]\n"
    "fmla v15.4s, v21.4s, v6.s[0]\n"
    "fmla v16.4s, v22.4s, v6.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v11.4s, v23.4s, v3.s[0]\n"
    "fmla v12.4s, v24.4s, v3.s[0]\n"
    "fmla v13.4s, v23.4s, v5.s[0]\n"
    "fmla v14.4s, v24.4s, v5.s[0]\n"
    "fmla v15.4s, v23.4s, v7.s[0]\n"
    "fmla v16.4s, v24.4s, v7.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v11.4s, v25.4s, v4.s[0]\n"
    "fmla v12.4s, v26.4s, v4.s[0]\n"
    "fmla v13.4s, v25.4s, v6.s[0]\n"
    "fmla v14.4s, v26.4s, v6.s[0]\n"
    "fmla v15.4s, v25.4s, v8.s[0]\n"
    "fmla v16.4s, v26.4s, v8.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v11.4s, v27.4s, v5.s[0]\n"
    "fmla v12.4s, v28.4s, v5.s[0]\n"
    "fmla v13.4s, v27.4s, v7.s[0]\n"
    "fmla v14.4s, v28.4s, v7.s[0]\n"
    "fmla v15.4s, v27.4s, v9.s[0]\n"
    "fmla v16.4s, v28.4s, v9.s[0]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v11.4s, v29.4s, v6.s[0]\n"
    "fmla v12.4s, v30.4s, v6.s[0]\n"
    "fmla v13.4s, v29.4s, v8.s[0]\n"
    "fmla v14.4s, v30.4s, v8.s[0]\n"
    "fmla v15.4s, v29.4s, v10.s[0]\n"
    "fmla v16.4s, v30.4s, v10.s[0]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v11.4s, v17.4s, v0.s[1]\n"
    "fmla v12.4s, v18.4s, v0.s[1]\n"
    "fmla v13.4s, v17.4s, v2.s[1]\n"
    "fmla v14.4s, v18.4s, v2.s[1]\n"
    "fmla v15.4s, v17.4s, v4.s[1]\n"
    "fmla v16.4s, v18.4s, v4.s[1]\n"
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v11.4s, v19.4s, v1.s[1]\n"
    "fmla v12.4s, v20.4s, v1.s[1]\n"
    "fmla v13.4s, v19.4s, v3.s[1]\n"
    "fmla v14.4s, v20.4s, v3.s[1]\n"
    "fmla v15.4s, v19.4s, v5.s[1]\n"
    "fmla v16.4s, v20.4s, v5.s[1]\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v11.4s, v21.4s, v2.s[1]\n"
    "fmla v12.4s, v22.4s, v2.s[1]\n"
    "fmla v13.4s, v21.4s, v4.s[1]\n"
    "fmla v14.4s, v22.4s, v4.s[1]\n"
    "fmla v15.4s, v21.4s, v6.s[1]\n"
    "fmla v16.4s, v22.4s, v6.s[1]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v11.4s, v23.4s, v3.s[1]\n"
    "fmla v12.4s, v24.4s, v3.s[1]\n"
    "fmla v13.4s, v23.4s, v5.s[1]\n"
    "fmla v14.4s, v24.4s, v5.s[1]\n"
    "fmla v15.4s, v23.4s, v7.s[1]\n"
    "fmla v16.4s, v24.4s, v7.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v11.4s, v25.4s, v4.s[1]\n"
    "fmla v12.4s, v26.4s, v4.s[1]\n"
    "fmla v13.4s, v25.4s, v6.s[1]\n"
    "fmla v14.4s, v26.4s, v6.s[1]\n"
    "fmla v15.4s, v25.4s, v8.s[1]\n"
    "fmla v16.4s, v26.4s, v8.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v11.4s, v27.4s, v5.s[1]\n"
    "fmla v12.4s, v28.4s, v5.s[1]\n"
    "fmla v13.4s, v27.4s, v7.s[1]\n"
    "fmla v14.4s, v28.4s, v7.s[1]\n"
    "fmla v15.4s, v27.4s, v9.s[1]\n"
    "fmla v16.4s, v28.4s, v9.s[1]\n"
    "ld1 {v27.4s - v28.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v11.4s, v29.4s, v6.s[1]\n"
    "fmla v12.4s, v30.4s, v6.s[1]\n"
    "fmla v13.4s, v29.4s, v8.s[1]\n"
    "fmla v14.4s, v30.4s, v8.s[1]\n"
    "fmla v15.4s, v29.4s, v10.s[1]\n"
    "fmla v16.4s, v30.4s, v10.s[1]\n"
    "ld1 {v29.4s - v30.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v11.4s, v17.4s, v0.s[2]\n"
    "fmla v12.4s, v18.4s, v0.s[2]\n"
    "fmla v13.4s, v17.4s, v2.s[2]\n"
    "fmla v14.4s, v18.4s, v2.s[2]\n"
    "fmla v15.4s, v17.4s, v4.s[2]\n"
    "fmla v16.4s, v18.4s, v4.s[2]\n"
// Filter width idx 1
    "fmla v11.4s, v19.4s, v1.s[2]\n"
    "fmla v12.4s, v20.4s, v1.s[2]\n"
    "fmla v13.4s, v19.4s, v3.s[2]\n"
    "fmla v14.4s, v20.4s, v3.s[2]\n"
    "fmla v15.4s, v19.4s, v5.s[2]\n"
    "fmla v16.4s, v20.4s, v5.s[2]\n"
// Filter width idx 2
    "fmla v11.4s, v21.4s, v2.s[2]\n"
    "fmla v12.4s, v22.4s, v2.s[2]\n"
    "fmla v13.4s, v21.4s, v4.s[2]\n"
    "fmla v14.4s, v22.4s, v4.s[2]\n"
    "fmla v15.4s, v21.4s, v6.s[2]\n"
    "fmla v16.4s, v22.4s, v6.s[2]\n"
// Filter width idx 3
    "fmla v11.4s, v23.4s, v3.s[2]\n"
    "fmla v12.4s, v24.4s, v3.s[2]\n"
    "fmla v13.4s, v23.4s, v5.s[2]\n"
    "fmla v14.4s, v24.4s, v5.s[2]\n"
    "fmla v15.4s, v23.4s, v7.s[2]\n"
    "fmla v16.4s, v24.4s, v7.s[2]\n"
// Filter width idx 4
    "fmla v11.4s, v25.4s, v4.s[2]\n"
    "fmla v12.4s, v26.4s, v4.s[2]\n"
    "fmla v13.4s, v25.4s, v6.s[2]\n"
    "fmla v14.4s, v26.4s, v6.s[2]\n"
    "fmla v15.4s, v25.4s, v8.s[2]\n"
    "fmla v16.4s, v26.4s, v8.s[2]\n"
// Filter width idx 5
    "fmla v11.4s, v27.4s, v5.s[2]\n"
    "fmla v12.4s, v28.4s, v5.s[2]\n"
    "fmla v13.4s, v27.4s, v7.s[2]\n"
    "fmla v14.4s, v28.4s, v7.s[2]\n"
    "fmla v15.4s, v27.4s, v9.s[2]\n"
    "fmla v16.4s, v28.4s, v9.s[2]\n"
// Filter width idx 6
    "fmla v11.4s, v29.4s, v6.s[2]\n"
    "fmla v12.4s, v30.4s, v6.s[2]\n"
    "fmla v13.4s, v29.4s, v8.s[2]\n"
    "fmla v14.4s, v30.4s, v8.s[2]\n"
    "fmla v15.4s, v29.4s, v10.s[2]\n"
    "fmla v16.4s, v30.4s, v10.s[2]\n"
    "stp q11, q12, [%[out], #0]\n"
    "stp q13, q14, [%[out], #32]\n"
    "stp q15, q16, [%[out], #64]\n"
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
void kernel_7_2_8_2_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Duplicate index: 5
// [1, 1, 2, 2, 2, 2, 2, 1, 1]
// Number of Input Regs: 9, Filter Regs: 14 Output Regs: 4
// Total number of registers required: 27
// In  - [0, 1, 2, 3, 4, 5, 6, 7, 8]
// Out - [9, 10, 11, 12]
// Fil - [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14]
// Register mapping diagram
//                       0  1  2  3  4  5  6  7  8 
//
//13 15 17 19 21 23 25   9    11    
//14 16 18 20 22 24 26  10    12    
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
    for (int wf = 0; wf < 7; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*7*8 + wf*8 + i));
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
    "ldp q9, q10, [%[out], #0]\n"
    "ldp q11, q12, [%[out], #32]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v13.4s - v14.4s}, [x11], #32\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
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
    "fmla v9.4s, v13.4s, v0.s[0]\n"
    "fmla v10.4s, v14.4s, v0.s[0]\n"
    "fmla v11.4s, v13.4s, v2.s[0]\n"
    "fmla v12.4s, v14.4s, v2.s[0]\n"
    "ld1 {v13.4s - v14.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v9.4s, v15.4s, v1.s[0]\n"
    "fmla v10.4s, v16.4s, v1.s[0]\n"
    "fmla v11.4s, v15.4s, v3.s[0]\n"
    "fmla v12.4s, v16.4s, v3.s[0]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v9.4s, v17.4s, v2.s[0]\n"
    "fmla v10.4s, v18.4s, v2.s[0]\n"
    "fmla v11.4s, v17.4s, v4.s[0]\n"
    "fmla v12.4s, v18.4s, v4.s[0]\n"
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v9.4s, v19.4s, v3.s[0]\n"
    "fmla v10.4s, v20.4s, v3.s[0]\n"
    "fmla v11.4s, v19.4s, v5.s[0]\n"
    "fmla v12.4s, v20.4s, v5.s[0]\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v9.4s, v21.4s, v4.s[0]\n"
    "fmla v10.4s, v22.4s, v4.s[0]\n"
    "fmla v11.4s, v21.4s, v6.s[0]\n"
    "fmla v12.4s, v22.4s, v6.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v9.4s, v23.4s, v5.s[0]\n"
    "fmla v10.4s, v24.4s, v5.s[0]\n"
    "fmla v11.4s, v23.4s, v7.s[0]\n"
    "fmla v12.4s, v24.4s, v7.s[0]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v9.4s, v25.4s, v6.s[0]\n"
    "fmla v10.4s, v26.4s, v6.s[0]\n"
    "fmla v11.4s, v25.4s, v8.s[0]\n"
    "fmla v12.4s, v26.4s, v8.s[0]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v9.4s, v13.4s, v0.s[1]\n"
    "fmla v10.4s, v14.4s, v0.s[1]\n"
    "fmla v11.4s, v13.4s, v2.s[1]\n"
    "fmla v12.4s, v14.4s, v2.s[1]\n"
    "ld1 {v13.4s - v14.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v9.4s, v15.4s, v1.s[1]\n"
    "fmla v10.4s, v16.4s, v1.s[1]\n"
    "fmla v11.4s, v15.4s, v3.s[1]\n"
    "fmla v12.4s, v16.4s, v3.s[1]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v9.4s, v17.4s, v2.s[1]\n"
    "fmla v10.4s, v18.4s, v2.s[1]\n"
    "fmla v11.4s, v17.4s, v4.s[1]\n"
    "fmla v12.4s, v18.4s, v4.s[1]\n"
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v9.4s, v19.4s, v3.s[1]\n"
    "fmla v10.4s, v20.4s, v3.s[1]\n"
    "fmla v11.4s, v19.4s, v5.s[1]\n"
    "fmla v12.4s, v20.4s, v5.s[1]\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v9.4s, v21.4s, v4.s[1]\n"
    "fmla v10.4s, v22.4s, v4.s[1]\n"
    "fmla v11.4s, v21.4s, v6.s[1]\n"
    "fmla v12.4s, v22.4s, v6.s[1]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v9.4s, v23.4s, v5.s[1]\n"
    "fmla v10.4s, v24.4s, v5.s[1]\n"
    "fmla v11.4s, v23.4s, v7.s[1]\n"
    "fmla v12.4s, v24.4s, v7.s[1]\n"
    "ld1 {v23.4s - v24.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v9.4s, v25.4s, v6.s[1]\n"
    "fmla v10.4s, v26.4s, v6.s[1]\n"
    "fmla v11.4s, v25.4s, v8.s[1]\n"
    "fmla v12.4s, v26.4s, v8.s[1]\n"
    "ld1 {v25.4s - v26.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v9.4s, v13.4s, v0.s[2]\n"
    "fmla v10.4s, v14.4s, v0.s[2]\n"
    "fmla v11.4s, v13.4s, v2.s[2]\n"
    "fmla v12.4s, v14.4s, v2.s[2]\n"
// Filter width idx 1
    "fmla v9.4s, v15.4s, v1.s[2]\n"
    "fmla v10.4s, v16.4s, v1.s[2]\n"
    "fmla v11.4s, v15.4s, v3.s[2]\n"
    "fmla v12.4s, v16.4s, v3.s[2]\n"
// Filter width idx 2
    "fmla v9.4s, v17.4s, v2.s[2]\n"
    "fmla v10.4s, v18.4s, v2.s[2]\n"
    "fmla v11.4s, v17.4s, v4.s[2]\n"
    "fmla v12.4s, v18.4s, v4.s[2]\n"
// Filter width idx 3
    "fmla v9.4s, v19.4s, v3.s[2]\n"
    "fmla v10.4s, v20.4s, v3.s[2]\n"
    "fmla v11.4s, v19.4s, v5.s[2]\n"
    "fmla v12.4s, v20.4s, v5.s[2]\n"
// Filter width idx 4
    "fmla v9.4s, v21.4s, v4.s[2]\n"
    "fmla v10.4s, v22.4s, v4.s[2]\n"
    "fmla v11.4s, v21.4s, v6.s[2]\n"
    "fmla v12.4s, v22.4s, v6.s[2]\n"
// Filter width idx 5
    "fmla v9.4s, v23.4s, v5.s[2]\n"
    "fmla v10.4s, v24.4s, v5.s[2]\n"
    "fmla v11.4s, v23.4s, v7.s[2]\n"
    "fmla v12.4s, v24.4s, v7.s[2]\n"
// Filter width idx 6
    "fmla v9.4s, v25.4s, v6.s[2]\n"
    "fmla v10.4s, v26.4s, v6.s[2]\n"
    "fmla v11.4s, v25.4s, v8.s[2]\n"
    "fmla v12.4s, v26.4s, v8.s[2]\n"
    "stp q9, q10, [%[out], #0]\n"
    "stp q11, q12, [%[out], #32]\n"
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
void kernel_7_1_8_2_1_0_0_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)
{
// Input index per position
// [[0, 1, 2, 3, 4, 5, 6]]
// Input registers required
// [0, 1, 2, 3, 4, 5, 6]
// Duplicate index: 0
// [1, 1, 1, 1, 1, 1, 1]
// Number of Input Regs: 7, Filter Regs: 14 Output Regs: 2
// Total number of registers required: 23
// In  - [0, 1, 2, 3, 4, 5, 6]
// Out - [7, 8]
// Fil - [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10]
// Register mapping diagram
//                       0  1  2  3  4  5  6 
//
// 9 11 13 15 17 19 21   7    
//10 12 14 16 18 20 22   8    
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
    for (int wf = 0; wf < 7; wf++)
    {
        printf("Wfil %d:\n", wf);
        for (int i = 0; i < 8; i++)
        {
            printf("Row %d:\t", i);
            for (int j = 0; j < k; j++)
            {
                printf("%6.3f\t", *(filterPtr + j*7*8 + wf*8 + i));
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
    "ldp q7, q8, [%[out], #0]\n"
    "mov x10, %[in]\n"
 // Load filters
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
    "ld1 {v11.4s - v12.4s}, [x11], #32\n"
    "ld1 {v13.4s - v14.4s}, [x11], #32\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
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
    "fmla v7.4s, v9.4s, v0.s[0]\n"
    "fmla v8.4s, v10.4s, v0.s[0]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v7.4s, v11.4s, v1.s[0]\n"
    "fmla v8.4s, v12.4s, v1.s[0]\n"
    "ld1 {v11.4s - v12.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v7.4s, v13.4s, v2.s[0]\n"
    "fmla v8.4s, v14.4s, v2.s[0]\n"
    "ld1 {v13.4s - v14.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v7.4s, v15.4s, v3.s[0]\n"
    "fmla v8.4s, v16.4s, v3.s[0]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v7.4s, v17.4s, v4.s[0]\n"
    "fmla v8.4s, v18.4s, v4.s[0]\n"
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v7.4s, v19.4s, v5.s[0]\n"
    "fmla v8.4s, v20.4s, v5.s[0]\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v7.4s, v21.4s, v6.s[0]\n"
    "fmla v8.4s, v22.4s, v6.s[0]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 1
// Filter width idx 0
    "fmla v7.4s, v9.4s, v0.s[1]\n"
    "fmla v8.4s, v10.4s, v0.s[1]\n"
    "ld1 {v9.4s - v10.4s}, [x11], #32\n"
// Filter width idx 1
    "fmla v7.4s, v11.4s, v1.s[1]\n"
    "fmla v8.4s, v12.4s, v1.s[1]\n"
    "ld1 {v11.4s - v12.4s}, [x11], #32\n"
// Filter width idx 2
    "fmla v7.4s, v13.4s, v2.s[1]\n"
    "fmla v8.4s, v14.4s, v2.s[1]\n"
    "ld1 {v13.4s - v14.4s}, [x11], #32\n"
// Filter width idx 3
    "fmla v7.4s, v15.4s, v3.s[1]\n"
    "fmla v8.4s, v16.4s, v3.s[1]\n"
    "ld1 {v15.4s - v16.4s}, [x11], #32\n"
// Filter width idx 4
    "fmla v7.4s, v17.4s, v4.s[1]\n"
    "fmla v8.4s, v18.4s, v4.s[1]\n"
    "ld1 {v17.4s - v18.4s}, [x11], #32\n"
// Filter width idx 5
    "fmla v7.4s, v19.4s, v5.s[1]\n"
    "fmla v8.4s, v20.4s, v5.s[1]\n"
    "ld1 {v19.4s - v20.4s}, [x11], #32\n"
// Filter width idx 6
    "fmla v7.4s, v21.4s, v6.s[1]\n"
    "fmla v8.4s, v22.4s, v6.s[1]\n"
    "ld1 {v21.4s - v22.4s}, [x11], #32\n"
// K index 2
// Filter width idx 0
    "fmla v7.4s, v9.4s, v0.s[2]\n"
    "fmla v8.4s, v10.4s, v0.s[2]\n"
// Filter width idx 1
    "fmla v7.4s, v11.4s, v1.s[2]\n"
    "fmla v8.4s, v12.4s, v1.s[2]\n"
// Filter width idx 2
    "fmla v7.4s, v13.4s, v2.s[2]\n"
    "fmla v8.4s, v14.4s, v2.s[2]\n"
// Filter width idx 3
    "fmla v7.4s, v15.4s, v3.s[2]\n"
    "fmla v8.4s, v16.4s, v3.s[2]\n"
// Filter width idx 4
    "fmla v7.4s, v17.4s, v4.s[2]\n"
    "fmla v8.4s, v18.4s, v4.s[2]\n"
// Filter width idx 5
    "fmla v7.4s, v19.4s, v5.s[2]\n"
    "fmla v8.4s, v20.4s, v5.s[2]\n"
// Filter width idx 6
    "fmla v7.4s, v21.4s, v6.s[2]\n"
    "fmla v8.4s, v22.4s, v6.s[2]\n"
    "stp q7, q8, [%[out], #0]\n"
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
