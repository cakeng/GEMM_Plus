#! /usr/bin/env python3 
import sys
import os

def generateKernel(wf, wt, co, stride, dilation, skipStart, skipEnd):
    print ("void kernel_%s_%s_%s_%s_%s_%s_%s_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)"%(wf, wt, co*4, stride, dilation, skipStart, skipEnd))
    print('{')

    # Finding the input index # on each output position.
    inputIdxPerPos=[]
    for i in range(wt):
        WinStart = i*stride
        index=[]
        for j in range(wf):
            index.append(WinStart + j*dilation)
        inputIdxPerPos.append(index)   
    print ("// Input index per position")
    print ('// ',end='')
    print(inputIdxPerPos)

    # Finding the entry of input registers
    regInPos = []
    dups = 0
    for index in inputIdxPerPos:
        for pos in index:
            if regInPos.count(pos) == 0:
                regInPos.append(pos)
            else:
                dups += 1
    regInPos.sort()
    print ("// Input registers required")
    print ('// ',end='')
    print(regInPos)
    # Finding the number of reuses per each input register
    dupPos = [0]*len(regInPos)
    for index in inputIdxPerPos:
        for pos in index:
            dupPos[regInPos.index(pos)] += 1
    print ("// Duplicate index: " + str(dups))
    print ('// ',end='')
    print(dupPos)

    # Finding the number of filter entries required.
    fnum = wf
    if (2 <= co and 3 < wt):
        fnum = 2
    print ("// Number of Input Regs: %d, Filter Regs: %d Output Regs: %d"%(len(regInPos), fnum*co, wt*co))
    print ("// Total number of registers required: " + str(len(regInPos) + fnum*co + wt*co))
    if (len(regInPos) + fnum*co + wt*co) > 32:
        print ('Unsupported number of registers!')
        sys.exit(0)

    # Setting the registers.
    regIn = list(range(len(regInPos)))
    regOut = list(range(len(regIn), len(regIn) + wt*co))
    regFil = list(range(len(regIn) + len(regOut), len(regIn) + len(regOut) +  fnum*co))
    for wfIdx in range(co*wf):
        regFil.append(regFil[wfIdx%(wt*co)])
    print ('// In  - ',end='')
    print (regIn)
    print ('// Out - ',end='')
    print (regOut)
    print ('// Fil - ',end='')
    print (regFil)
    
    print ('// Register mapping diagram')
    print ('// ',end='')
    for wfIdx in range(wf):
        print('   ',end='')
    for wiIdx in regIn:
        print('%2s '%(wiIdx),end='')
    print ('')
    print ('//')
    print ('//',end='')
    for coIdx in range(co):
        for wfIdx in range(wf):
            print('%2s '%(regFil[wfIdx*co + coIdx]),end='')
        print (' ',end='')
        for wtIdx in range(wt):
            print('%2s '%(regOut[wtIdx*co + coIdx]),end='')
            print ('   '*(stride-1),end='')
        print ('')
        print ('//',end='')

    print ('')
    print ('    #ifdef __DEBUG_PTMM_OFF')
    print ('    printf (\"Input:\\n\");')
    # print ('    for (int i = 0; i < (k/%s); i++)'%(co*4))
    print ('    {')
    print ('        for (int j = 0; j < %s; j++)'%(3))
    print ('        {')
    print ('            printf(\"Row %d:\\t\", j);')
    for wiIdx in regInPos:
        print ('            printf(\"%%6.3f\\t\", *(inputPtr + %s*%s + j));'%(3, regIn[regInPos.index(wiIdx)]))
    print ('            printf (\"\\n\");')
    print ('        }')
    print ('    }')
    print ('    printf (\"Filter:\\n\");')
    print ('    for (int wf = 0; wf < %s; wf++)'%(wf))
    print ('    {')
    print ('        printf(\"Wfil %d:\\n\", wf);')
    print ('        for (int i = 0; i < %s; i++)'%(co*4))
    print ('        {')
    print ('            printf(\"Row %d:\\t\", i);')
    print ('            for (int j = 0; j < k; j++)')
    print ('            {')
    print ('                printf(\"%%6.3f\\t\", *(filterPtr + j*%s*%s + wf*%s + i));'%(wf, co*4, co*4))
    print ('            }')
    print ('            printf(\"\\n\");')
    print ('        }')
    print ('    }')
    print ('    printf (\"Output:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Row %d:\\t\", i);')
    for woIdx in range(int(len(regOut) / co)):
        print ('        printf(\"%%6.3f\\t\", *(outputPtr + %s*%s + i));'%(co*4, woIdx))
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"\\n\");')
    print ('    #endif')
    print ('')

    print ("    float* in = inputPtr + %s*%s;"%(skipStart, 3))
    print ("    float* fil = filterPtr;")
    print ("    float* out = outputPtr;")
    print ('    {')
    print ('    __asm __volatile (')
    print('// Prefetch input and filter')
    print ('    \"prfm pldl1keep, [%[fil], #0]\\n\"')
    print ('    \"prfm pldl1keep, [%[fil], #64]\\n\"')
    print ('    \"prfm pldl1keep, [%[fil], #128]\\n\"')
    print ('    \"prfm pldl1keep, [%[fil], #192]\\n\"')
    inPfNum = wt*3//16
    for inPfIdx in range(inPfNum):
        print ('    \"prfm pldl1keep, [%%[in], #%s]\\n\"'%(inPfIdx*64))
    print ('    \"mov x11, %[fil]\\n\"')
    print('// Load Output')
    for i in range(wt):
        if co == 2:
            print ('    \"ldp q%s, q%s, [%%[out], #%s]\\n\"'%(regOut[i*co + 0], regOut[i*co + 1], i*8*4))
        elif co == 1:
            print ('    \"ld1 {v%s.4s}, [%%[out], #%s]\\n\"'%(regOut[i*co], i*4*4))
    print ('    \"mov x10, %[in]\\n\"')
    print(' // Load filters')
    for i in range(fnum):
        if co == 2:
            print ('    \"ld1 {v%s.4s - v%s.4s}, [x11], #32\\n\"'%(regFil[i*co + 0], regFil[i*co + 1]))
        elif co == 1:
            print ('    \"ldr q%s, [x11], #16\\n\"'%(regFil[wfIdx*co + 0]))
    print('// Load Input')
    for wiIdx in regInPos:
        if skipStart <= wiIdx and wiIdx <= len(regInPos) - 1 - skipEnd:
            print ('    \"ldr q%s, [x10, #%s]\\n\"'%(wiIdx, 12*(wiIdx - skipStart)))
    print ('')

    fidx = 0
    inPfIdx=0
    for i in range(1):
        for k in range (3):
            print('// K index %s'%(i*4 + k))
            for wfIdx in range(wf):
                print('// Filter width idx %s'%(wfIdx))
                for wtIdx in range(wt):
                    iPos = regInPos.index(inputIdxPerPos[wtIdx][wfIdx])
                    ireg = regIn[iPos]
                    for coIdx in range(co):
                        freg = regFil[(fidx%fnum)*co + coIdx]     
                        oreg = regOut[wtIdx*co + coIdx]
                        if skipStart <= iPos and iPos <= len(regInPos) - 1 - skipEnd:
                            print ('    \"fmla v%s.4s, v%s.4s, v%s.s[%s]\\n\"'%(oreg, freg, ireg, k))
                if k != 2 or wfIdx < wf-fnum:
                    if co == 1:
                        print ('    \"ldr q%s, [x11], #16\\n\"'%(regFil[(fidx%fnum)*co]))
                    elif co == 2:
                        print ('    \"ld1 {v%s.4s - v%s.4s}, [x11], #32\\n\"'%(regFil[(fidx%fnum)*co], regFil[(fidx%fnum)*co + 1]))
                fidx = fidx + 1

    for i in range(wt):
        if co == 2:
            print ('    \"stp q%s, q%s, [%%[out], #%s]\\n\"'%(regOut[i*co + 0], regOut[i*co + 1], i*8*4))
        elif co == 1:
            print ('    \"st1 {v%s.4s}, [%%[out], #%s]\\n\"'%(regOut[i*co], i*4*4))
    print ('    :')
    print ('    :   ', end='')
    print ('        [in] \"r\" (in), [fil] \"r\" (fil), [out] \"r\" (out), [inStr] \"r\" (inStride*sizeof(float))')
    print ('    :   \"x8\", \"x9\", \"x10\", \"x11\",')
    print ('        \"v0\", \"v1\", \"v2\", \"v3\", \"v4\", \"v5\", \"v6\", \"v7\", \"v8\", \"v9\", \"v10\", \"v11\", \"v12\", \"v13\", \"v14\", \"v15\",')
    print ('        \"v16\", \"v17\", \"v18\", \"v19\", \"v20\", \"v21\", \"v22\", \"v23\", \"v24\", \"v25\", \"v26\", \"v27\", \"v28\", \"v29\", \"v30\", \"v31\", \"cc\", \"memory\"')
    print ('    );')
    print ('    }')
    print ('    #ifdef __DEBUG_PTMM_OFF')
    print ('    printf (\"Output After Kernel:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Row %d:\\t\", i);')
    for woIdx in range(int(len(regOut) / co)):
        print ('        printf(\"%%6.3f\\t\", *(outputPtr + %s*%s + i));'%(co*4, woIdx))
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"\\n\");')
    print ('    #endif')
    print('}')

def generateKernelCoWt(wf, wt, co, stride, dilation, skipStart, skipEnd):
    print ("void kernel_%s_%s_%s_%s_%s_%s_%s_ci3(float* inputPtr, float* filterPtr, float* outputPtr, const int k, const int inStride)"%(wf, wt, co*4, stride, dilation, skipStart, skipEnd))
    print('{')

    # Finding the input index # on each output position.
    inputIdxPerPos=[]
    for i in range(wt):
        WinStart = i*stride
        index=[]
        for j in range(wf):
            index.append(WinStart + j*dilation)
        inputIdxPerPos.append(index)   
    print ("// Input index per position")
    print ('// ',end='')
    print(inputIdxPerPos)

    # Finding the entry of input registers
    regInPos = []
    dups = 0
    for index in inputIdxPerPos:
        for pos in index:
            if regInPos.count(pos) == 0:
                regInPos.append(pos)
            else:
                dups += 1
    regInPos.sort()
    print ("// Input registers required")
    print ('// ',end='')
    print(regInPos)
    # Finding the number of reuses per each input register
    dupPos = [0]*len(regInPos)
    for index in inputIdxPerPos:
        for pos in index:
            dupPos[regInPos.index(pos)] += 1
    print ("// Duplicate index: " + str(dups))
    print ('// ',end='')
    print(dupPos)

    # Finding the number of filter entries required.
    fnum = wf
    if (2 <= co and 5 < wt):
        fnum = 1
    print ("// Number of Input Regs: %d, Filter Regs: %d Output Regs: %d"%(len(regInPos), fnum*co, wt*co))
    print ("// Total number of registers required: " + str(len(regInPos) + fnum*co + wt*co))
    if (len(regInPos) + fnum*co + wt*co) > 32:
        print ('Unsupported number of registers!')
        sys.exit(0)

    # Setting the registers.
    regIn = list(range(len(regInPos)))
    regOut = list(range(len(regIn), len(regIn) + wt*co))
    regFil = list(range(len(regIn) + len(regOut), len(regIn) + len(regOut) +  fnum*co))
    for wfIdx in range(co*wf):
        regFil.append(regFil[wfIdx%(wt*co)])
    print ('// In  - ',end='')
    print (regIn)
    print ('// Out - ',end='')
    print (regOut)
    print ('// Fil - ',end='')
    print (regFil)
    
    print ('// Register mapping diagram')
    print ('// ',end='')
    for wfIdx in range(wf):
        print('   ',end='')
    for wiIdx in regIn:
        print('%2s '%(wiIdx),end='')
    print ('')
    print ('//')
    print ('//',end='')
    for coIdx in range(co):
        for wfIdx in range(wf):
            print('%2s '%(regFil[wfIdx*co + coIdx]),end='')
        print (' ',end='')
        for wtIdx in range(wt):
            print('%2s '%(regOut[wtIdx*co + coIdx]),end='')
            print ('   '*(stride-1),end='')
        print ('')
        print ('//',end='')

    print ('')
    print ('    #ifdef __DEBUG_PTMM_OFF')
    print ('    printf (\"Input:\\n\");')
    print ('    for (int i = 0; i < 1; i++)')
    print ('    {')
    print ('        for (int j = 0; j < 3; j++)')
    print ('        {')
    print ('            printf(\"Row %%d:\\t\", i*%s + j);'%(co*4))
    for wiIdx in regInPos:
        print ('            printf(\"%%6.3f\\t\", *(inputPtr + i*inStride + %s*%s + j));'%(3, regIn[regInPos.index(wiIdx)]))
    print ('            printf (\"\\n\");')
    print ('        }')
    print ('    }')
    print ('    printf (\"Filter:\\n\");')
    print ('    for (int wf = 0; wf < %s; wf++)'%(wf))
    print ('    {')
    print ('        printf(\"Wfil %d:\\n\", wf);')
    print ('        for (int i = 0; i < %s; i++)'%(co*4))
    print ('        {')
    print ('            printf(\"Row %d:\\t\", i);')
    print ('            for (int j = 0; j < k; j++)')
    print ('            {')
    print ('                printf(\"%%6.3f\\t\", *(filterPtr + j*%s*%s + wf*%s + i));'%(wf, co*4, co*4))
    print ('            }')
    print ('            printf(\"\\n\");')
    print ('        }')
    print ('    }')
    print ('    printf (\"Output:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Row %d:\\t\", i);')
    for woIdx in range(int(len(regOut) / co)):
        print ('        printf(\"%%6.3f\\t\", *(outputPtr + %s*%s + i));'%(co*4, woIdx))
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"\\n\");')
    print ('    #endif')
    print ('')

    print ("    float* in = inputPtr + %s*%s;"%(skipStart, 3))
    print ("    float* fil = filterPtr;")
    print ("    float* out = outputPtr;")
    print ('    {')
    print ('    __asm __volatile (')
    print ('    \"add x8, %[in], %[inStr]\\n\"')
    print('// Prefetch input and filter')
    print ('    \"prfm pldl1keep, [%[fil], #0]\\n\"')
    print ('    \"prfm pldl1keep, [%[fil], #64]\\n\"')
    print ('    \"prfm pldl1keep, [%[fil], #128]\\n\"')
    print ('    \"prfm pldl1keep, [%[fil], #192]\\n\"')
    inPfNum = (regInPos[len(regInPos)-1] + 1 - (skipStart + skipEnd))*co
    inPfNum = wt*stride*3//16
    for inPfIdx in range(inPfNum):
        print ('    \"prfm pldl1keep, [%%[in], #%s]\\n\"'%(inPfIdx*64))
    print ('    \"mov x11, %[fil]\\n\"')
    print('// Load Output')
    for i in range(wt):
        if co == 2:
            print ('    \"ldp q%s, q%s, [%%[out], #%s]\\n\"'%(regOut[i*co + 0], regOut[i*co + 1], i*8*4))
        elif co == 1:
            print ('    \"ld1 {v%s.4s}, [%%[out], #%s]\\n\"'%(regOut[i*co], i*4*4))
    print ('    \"mov x10, %[in]\\n\"')
    print(' // Load filters')
    for i in range(fnum):
        if co == 2:
            print ('    \"ld1 {v%s.4s - v%s.4s}, [x11], #32\\n\"'%(regFil[i*co + 0], regFil[i*co + 1]))
        elif co == 1:
            print ('    \"ldr q%s, [x11], #16\\n\"'%(regFil[wfIdx*co + 0]))
    print('// Load Input')
    for wiIdx in regInPos:
        if skipStart <= wiIdx and wiIdx <= len(regInPos) - 1 - skipEnd:
            print ('    \"ldr q%s, [x10, #%s]\\n\"'%(wiIdx, 12*(wiIdx - skipStart)))
    print ('')

    fidx = 0
    inPfIdx=0
    inPfCounter=0
    print('// Remaining 3 channels')
    for i in range(1):
        for k in range (3):
            kInLoaded = dupPos.copy()
            print('// K index %s'%(i*4 + k))
            for wfIdx in range(wf):
                print('// Filter width idx %s'%(wfIdx))
                for coIdx in range(co):
                    for wtIdx in range(wt):
                        iPos = regInPos.index(inputIdxPerPos[wtIdx][wfIdx])
                        ireg = regIn[iPos]
                        freg = regFil[(fidx%fnum)*co + coIdx]     
                        oreg = regOut[wtIdx*co + coIdx]
                        if skipStart <= iPos and iPos <= len(regInPos) - 1 - skipEnd:
                            print ('    \"fmla v%s.4s, v%s.4s, v%s.s[%s]\\n\"'%(oreg, freg, ireg, k))
                        if k == 3 and coIdx == co-1:
                            kInLoaded[ireg] -= 1
                            if kInLoaded[ireg] == 0:
                                if skipStart <= iPos and iPos <= len(regInPos) - 1 - skipEnd and i != co-1:
                                    print ('    \"ldr q%s, [x10, #%s]\\n\"'%(ireg, 16*(co-1 - i) + 32*(regInPos[iPos] - skipStart)))
                    print ('    \"ldr q%s, [x11], #16\\n\"'%(regFil[fidx%(fnum*co)]))
                    fidx = fidx + 1
    for i in range(wt):
        if co == 2:
            print ('    \"stp q%s, q%s, [%%[out], #%s]\\n\"'%(regOut[i*co + 0], regOut[i*co + 1], i*8*4))
        elif co == 1:
            print ('    \"st1 {v%s.4s}, [%%[out], #%s]\\n\"'%(regOut[i*co], i*4*4))
    print ('    :')
    print ('    :   ', end='')
    print ('        [in] \"r\" (in), [fil] \"r\" (fil), [out] \"r\" (out), [k] \"r\" (k), [inStr] \"r\" (inStride*sizeof(float))')
    print ('    :   \"x8\", \"x9\", \"x10\", \"x11\",')
    print ('        \"v0\", \"v1\", \"v2\", \"v3\", \"v4\", \"v5\", \"v6\", \"v7\", \"v8\", \"v9\", \"v10\", \"v11\", \"v12\", \"v13\", \"v14\", \"v15\",')
    print ('        \"v16\", \"v17\", \"v18\", \"v19\", \"v20\", \"v21\", \"v22\", \"v23\", \"v24\", \"v25\", \"v26\", \"v27\", \"v28\", \"v29\", \"v30\", \"v31\", \"cc\", \"memory\"')
    print ('    );')
    print ('    }')
    print ('    #ifdef __DEBUG_PTMM_OFF')
    print ('    printf (\"Output After Kernel:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Row %d:\\t\", i);')
    for woIdx in range(int(len(regOut) / co)):
        print ('        printf(\"%%6.3f\\t\", *(outputPtr + %s*%s + i));'%(co*4, woIdx))
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"\\n\");')
    print ('    #endif')
    print('}')

#def generateKernel(wf, wt, co, stride, dilation, skipStart, skipEnd):

# generateKernel (3, 7, 2, 1, 1, 0, 0)
# generateKernel (3, 7, 2, 1, 1, 1, 0)
# generateKernel (3, 7, 2, 1, 1, 0, 1)

# generateKernel (3, 8, 2, 1, 1, 0, 0)
# generateKernel (3, 8, 2, 1, 1, 1, 0)
# generateKernel (3, 8, 2, 1, 1, 0, 1)

# generateKernelCoWt (3, 7, 2, 2, 1, 1, 0)
# generateKernelCoWt (3, 7, 2, 2, 1, 0, 1)
# generateKernelCoWt (3, 7, 2, 2, 1, 0, 0)

generateKernel (7, 5, 2, 2, 1, 3, 0)
generateKernel (7, 5, 2, 2, 1, 0, 2)

for wti in range(5,0,-1):
    generateKernel (7, wti, 2, 2, 1, 0, 0)

# for wti in range(7,0,-1):
#     generateKernel (7, wti, 2, 1, 1, 0, 0)

