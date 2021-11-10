#! /usr/bin/env python3 
import sys
import os

def generateKernelDepthAllH(wf, wo, co, stride, dilation, padding):
    print ("void kernel_depth_%s_%s_%s_%s_%s_%s_%s_%s(float* inputPtr, float* filterPtr, float* outputPtr, float* biasPtr)"%(wf, wo, co*4, stride, dilation, padding, 1, 1))
    print('{')
    outNum = 3
    ho = wo
    hf = wf
    regIn = list(range(hf*co))
    regFil = list(range(hf*co, hf*co + hf*wf*co))
    regOut = list(range(hf*co + hf*wf*co, hf*co + hf*wf*co + outNum*co))
    regBias = list(range(hf*co + hf*wf*co + outNum*co, hf*co + hf*wf*co + outNum*co + co))
    print ('// In  - ',end='')
    print (regIn)
    print ('// Out - ',end='')
    print (regOut)
    print ('// Fil - ',end='')
    print (regFil)
    print ('// Bias - ',end='')
    print (regBias)
    
    # Finding the input index # on each output position.
    inputIdxPerPos=[]
    for i in range(wo):
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
    wiStart = -padding
    hiStart = -padding
    wi = wo*stride
    hi = ho*stride

    print ('// Register mapping diagram')
    print ('// ',end='')
    for wfIdx in range(wf*wf):
        print('   ',end='')
    for wiIdx in regIn:
        print('%2s '%(wiIdx),end='')
    print ('')
    print ('//')
    print ('//',end='')
    for coIdx in range(co):
        for wfIdx in range(wf*wf):
            print('%2s '%(regFil[wfIdx*co + coIdx]),end='')
        print (' ',end='')
        for wtIdx in range(wo):
            print('%2s '%(regOut[(wtIdx%outNum)*co + coIdx]),end='')
            print ('   '*(stride-1),end='')
        print ('')
        print ('//',end='')
    print ('')

    print ('')
    print ('    #ifdef __DEBUG_PTMM_OFF')
    print ('    printf (\"Input:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Ci %d:\\t\", i);')
    print ('        for (int j = 0; j < %s; j++)'%(wi))
    print ('        {')
    print ('            printf(\"%%6.3f\\t\", *(inputPtr + j*%s + i));'%(co*4))
    print ('        }')
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"Filter:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Ci %d:\\t\", i);')
    print ('        for (int fi = 0; fi < %s; fi++)'%(hf*wf))
    print ('        {')
    print ('            printf(\"%%6.3f\\t\", *(filterPtr + fi*%s + i));'%(co*4))
    print ('        }')
    print ('        printf(\"\\n\");')
    print ('    }')
    print ('    printf (\"Output:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Ci %d:\\t\", i);')
    print ('        for (int j = 0; j < %s; j++)'%(wo))
    print ('        {')
    print ('            printf(\"%%6.3f\\t\", *(outputPtr + j*%s + i));'%(co*4))
    print ('        }')
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"\\n\");')
    print ('    #endif')
    print ('')

    hiLoopCnt = 1
    hiLoopStart = 0
    hiIdx = hiStart
    hiNoSkipNum = 0
    while hiNoSkipNum != wf:
        hiNoSkipNum = 0
        for hfIdx in range(hf):
            if (0 <= (hiIdx+hfIdx*dilation) and (hiIdx+hfIdx*dilation) < hi):
                hiNoSkipNum = hiNoSkipNum + 1
        if hiNoSkipNum != wf:
            hiIdx = hiIdx + 1
    hiLoopStart = hiIdx
    hiNoSkipNum = wf
    while hiNoSkipNum == wf:
        hiNoSkipNum = 0
        for hfIdx in range(hf):
            if (0 <= (hiIdx+hfIdx*dilation) and (hiIdx+hfIdx*dilation) < hi):
                hiNoSkipNum = hiNoSkipNum + 1
        if hiNoSkipNum == wf:
            hiIdx = hiIdx + 1
            hiLoopCnt = hiLoopCnt + 1
    hiIdx -= 1
    hiLoopCnt -= 1
    print('// Hi has paddings on indexes %s ~ %s, No paddings until index %s, %s indexes (%s ~ %s), and has paddings for the remaining indexes.'%(hiStart, hiLoopStart-1, hiIdx, hiLoopCnt, hiLoopStart, hiLoopStart+hiLoopCnt-1))
    

    print ('    {')
    print ('    __asm __volatile (')
    print('// Loading Bias.')
    print ('    \"ld1 {v%s.4s - v%s.4s}, [%%[bias]]\\n\"'%(regBias[0], regBias[1]))
    print ('    \"mov x9, #%s\\n\"'%(wi*32))
    print('// Loading Fiters.')
    for hfIdx in range(hf):
        for wfIdx in range(wf):
            print ('    \"ldp q%s, q%s, [%%[fil], #%s]\\n\"'%(regFil[(hfIdx*wf + wfIdx)*co + 0], regFil[(hfIdx*wf + wfIdx)*co + 1], 32*(hfIdx*wf + wfIdx)))
    print('// Biasing Outputs.')
    for coIdx in range(co):
        for oN in range(outNum):
            print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[oN*co + coIdx], regBias[coIdx]))
    print ('')
    hoFin = 0
    hiIdx = hiStart
    for hfIdx in range(hf):
        if (0 <= (hiIdx+hfIdx-1) and (hiIdx+hfIdx-1) < hi):
            print ('    \"add x%s, x%s, x9\\n\"'%(11 + hfIdx, 11 + hfIdx - 1))
        else:
            print ('    \"mov x%s, %%[in]\\n\"'%(11 + hfIdx))
    print ('    \"mov x8, %[out]\\n\"')
    loopPrinted = 0
    
    while hoFin < ho:
        woPrfmIdx = 0
        if hiIdx < hiLoopStart or hiIdx >= hiLoopStart+hiLoopCnt:
            wiIdx = wiStart
            woFin = 0
            woQue = []
            for oN in range(outNum):
                woQue += [[oN, wf]]
            print('// Wout queue state: ', end='')
            print(woQue)
            while woFin < wo:
                print('// Hin index %s, Win index %s, HoFin index %s, WoFin index %s'%(hiIdx, wiIdx, hoFin, woFin))
                if (0 <= wiIdx and wiIdx < wi):
                    if (woFin != wo):
                        for hfIdx in range(hf):
                            if (0 <= (hiIdx+hfIdx*dilation) and (hiIdx+hfIdx*dilation) < hi):
                                print ('    \"ld1 {v%s.4s - v%s.4s}, [x%s], #32\\n\"'%(regIn[hfIdx*co], regIn[hfIdx*co + 1], 11 + hfIdx))
                    for hfIdx in range(hf):
                        if (0 <= (hiIdx+hfIdx*dilation) and (hiIdx+hfIdx*dilation) < hi):
                            for woQ in woQue:
                                for wfIdx in range(wf):
                                    if (woQ[0]*stride + wfIdx*dilation - padding) == wiIdx:
                                        if (woQ[0] < wo):
                                            for coIdx in range(co):
                                                ireg = regIn[hfIdx*co + coIdx]
                                                freg = regFil[(hfIdx*wf + wfIdx)*co + coIdx]     
                                                oreg = regOut[(woQ[0]%outNum)*co + coIdx]
                                                breg = regBias[coIdx]
                                                print ('    \"fmla v%s.4s, v%s.4s, v%s.4s\\n\"'%(oreg, freg, ireg))         
                                            if hfIdx == hf-1:
                                                woQ[1] = woQ[1] - 1
                                            if woQ[1] == 0:
                                                print('// Writing Wout index %s'%(woQ[0]))
                                                woFin = woFin + 1
                                                print ('    \"stp q%s, q%s, [x8], #%s\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regOut[(woQ[0]%outNum)*co + 1], 32))
                                                print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regBias[0]))
                                                print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 1], regBias[1]))
                                                woQ[0] = woQ[0] + outNum
                                                woQ[1] = wf
                                    if wfIdx == 0 and hfIdx == 1 and woQue.index(woQ) == 1:
                                        if (woPrfmIdx%2) == 0:
                                            if (0 <= (hiIdx+(hf-1)*dilation) and (hiIdx+(hf-1)*dilation) < hi):
                                                print ('    \"prfm pldl1keep, [x%s, #256]\\n\"'%(11 + hf-1))
                                        woPrfmIdx += 1
                        else:
                            for woQ in woQue:
                                for wfIdx in range(wf):
                                    if (woQ[0]*stride + wfIdx*dilation - padding) == wiIdx:
                                        if (woQ[0] < wo):
                                            for coIdx in range(co):
                                                ireg = regIn[hfIdx*co + coIdx]
                                                freg = regFil[(hfIdx*wf + wfIdx)*co + coIdx]     
                                                oreg = regOut[(woQ[0]%outNum)*co + coIdx]
                                            if hfIdx == hf-1:
                                                woQ[1] = woQ[1] - 1
                                            if woQ[1] == 0:
                                                print('// Writing Wout index %s'%(woQ[0]))
                                                woFin = woFin + 1
                                                print ('    \"stp q%s, q%s, [x8], #%s\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regOut[(woQ[0]%outNum)*co + 1], 32))
                                                print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regBias[0]))
                                                print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 1], regBias[1]))
                                                woQ[0] = woQ[0] + outNum
                                                woQ[1] = wf
                else:
                    for hfIdx in range(hf):
                        for woQ in woQue:
                            for wfIdx in range(wf):
                                if (woQ[0]*stride + wfIdx*dilation - padding) == wiIdx:
                                    if (woQ[0] < wo):
                                        for coIdx in range(co):
                                            ireg = regIn[hfIdx*co + coIdx]
                                            freg = regFil[(hfIdx*wf + wfIdx)*co + coIdx]     
                                            oreg = regOut[(woQ[0]%outNum)*co + coIdx]
                                        if hfIdx == hf-1:
                                            woQ[1] = woQ[1] - 1
                                        if woQ[1] == 0:
                                            print('// Writing Wout index %s'%(woQ[0]))
                                            woFin = woFin + 1
                                            print ('    \"stp q%s, q%s, [x8], #%s\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regOut[(woQ[0]%outNum)*co + 1], 32))
                                            print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regBias[0]))
                                            print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 1], regBias[1]))
                                            woQ[0] = woQ[0] + outNum
                                            woQ[1] = wf
                wiIdx = wiIdx + 1
                print('// Wout queue state: ', end='')
                print(woQue)
            if stride == 2:
                for hfIdx in range(hf):
                    print ('    \"add x%s, x%s, x9\\n\"'%(11 + hfIdx, 11 + hfIdx))
        else:
            if loopPrinted == 0:
                print('')
                print('// Loop Starts Here')
                print ('    \"mov x10, #%s\\n\"'%(hiLoopCnt))
                print ('    \"LOOP_START%=:\\n\"')
                print ('    \"subs x10, x10, #%s\\n\"'%(stride))
                print('')
                wiIdx = wiStart
                woFin = 0
                woQue = []
                for oN in range(outNum):
                    woQue += [[oN, wf]]
                print('// Wout queue state: ', end='')
                print(woQue)
                while woFin < wo:
                    print('// Hin index %s, Win index %s, HoFin index %s, WoFin index %s'%(hiIdx, wiIdx, hoFin, woFin))
                    if (0 <= wiIdx and wiIdx < wi):
                        if (woFin != wo):
                            for hfIdx in range(hf):
                                if (0 <= (hiIdx+hfIdx*dilation) and (hiIdx+hfIdx*dilation) < hi):
                                    print ('    \"ld1 {v%s.4s - v%s.4s}, [x%s], #32\\n\"'%(regIn[hfIdx*co], regIn[hfIdx*co + 1], 11 + hfIdx))
                        for hfIdx in range(hf):
                            if (0 <= (hiIdx+hfIdx*dilation) and (hiIdx+hfIdx*dilation) < hi):
                                for woQ in woQue:
                                    for wfIdx in range(wf):
                                        if (woQ[0]*stride + wfIdx*dilation - padding) == wiIdx:
                                            if (woQ[0] < wo):
                                                for coIdx in range(co):
                                                    ireg = regIn[hfIdx*co + coIdx]
                                                    freg = regFil[(hfIdx*wf + wfIdx)*co + coIdx]     
                                                    oreg = regOut[(woQ[0]%outNum)*co + coIdx]
                                                    breg = regBias[coIdx]
                                                    print ('    \"fmla v%s.4s, v%s.4s, v%s.4s\\n\"'%(oreg, freg, ireg))         
                                                if hfIdx == hf-1:
                                                    woQ[1] = woQ[1] - 1
                                                if woQ[1] == 0:
                                                    print('// Writing Wout index %s'%(woQ[0]))
                                                    woFin = woFin + 1
                                                    print ('    \"stp q%s, q%s, [x8], #%s\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regOut[(woQ[0]%outNum)*co + 1], 32))
                                                    print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regBias[0]))
                                                    print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 1], regBias[1]))
                                                    woQ[0] = woQ[0] + outNum
                                                    woQ[1] = wf
                                        if wfIdx == 0 and hfIdx == 1 and woQue.index(woQ) == 1:
                                            if (woPrfmIdx%2) == 0:
                                                if (0 <= (hiIdx+(hf-1)*dilation) and (hiIdx+(hf-1)*dilation) < hi):
                                                    print ('    \"prfm pldl1keep, [x%s, #256]\\n\"'%(11 + hf-1))
                                            woPrfmIdx += 1
                            else:
                                for woQ in woQue:
                                    for wfIdx in range(wf):
                                        if (woQ[0]*stride + wfIdx*dilation - padding) == wiIdx:
                                            if (woQ[0] < wo):
                                                for coIdx in range(co):
                                                    ireg = regIn[hfIdx*co + coIdx]
                                                    freg = regFil[(hfIdx*wf + wfIdx)*co + coIdx]     
                                                    oreg = regOut[(woQ[0]%outNum)*co + coIdx]
                                                if hfIdx == hf-1:
                                                    woQ[1] = woQ[1] - 1
                                                if woQ[1] == 0:
                                                    print('// Writing Wout index %s'%(woQ[0]))
                                                    woFin = woFin + 1
                                                    print ('    \"stp q%s, q%s, [x8], #%s\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regOut[(woQ[0]%outNum)*co + 1], 32))
                                                    print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regBias[0]))
                                                    print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 1], regBias[1]))
                                                    woQ[0] = woQ[0] + outNum
                                                    woQ[1] = wf
                    else:
                        for hfIdx in range(hf):
                            for woQ in woQue:
                                for wfIdx in range(wf):
                                    if (woQ[0]*stride + wfIdx*dilation - padding) == wiIdx:
                                        if (woQ[0] < wo):
                                            for coIdx in range(co):
                                                ireg = regIn[hfIdx*co + coIdx]
                                                freg = regFil[(hfIdx*wf + wfIdx)*co + coIdx]     
                                                oreg = regOut[(woQ[0]%outNum)*co + coIdx]
                                            if hfIdx == hf-1:
                                                woQ[1] = woQ[1] - 1
                                            if woQ[1] == 0:
                                                print('// Writing Wout index %s'%(woQ[0]))
                                                woFin = woFin + 1
                                                print ('    \"stp q%s, q%s, [x8], #%s\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regOut[(woQ[0]%outNum)*co + 1], 32))
                                                print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regBias[0]))
                                                print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 1], regBias[1]))
                                                woQ[0] = woQ[0] + outNum
                                                woQ[1] = wf
                    wiIdx = wiIdx + 1
                    print('// Wout queue state: ', end='')
                    print(woQue)
                if stride == 2:
                    for hfIdx in range(hf):
                        print ('    \"add x%s, x%s, x9\\n\"'%(11 + hfIdx, 11 + hfIdx))
                print('')
                print('// Loop Ends here')
                print ('    \"b.ne LOOP_START%=\\n\"')
                print('')
                loopPrinted = 1
            # else:
                # print('// Hin index %s, Win index %s, HoFin index %s, WoFin index %s'%(hiIdx, wiIdx, hoFin, woFin))
        hiIdx = hiIdx + stride
        hoFin = hoFin + 1

    print ('    :')
    print ('    :   ', end='')
    print (' [in] \"r\" (inputPtr), [fil] \"r\" (filterPtr), [out] \"r\" (outputPtr), [bias] \"r\" (biasPtr)')
    print ('    :   \"x8\", \"x9\", \"x10\", ', end = '')
    for hfIdx in range(hf):
        print ("\"x%s\", "%(11 + hfIdx), end = '')
    print('')
    print ('        \"v0\", \"v1\", \"v2\", \"v3\", \"v4\", \"v5\", \"v6\", \"v7\", \"v8\", \"v9\", \"v10\", \"v11\", \"v12\", \"v13\", \"v14\", \"v15\",')
    print ('        \"v16\", \"v17\", \"v18\", \"v19\", \"v20\", \"v21\", \"v22\", \"v23\", \"v24\", \"v25\", \"v26\", \"v27\", \"v28\", \"v29\", \"v30\", \"v31\", \"cc\", \"memory\"')
    print ('    );')
    print ('    }')
    print ('}')

def generateKernelDepth(wf, wo, co, stride, dilation, padding, skipStart, skipEnd):
    print ("void kernel_depth_%s_%s_%s_%s_%s_%s_%s_%s(float* inputPtr, float* filterPtr, float* outputPtr, float* biasPtr, const int k)"%(wf, wo, co*4, stride, dilation, padding, skipStart, skipEnd))
    print('{')
    outNum = 3
    ho = wo
    hf = wf
    regIn = list(range(hf*co))
    regFil = list(range(hf*co, hf*co + hf*wf*co))
    regOut = list(range(hf*co + hf*wf*co, hf*co + hf*wf*co + outNum*co))
    regBias = list(range(hf*co + hf*wf*co + outNum*co, hf*co + hf*wf*co + outNum*co + co))
    print ('// In  - ',end='')
    print (regIn)
    print ('// Out - ',end='')
    print (regOut)
    print ('// Fil - ',end='')
    print (regFil)
    print ('// Bias - ',end='')
    print (regBias)
    
    # Finding the input index # on each output position.
    inputIdxPerPos=[]
    for i in range(wo):
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
    wiStart = -padding
    wi = wo*stride
    hi = ho*stride

    print ('// Register mapping diagram')
    print ('// ',end='')
    for wfIdx in range(wf*wf):
        print('   ',end='')
    for wiIdx in regIn:
        print('%2s '%(wiIdx),end='')
    print ('')
    print ('//')
    print ('//',end='')
    for coIdx in range(co):
        for wfIdx in range(wf*wf):
            print('%2s '%(regFil[wfIdx*co + coIdx]),end='')
        print (' ',end='')
        for wtIdx in range(wo):
            print('%2s '%(regOut[(wtIdx%outNum)*co + coIdx]),end='')
            print ('   '*(stride-1),end='')
        print ('')
        print ('//',end='')
    print ('')

    print ('')
    print ('    #ifdef __DEBUG_PTMM_OFF')
    print ('    printf (\"Input:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Ci %d:\\t\", i);')
    print ('        for (int j = 0; j < %s; j++)'%(wi))
    print ('        {')
    print ('            printf(\"%%6.3f\\t\", *(inputPtr + j*%s + i));'%(co*4))
    print ('        }')
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"Filter:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Ci %d:\\t\", i);')
    print ('        for (int fi = 0; fi < %s; fi++)'%(hf*wf))
    print ('        {')
    print ('            printf(\"%%6.3f\\t\", *(filterPtr + fi*%s + i));'%(co*4))
    print ('        }')
    print ('        printf(\"\\n\");')
    print ('    }')
    print ('    printf (\"Output:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(co*4))
    print ('    {')
    print ('        printf(\"Ci %d:\\t\", i);')
    print ('        for (int j = 0; j < %s; j++)'%(wo))
    print ('        {')
    print ('            printf(\"%%6.3f\\t\", *(outputPtr + j*%s + i));'%(co*4))
    print ('        }')
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"\\n\");')
    print ('    #endif')
    print ('')

    print ('    {')
    print ('    __asm __volatile (')
    print('// Loading Bias.')
    print ('    \"ld1 {v%s.4s - v%s.4s}, [%%[bias]]\\n\"'%(regBias[0], regBias[1]))
    print ('    \"mov x9, #%s\\n\"'%(wi*32))
    for hfIdx in range(hf):
        if hfIdx == 0:
            print ('    \"mov x%s, %%[in]\\n\"'%(11 + hfIdx))
        else:
            print ('    \"add x%s, x%s, x9\\n\"'%(11 + hfIdx, 11 + hfIdx - 1))
    print('// Loading Fiters.')
    for hfIdx in range(hf):
        for wfIdx in range(wf):
            print ('    \"ldp q%s, q%s, [%%[fil], #%s]\\n\"'%(regFil[(hfIdx*wf + wfIdx)*co + 0], regFil[(hfIdx*wf + wfIdx)*co + 1], 32*(hfIdx*wf + wfIdx)))
    print('// Loading Inputs.')
    for hfIdx in range(hf):
        if (skipStart <= hfIdx and hfIdx < hf - skipEnd):
            print ('    \"ld1 {v%s.4s - v%s.4s}, [x%s], #32\\n\"'%(regIn[hfIdx*co], regIn[hfIdx*co + 1], 11 + hfIdx))
    print('// Biasing Outputs.')
    for coIdx in range(co):
        for oN in range(outNum):
            print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[oN*co + coIdx], regBias[coIdx]))
    print ('')
    print ('    \"mov x8, %[out]\\n\"')

    woPrfmIdx = 0
    wiIdx = wiStart
    woFin = 0
    woQue = []
    for oN in range(outNum):
        woQue += [[oN, wf]]

    print('')
    print('// Loop Starts Here')
    print ('    \"mov w10, %w[k]\\n\"')
    print ('    \"LOOP_START%=:\\n\"')
    print ('    \"subs x10, x10, #1\\n\"')
    print('')
    
    print('// Wout queue state: ', end='')
    strideJumpFlag = 0
    print(woQue)
    while woFin < wo:
        print('// Win index %s, WoFin index %s'%(wiIdx, woFin))
        if (0 <= wiIdx and wiIdx < wi): 
            if stride == 2 and woFin == wo-1 and wiIdx == wi - 1:
                for hfIdx in range(hf):
                    print ('    \"add x%s, x%s, x9\\n\"'%(11 + hfIdx, 11 + hfIdx))
            for hfIdx in range(hf):
                if (skipStart <= hfIdx and hfIdx < hf - skipEnd):
                    for woQ in woQue:
                        for wfIdx in range(wf):
                            if (woQ[0]*stride + wfIdx*dilation - padding) == wiIdx:
                                if (woQ[0] < wo):
                                    for coIdx in range(co):
                                        ireg = regIn[hfIdx*co + coIdx]
                                        freg = regFil[(hfIdx*wf + wfIdx)*co + coIdx]     
                                        oreg = regOut[(woQ[0]%outNum)*co + coIdx]
                                        print ('    \"fmla v%s.4s, v%s.4s, v%s.4s\\n\"'%(oreg, freg, ireg))         
                                    if hfIdx == hf-1:
                                        woQ[1] = woQ[1] - 1
                                    if woQ[1] == 0:
                                        print('// Writing Wout index %s'%(woQ[0]))
                                        woFin = woFin + 1
                                        print ('    \"stp q%s, q%s, [x8], #%s\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regOut[(woQ[0]%outNum)*co + 1], 32))
                                        print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regBias[0]))
                                        print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 1], regBias[1]))
                                        woQ[0] = woQ[0] + outNum
                                        woQ[1] = wf
                            if wfIdx == 0 and hfIdx == 1 and woQue.index(woQ) == 1 and skipEnd != 1:
                                if (woPrfmIdx%2) == 0:
                                    print ('    \"prfm pldl1keep, [x%s, #256]\\n\"'%(11 + hf-1))    
                                woPrfmIdx += 1
                        
                    print ('    \"ld1 {v%s.4s - v%s.4s}, [x%s], #32\\n\"'%(regIn[hfIdx*co], regIn[hfIdx*co + 1], 11 + hfIdx))
                else:
                    for woQ in woQue:
                        for wfIdx in range(wf):
                            if (woQ[0]*stride + wfIdx*dilation - padding) == wiIdx:
                                if (woQ[0] < wo):
                                    for coIdx in range(co):
                                        ireg = regIn[hfIdx*co + coIdx]
                                        freg = regFil[(hfIdx*wf + wfIdx)*co + coIdx]     
                                        oreg = regOut[(woQ[0]%outNum)*co + coIdx]
                                    if hfIdx == hf-1:
                                        woQ[1] = woQ[1] - 1
                                    if woQ[1] == 0:
                                        print('// Writing Wout index %s'%(woQ[0]))
                                        woFin = woFin + 1
                                        print ('    \"stp q%s, q%s, [x8], #%s\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regOut[(woQ[0]%outNum)*co + 1], 32))
                                        print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regBias[0]))
                                        print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 1], regBias[1]))
                                        woQ[0] = woQ[0] + outNum
                                        woQ[1] = wf
        else:
            for hfIdx in range(hf):
                for woQ in woQue:
                    for wfIdx in range(wf):
                        if (woQ[0]*stride + wfIdx*dilation - padding) == wiIdx:
                            if (woQ[0] < wo):
                                for coIdx in range(co):
                                    ireg = regIn[hfIdx*co + coIdx]
                                    freg = regFil[(hfIdx*wf + wfIdx)*co + coIdx]     
                                    oreg = regOut[(woQ[0]%outNum)*co + coIdx]
                                if hfIdx == hf-1:
                                    woQ[1] = woQ[1] - 1
                                if woQ[1] == 0:
                                    print('// Writing Wout index %s'%(woQ[0]))
                                    woFin = woFin + 1
                                    print ('    \"stp q%s, q%s, [x8], #%s\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regOut[(woQ[0]%outNum)*co + 1], 32))
                                    print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 0], regBias[0]))
                                    print ('    \"mov v%s.16b, v%s.16b\\n\"'%(regOut[(woQ[0]%outNum)*co + 1], regBias[1]))
                                    woQ[0] = woQ[0] + outNum
                                    woQ[1] = wf
        wiIdx = wiIdx + 1
        print('// Wout queue state: ', end='')
        print(woQue)

    print('// Loop Ends here')
    print ('    \"b.ne LOOP_START%=\\n\"')

    print ('    :')
    print ('    :   ', end='')
    print (' [in] \"r\" (inputPtr), [fil] \"r\" (filterPtr), [out] \"r\" (outputPtr), [bias] \"r\" (biasPtr), [k] \"r\" (k)')
    print ('    :   \"x8\", \"x9\", \"x10\", ', end = '')
    for hfIdx in range(hf):
        print ("\"x%s\", "%(11 + hfIdx), end = '')
    print('')
    print ('        \"v0\", \"v1\", \"v2\", \"v3\", \"v4\", \"v5\", \"v6\", \"v7\", \"v8\", \"v9\", \"v10\", \"v11\", \"v12\", \"v13\", \"v14\", \"v15\",')
    print ('        \"v16\", \"v17\", \"v18\", \"v19\", \"v20\", \"v21\", \"v22\", \"v23\", \"v24\", \"v25\", \"v26\", \"v27\", \"v28\", \"v29\", \"v30\", \"v31\", \"cc\", \"memory\"')
    print ('    );')
    print ('    }')
    print ('}')

for wti in [28,56,112]:
    generateKernelDepth(3, wti, 2, 1, 1, 1, 1, 0)
    generateKernelDepth(3, wti, 2, 1, 1, 1, 0, 1)
    generateKernelDepth(3, wti, 2, 1, 1, 1, 0, 0)

for wti in [28, 56]:
    generateKernelDepth(3, wti, 2, 2, 1, 1, 1, 0)
    generateKernelDepth(3, wti, 2, 2, 1, 1, 0, 1)
    generateKernelDepth(3, wti, 2, 2, 1, 1, 0, 0)


