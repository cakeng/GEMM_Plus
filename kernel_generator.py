#! /usr/bin/env python3 
import sys
import os

def generateKernel(Wf, Wt, Co, stride, dilation, skipStart, skipEnd):
    print ("void kernel_%s_%s_%s_%s_%s_%s_%s(float* in, float* filter, float* out, const int k, const int cin, const int cout)"%(Wf, Wt, Co, stride, dilation, skipStart, skipEnd))
    print('{')
    indexes=[]
    for i in range(Wt):
        WinStart = i*stride
        index=[]
        for j in range(Wf):
            index.append(WinStart + j*dilation)
        indexes.append(index)   
    print ("// Input indexes per position")
    print ('// ',end='')
    print(indexes)
    WinRegs = []
    dups = 0
    fnum = Wf
    if (5 < Wt):
        fnum = 1
    for index in indexes:
        for pos in index:
            if WinRegs.count(pos) == 0:
                WinRegs.append(pos)
            else:
                dups += 1
    WinRegs.sort()
    print ("// Input registers required")
    print ('// ',end='')
    print(WinRegs)
    dupPos = [0]*len(WinRegs)
    for index in indexes:
        for pos in index:
            dupPos[WinRegs.index(pos)] += 1
    totalRegs = Wt*Co + len(WinRegs) + fnum*Co
    print ("// Duplicate indexes: " + str(dups))
    print ('// ',end='')
    print(dupPos)
    print ("//Number of Input Regs: %d, Filter Regs: %d Output Regs: %d"%(len(WinRegs), fnum*Co, Wt*Co))
    print ("//Total number of registers required: " + str(totalRegs))
    if totalRegs > 32:
        print ('Unsupported number of registers!')
        sys.exit(0)
    
    regFilter = list(range(fnum*Co))
    regOut = list(range(fnum*Co, fnum*Co + Wt*Co))
    regIn = list(range(fnum*Co + Wt*Co, fnum*Co + Wt*Co + len(WinRegs)))
    print ('// Fil - ',end='')
    print (regFilter)
    print ('// Out - ',end='')
    print (regOut)
    print ('// In  - ',end='')
    print (regIn)
    print ('// Register mapping diagram')
    print ('// ',end='')
    for wfIdx in range(Wf):
        print('   ',end='')
    for wiIdx in range(len(WinRegs)):
        print('%2s '%(wiIdx),end='')
    print ('')
    print ('// ',end='')
    for wfIdx in range(Wf):
        print('   ',end='')
    for wiIdx in range(len(WinRegs)):
        print('%2s '%(regIn[wiIdx]),end='')
    print ('')
    print ('//')
    print ('//',end='')
    for coIdx in range(Co):
        for wfIdx in range(Wf):
            print('%2s '%(0),end='')
        print (' ',end='')
        for wtIdx in range(Wt):
            print('%2s '%(regOut[wtIdx*Co + coIdx]),end='')
            print ('   '*(stride-1),end='')
        print ('')
        print ('//',end='')

    print ('')
    print ('    #ifdef __DEBUG_PTMM_OFF')
    print ('    printf (\"Input:\\n\");')
    print ('    for (int i = 0; i < k; i++)')
    print ('    {')
    print ('        printf(\"Row %d:\\t\", i);')
    for wiIdx in WinRegs:
        print ('        printf(\"%%6.3f\\t\", *(in + cin*%s + i));'%(wiIdx))
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"Filter:\\n\");')
    print ('    for (int wf = 0; wf < %s; wf++)'%(Wf))
    print ('    {')
    print ('        printf(\"Wfil %d:\\n\", wf);')
    print ('        for (int i = 0; i < %s; i++)'%(Co*4))
    print ('        {')
    print ('            printf(\"Row %d:\\t\", i);')
    print ('            for (int j = 0; j < k; j++)')
    print ('            {')
    print ('                printf(\"%%6.3f\\t\", *(filter + j*%s*__COUTB1 + wf*__COUTB1 + i));'%(Wf))
    print ('            }')
    print ('            printf(\"\\n\");')
    print ('        }')
    print ('    }')
    print ('    printf (\"Output:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(Co*4))
    print ('    {')
    print ('        printf(\"Row %d:\\t\", i);')
    for wiIdx in WinRegs:
        print ('        printf(\"%%6.3f\\t\", *(out + cout*%s + i));'%(wiIdx))
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"\\n\");')
    print ('    #endif')

    print('// Load Input ptr')
    for wiIdx in WinRegs:
        if skipStart <= wiIdx and wiIdx <= WinRegs[len(WinRegs)-1] - skipEnd:
            print ("    float* inP%s = in + %s*cin;"%(wiIdx, wiIdx))
    print ('    {')
    print ('    __asm __volatile (')
    loadRegOffset = len(WinRegs) + 7
    fReg = loadRegOffset + 1
    oReg = loadRegOffset + 2

    print('// Prefetch input and filter')
    print ('    \"prfm pldl1keep, [%[fil], #0]\\n\"')
    print ('    \"prfm pldl1keep, [%[fil], #64]\\n\"')
    print ('    \"prfm pldl1keep, [%[fil], #128]\\n\"')
    print ('    \"prfm pldl1keep, [%[fil], #192]\\n\"')
    for wiIdx in WinRegs:
        if skipStart <= wiIdx and wiIdx <= WinRegs[len(WinRegs)-1] - skipEnd:
            print ('    \"prfm pldl1keep, [%%[inP%s], #0]\\n\"'%(wiIdx))
    print('// Load Output')
    print ('    \"mov x%s, %%[out]\\n\"'%(oReg))
    for i in range(Wt):
        if Co == 2:
            print ('    \"ld1 {v%s.4s - v%s.4s}, [x%s], %%[cout]\\n\"'%(regOut[i*Co], regOut[i*Co + 1], oReg))
        elif Co == 1:
            print ('    \"ld1 {v%s.4s}, [x%s], %%[cout]\\n\"'%(regOut[i*Co], oReg))
    print(' // Load filters')
    print ('    \"mov x%s, %%[fil]\\n\"'%(fReg))
    for wfIdx in range(fnum):
        if Co == 2:
            print ('    \"ld1 {v%s.4s - v%s.4s}, [x%s], #32\\n\"'%(regFilter[wfIdx*Co + 0], regFilter[wfIdx*Co + Co-1], fReg))
        elif Co == 1:
            print ('    \"ldr q%s, [x%s], #16\\n\"'%(regFilter[wfIdx*Co + 0], fReg))
    print('// Load Input')
    for wiIdx in WinRegs:
        if skipStart <= wiIdx and wiIdx <= WinRegs[len(WinRegs)-1] - skipEnd:
            print ('    \"ldr q%s, [%%[inP%s]], #16\\n\"'%(regIn[WinRegs.index(wiIdx)], wiIdx))
    for wiIdx in WinRegs:
        if skipStart <= wiIdx and wiIdx <= WinRegs[len(WinRegs)-1] - skipEnd:
            print ('    \"prfm pldl1keep, [%%[inP%s], #64]\\n\"'%(wiIdx))
    print ('')
    print ('    \"mov x%s, %%[k]\\n\"'%(oReg))
    
    print ('    \"LOOP_START%=:\\n\"')
    print ('    \"subs x%s, x%s, #16\\n\"'%(oReg, oReg))
    iInFetched = dupPos.copy()
    fidx = 0
    for i in range(4):
        kInLoaded = dupPos.copy()
        for k in range(4):
            print('// K index %s'%(i*4 + k))
            for wfIdx in range(Wf):
                print('// Filter width idx %s'%(wfIdx))
                for wtIdx in range(Wt):
                    for coIdx in range(Co):
                        oreg = regOut[wtIdx*Co + coIdx]
                        freg = (fidx%fnum)*Co + coIdx
                        ireg = regIn[WinRegs.index(indexes[wtIdx][wfIdx])]
                        if skipStart <= WinRegs.index(indexes[wtIdx][wfIdx]) and WinRegs.index(indexes[wtIdx][wfIdx]) <= WinRegs[len(WinRegs)-1] - skipEnd:
                            print ('    \"fmla v%s.4s, v%s.4s, v%s.s[%s]\\n\"'%(oreg, freg, ireg, k))
                        if coIdx == Co-1 and wtIdx == (int)(Wt/2) and fidx%4 == 0:
                            print ('    \"prfm pldl1keep, [x%s, #192]\\n\"'%(fReg))
                    if i == 0 and k==1:
                        iInFetched[WinRegs.index(indexes[wtIdx][wfIdx])] -= 1
                        if iInFetched[WinRegs.index(indexes[wtIdx][wfIdx])] == 0:
                            if skipStart <= WinRegs.index(indexes[wtIdx][wfIdx]) and WinRegs.index(indexes[wtIdx][wfIdx]) <= WinRegs[len(WinRegs)-1] - skipEnd:
                                print ('    \"prfm pldl1keep, [%%[inP%s], #192]\\n\"'%(indexes[wtIdx][wfIdx]))
                    if k == 3:
                        ireg = regIn[WinRegs.index(indexes[wtIdx][wfIdx])]
                        kInLoaded[WinRegs.index(indexes[wtIdx][wfIdx])] -= 1
                        if kInLoaded[WinRegs.index(indexes[wtIdx][wfIdx])] == 0:
                            if skipStart <= WinRegs.index(indexes[wtIdx][wfIdx]) and WinRegs.index(indexes[wtIdx][wfIdx]) <= WinRegs[len(WinRegs)-1] - skipEnd:
                                print ('    \"ldr q%s, [%%[inP%s]], #16\\n\"'%(ireg, indexes[wtIdx][wfIdx]))
                if Co == 1:
                    print ('    \"ldr q%s, [x%s], #16\\n\"'%((fidx%fnum)*Co, fReg))
                elif Co == 2:
                    print ('    \"ld1 {v%s.4s - v%s.4s}, [x%s], #32\\n\"'%((fidx%fnum)*Co, (fidx%fnum)*Co + 1, fReg))
                fidx += 1
    print ('    \"b.ne LOOP_START%=\\n\"')
    print ('    \"mov x%s, %%[out]\\n\"'%(oReg))
    for i in range(Wt):
        if Co == 2:
            print ('    \"st1 {v%s.4s,v%s.4s}, [x%s], %%[cout]\\n\"'%(regOut[i*Co], regOut[i*Co + 1], oReg))
        elif Co == 1:
            print ('    \"st1 {v%s.4s}, [x%s], %%[cout]\\n\"'%(regOut[i*Co + 0], oReg))
    print ('    :')
    print ('    :   ', end='')
    for wiIdx in WinRegs:
        if skipStart <= wiIdx and wiIdx <= WinRegs[len(WinRegs)-1] - skipEnd:
            print ('[inP%s] \"r\" (inP%s), '%(wiIdx, wiIdx), end='')
    print ('')
    print ('        [fil] \"p\" (filter), [out] \"p\" (out), [k] \"r\" (k), [cin] \"r\" (cin*sizeof(float)), [cout] \"r\" (cout*sizeof(float))')
    print ('    :   \"x%s\", \"x%s\", '%(fReg, oReg))
    print ('        \"v0\", \"v1\", \"v2\", \"v3\", \"v4\", \"v5\", \"v6\", \"v7\", \"v8\", \"v9\", \"v10\", \"v11\", \"v12\", \"v13\", \"v14\", \"v15\",')
    print ('        \"v16\", \"v17\", \"v18\", \"v19\", \"v20\", \"v21\", \"v22\", \"v23\", \"v24\", \"v25\", \"v26\", \"v27\", \"v28\", \"v29\", \"v30\", \"v31\", \"cc\", \"memory\"')
    print ('    );')
    print ('    }')
    print ('    #ifdef __DEBUG_PTMM_OFF')
    print ('    printf (\"Output After Kernel:\\n\");')
    print ('    for (int i = 0; i < %s; i++)'%(Co*4))
    print ('    {')
    print ('        printf(\"Row %d:\\t\", i);')
    for wiIdx in WinRegs:
        print ('        printf(\"%%6.3f\\t\", *(out + cout*%s + i));'%(wiIdx))
    print ('        printf (\"\\n\");')
    print ('    }')
    print ('    printf (\"\\n\");')
    print ('    #endif')
    print('}')

#def generateKernel(Wf, Wt, Co, stride, dilation, skipStart, skipEnd):
generateKernel (3, 7, 2, 2, 1, 0, 0)
