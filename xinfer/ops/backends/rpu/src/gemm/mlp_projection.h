// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#include "__clang_rpu_builtin_vars.h"
#if PROJ_N%16 || PROJ_K%16 || PROJ_TM%16 || PROJ_TN%16 || PROJ_TK%16 || PROJ_K%PROJ_TK
#error "Projection dimensions must be complete vector tiles"
#endif
#if (PROJ_TM/16)*(PROJ_TK/16)+(PROJ_TN/16)*(PROJ_K/16)+(PROJ_TM/16)*(PROJ_TN/16)+(PROJ_TN/16)>390
#error "Projection VLM tile exceeds 390 FP16 vectors per thread"
#endif
#if PROJ_TK != PROJ_K && (PROJ_TM != 16 || PROJ_TN != 16)
#error "Streaming K preserves one FP32 VMAT accumulator; use M16 N16"
#endif
// A/C/bias: compact FP16 LocalSPM. W: [N/16][K/16][8][16 N][16 K] DDR.
// Bias initializes the FP32 VMAT accumulator. No FP16 partial is read back.
// A complete N tile of W stays in VLM across every M tile. FC1 streams A
// chunks while keeping the same FP32 accumulator across all K iterations.
// Pointer ABI: A[0:1], W/256[2:3], C[4:5], bias[6:7], M[8].
// Optional fused epilogue (fusion/): PROJ_EPILOGUE(r, scratch, count)
// runs on the count FP16 output tiles r[] of one M tile before their store;
// scratch is the A buffer, free there. PROJ_ENTRY_PROLOGUE runs at entry.
#ifndef PROJ_EPILOGUE
#define PROJ_EPILOGUE(r, scratch, count)
#endif
#ifndef PROJ_ENTRY_PROLOGUE
#define PROJ_ENTRY_PROLOGUE
#endif
__rprog__ void PROJ_ENTRY(half *av, DDR_large wv, half *cv, half *bias,
                         unsigned short rows) {
  const short AK=PROJ_K/16, CK=PROJ_TK/16, AN=PROJ_TN/16, AM=PROJ_TM/16;
#if PROJ_N == 128 && PROJ_TN == 48
  // With N48, implicit allocation puts both A and R beyond the 8-bit VMAT
  // constant-base field. Keep all VMAT bases low using one reviewed VLM arena.
  __local__ f16v16 arena[AM*AN+AN+AM*CK+AN*AK];
  auto *r=arena, *biasv=r+AM*AN, *a=biasv+AN, *b=a+AM*CK;
#else
  __local__ f16v16 a[AM*CK], b[AN*AK], r[AM*AN], biasv[AN];
#endif
  PROJ_ENTRY_PROLOGUE
  const short n0=(short)blockIdx.x*AN;
  const short nc=PROJ_N/16-n0 < AN ? PROJ_N/16-n0 : AN;
  const short mb=(rows+15)/16;
  short lp[3];
  DDR_large wg=wv+(int)n0*AK*16+(int)coreId_u*2;
  LoopStepInfo bd[2]={{&lp[1],AK},{&lp[0],1}};
  FragVData vb(b,bd);
  LoopStepInfo bs[2]={{&lp[1],(int)AK*4096,rpu::__AGran::a_g1B},
                       {&lp[0],(int)4096,rpu::__AGran::a_g1B}};
  FragSData sw(wg,bs);
#pragma rpu_hwloop_ul
  for(lp[1]=0;lp[1]<nc;++lp[1]) {
#pragma rpu_hwloop_ul
    for(lp[0]=0;lp[0]<AK;++lp[0])
      rpu::vldExecute_async(vb,sw,(short)(threadId*32));
  }
  LoopStepInfo biasd[1]={{&lp[0],1}};
  FragVData bvv(biasv,biasd);
  LoopStepInfo biass[1]={{&lp[0],(int)16}};
  FragSData sbias(bias+(int)n0*16,biass);
#pragma rpu_hwloop_ul
  for(lp[0]=0;lp[0]<nc;++lp[0])
    rpu::vldExecute_async(bvv,sbias,(short)0);
  rpu::fenceVld(0);
#pragma rpu_hwloop_ul
  for(short m0=0;m0<mb;m0+=AM) {
    const short mc=mb-m0<AM?mb-m0:AM;
    const short live=(rows+15-threadId)/16-m0;
    const short valid=live>0?(live<AM?live:AM):0;
#if PROJ_TK == PROJ_K
    LoopStepInfo ad[2]={{&lp[1],AK},{&lp[0],1}};
    FragVData va(a,ad);
    LoopStepInfo as[2]={{&lp[1],(int)16*PROJ_K},{&lp[0],(int)16}};
    FragSData sa(av+((int)m0*16+threadId)*PROJ_K,as);
#pragma rpu_hwloop_ul
    for(lp[1]=0;lp[1]<valid;++lp[1]) {
#pragma rpu_hwloop_ul
      for(lp[0]=0;lp[0]<AK;++lp[0])
        rpu::vldExecute_async(va,sa,(short)0);
    }
    rpu::fenceVld(0);
    rpu::fenceVst(0);
    LoopStepInfo cs[3]={{&lp[2],AN},{&lp[1],1},{&lp[0],0}};
    FragVData vr(r,cs);
    LoopStepInfo ac[3]={{&lp[2],AK},{&lp[1],0},{&lp[0],1}};
    FragVData aa(a,ac);
    LoopStepInfo bc[3]={{&lp[2],0},{&lp[1],AK},{&lp[0],1}};
    FragVData bb(b,bc);
    LoopStepInfo bis[1]={{&lp[1],1}};
    FragVData init(biasv,bis);
    LoopOutInfo oi[1]={{&lp[0]}};
    LoopOutLPAttr oa(oi);
#pragma rpu_hwloop_ul
    for(lp[2]=0;lp[2]<mc;++lp[2]) {
#pragma rpu_hwloop_ul
      for(lp[1]=0;lp[1]<nc;++lp[1]) {
        rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp16,init);
#pragma rpu_hwloop_ul
        for(lp[0]=0;lp[0]<AK;++lp[0])
          rpu::vmatExecute_async(vr,aa,bb,&oa,rpu::__ClrMode::clroff);
      }
    }
#else
    rpu::fenceVst(0);

#pragma rpu_hwloop_ul
    for(short k0=0;k0<AK;k0+=CK) {
      LoopStepInfo ad[1]={{&lp[0],1}};
      FragVData va(a,ad);
      LoopStepInfo as[1]={{&lp[0],(int)16}};
      FragSData sa(av+((int)m0*16+threadId)*PROJ_K+(int)k0*16,as);
      if(valid) {
#pragma rpu_hwloop_ul
        for(lp[0]=0;lp[0]<CK;++lp[0])
          rpu::vldExecute_async(va,sa,(short)0);
      }
      rpu::fenceVld(0);
      // Initialize only after the first A chunk is resident, next to VMAT.
      if(k0==0) rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp16,biasv);
      LoopStepInfo cs[1]={{&lp[0],0}};
      FragVData vr(r,cs);
      LoopStepInfo ac[1]={{&lp[0],1}};
      FragVData aa(a,ac),bb(b+k0,ac);
      // Intermediate outputs are never consumed: clroff preserves the FP32
      // accumulator across K chunks; the final K output overwrites r.
      LoopOutInfo oi[1]={{&lp[0]}};
      LoopOutLPAttr oa(oi);
#pragma rpu_hwloop_ul
      for(lp[0]=0;lp[0]<CK;++lp[0])
        rpu::vmatExecute_async(vr,aa,bb,&oa,rpu::__ClrMode::clroff);
      rpu::fenceVmat(0);
    }
#endif
    rpu::fenceVmat(0);
    PROJ_EPILOGUE(r, a, mc*AN);
    LoopStepInfo rs[2]={{&lp[1],AN},{&lp[0],1}};
    FragVData result(r,rs);
    LoopStepInfo ds[2]={{&lp[1],(int)16*PROJ_N},{&lp[0],(int)16}};
    FragSData dst(cv+((int)m0*16+threadId)*PROJ_N+(int)n0*16,ds);
#pragma rpu_hwloop_ul
    for(lp[1]=0;lp[1]<valid;++lp[1]) {
#pragma rpu_hwloop_ul
      for(lp[0]=0;lp[0]<nc;++lp[0])
        rpu::vstExecute_async(result,dst,(short)0);
    }
  }
  rpu::fenceVst(0);
}
