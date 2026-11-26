// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "__clang_rpu_builtin_vars.h"
using namespace rpu;
#define GD_INLINE __attribute__((always_inline)) inline

GD_INLINE void gd_bmm_rr(half *aa, half *bb, half *cc, __local__ f16v16 *pool) {
  auto *a=pool+0;
  auto *b=pool+32;
  auto *r=pool+64;

  short lp[3];
  vldSetTsize32B(8, 4);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(8, 2048);
  vldRepeat_async(a, aa, (int)threadId * 128);
  vldRepeat_async(b, bb, (int)threadId * 128);
  fenceVld(0);
  LoopStepInfo cs[3] = {{&lp[2], 4}, {&lp[1], 1}, {&lp[0], 0}};
  FragVData vc(r, cs);
  LoopStepInfo as[3] = {{&lp[2], 8}, {&lp[1], 0}, {&lp[0], 1}};
  FragVData fa(a, as);
  LoopStepInfo bs[3] = {{&lp[2], 0}, {&lp[1], 8}, {&lp[0], 1}};
  FragVData fb(b, bs);
  LoopOutInfo oi[1] = {{&lp[0]}};
  // Each output tile starts a fresh K reduction.
  LoopClrInfo ci[1] = {{&lp[0]}};
  LoopOutLPAttr out{oi};
  LoopClrLPAttr clr{ci};
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < 4; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < 4; ++lp[1]) {
#pragma rpu_hwloop_ul
      for (lp[0] = 0; lp[0] < 8; ++lp[0])
        vmatExecute_async(vc, fa, fb, &out, &clr);
    }
  }
  fenceVmat(0);
  vstSetTsize32B(4, 4);
  vstSetTstep32B_0(1, 16);
  vstSetTstep32B_1(4, 1024);
  vstRepeat(r, cc, (int)threadId * 64);
  fenceVst(0);
}

GD_INLINE void gd_bmm_rc64(half *aa, half *bb, half *cc, __local__ f16v16 *pool) {
  auto *a=pool+0;
  auto *b=pool+16;
  auto *r=pool+48;
  auto *tmp=pool+80;

  short lp[3];
  vldSetTsize32B(4, 4);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(4, 1024);
  vldRepeat_async(a, aa, (int)threadId * 64);
  vldSetTsize32B(8, 4);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(8, 2048);
  vldRepeat_async(tmp, bb, (int)threadId * 128);
  fenceVld(0);
  vsfuSetTsize32B(4, 8);
  vsfuSetTstep32B_0(1, 8, 0);
  vsfuSetTstep32B_1(4, 1, 0);
  vsfuRepeat(b, tmp, (short)0, __ShuffleMode::Trs);
  LoopStepInfo cs[3] = {{&lp[2], 8}, {&lp[1], 1}, {&lp[0], 0}};
  FragVData vc(r, cs);
  LoopStepInfo as[3] = {{&lp[2], 4}, {&lp[1], 0}, {&lp[0], 1}};
  FragVData fa(a, as);
  LoopStepInfo bs[3] = {{&lp[2], 0}, {&lp[1], 4}, {&lp[0], 1}};
  FragVData fb(b, bs);
  LoopOutInfo oi[1] = {{&lp[0]}};
  // Each output tile starts a fresh K reduction.
  LoopClrInfo ci[1] = {{&lp[0]}};
  LoopOutLPAttr out{oi};
  LoopClrLPAttr clr{ci};
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < 4; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < 8; ++lp[1]) {
#pragma rpu_hwloop_ul
      for (lp[0] = 0; lp[0] < 4; ++lp[0])
        vmatExecute_async(vc, fa, fb, &out, &clr);
    }
  }
  fenceVmat(0);
  vstSetTsize32B(8, 4);
  vstSetTstep32B_0(1, 16);
  vstSetTstep32B_1(8, 2048);
  vstRepeat(r, cc, (int)threadId * 128);
  fenceVst(0);
}

GD_INLINE void gd_bmm_rc128(half *aa, half *bb, half *cc, __local__ f16v16 *pool) {
  auto *a=pool+0;
  auto *b=pool+32;
  auto *r=pool+96;
  auto *tmp=pool+128;

  short lp[3];
  vldSetTsize32B(8, 4);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(8, 2048);
  vldRepeat_async(a, aa, (int)threadId * 128);
  vldSetTsize32B(8, 8);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(8, 2048);
  vldRepeat_async(tmp, bb, (int)threadId * 128);
  fenceVld(0);
  vsfuSetTsize32B(8, 8);
  vsfuSetTstep32B_0(1, 8, 0);
  vsfuSetTstep32B_1(8, 1, 0);
  vsfuRepeat(b, tmp, (short)0, __ShuffleMode::Trs);
  LoopStepInfo cs[3] = {{&lp[2], 8}, {&lp[1], 1}, {&lp[0], 0}};
  FragVData vc(r, cs);
  LoopStepInfo as[3] = {{&lp[2], 8}, {&lp[1], 0}, {&lp[0], 1}};
  FragVData fa(a, as);
  LoopStepInfo bs[3] = {{&lp[2], 0}, {&lp[1], 8}, {&lp[0], 1}};
  FragVData fb(b, bs);
  LoopOutInfo oi[1] = {{&lp[0]}};
  // Each output tile starts a fresh K reduction.
  LoopClrInfo ci[1] = {{&lp[0]}};
  LoopOutLPAttr out{oi};
  LoopClrLPAttr clr{ci};
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < 4; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < 8; ++lp[1]) {
#pragma rpu_hwloop_ul
      for (lp[0] = 0; lp[0] < 8; ++lp[0])
        vmatExecute_async(vc, fa, fb, &out, &clr);
    }
  }
  fenceVmat(0);
  vstSetTsize32B(8, 4);
  vstSetTstep32B_0(1, 16);
  vstSetTstep32B_1(8, 2048);
  vstRepeat(r, cc, (int)threadId * 128);
  fenceVst(0);
}

GD_INLINE void gd_bmm_cc(half *aa, half *bb, half *cc, __local__ f16v16 *pool) {
  const short M=128,N=128,K=64,AM=8,AN=8,AK=4;
  auto *a=pool+0;
  auto *b=pool+32;
  auto *r=pool+64;
  auto *tmp=pool+128;

  short lp[3];
  vldSetTsize32B(AM, AK);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(AM, (short)(16 * M));
  vldRepeat(tmp, aa, (int)threadId * M);
  vsfuSetTsize32B(AK, AM);
  vsfuSetTstep32B_0(1, AM, 0);
  vsfuSetTstep32B_1(AK, 1, 0);
  vsfuRepeat(a, tmp, (short)0, __ShuffleMode::Trs);
  vldSetTsize32B(AN, AK);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(AN, (short)(16 * N));
  vldRepeat(tmp, bb, (int)threadId * N);
  vsfuSetTsize32B(AK, AN);
  vsfuSetTstep32B_0(1, AN, 0);
  vsfuSetTstep32B_1(AK, 1, 0);
  vsfuRepeat(b, tmp, (short)0, __ShuffleMode::Trs);
  LoopStepInfo cs[3] = {{&lp[2], AN}, {&lp[1], 1}, {&lp[0], 0}};
  FragVData vc(r, cs);
  LoopStepInfo as[3] = {{&lp[2], AK}, {&lp[1], 0}, {&lp[0], 1}};
  FragVData fa(a, as);
  LoopStepInfo bs[3] = {{&lp[2], 0}, {&lp[1], AK}, {&lp[0], 1}};
  FragVData fb(b, bs);
  LoopOutInfo oi[1] = {{&lp[0]}};
  // Each output tile starts a fresh K reduction.
  LoopClrInfo ci[1] = {{&lp[0]}};
  LoopOutLPAttr out{oi};
  LoopClrLPAttr clr{ci};
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < AM; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < AN; ++lp[1]) {
#pragma rpu_hwloop_ul
      for (lp[0] = 0; lp[0] < AK; ++lp[0])
        vmatExecute_async(vc, fa, fb, &out, &clr);
    }
  }
  fenceVmat(0);
  vstSetTsize32B(AN, AM);
  vstSetTstep32B_0(1, 16);
  vstSetTstep32B_1(AN, (short)(16 * N));
  vstRepeat(r, cc, (int)threadId * N);
  fenceVst(0);
}

GD_INLINE void gd_tril(half *m, half *x, __local__ f16v16 *pool) {
  auto *mt=pool, *xt=pool+10, *xtt=pool+20;
  auto *work=pool+30, *wt=pool+34, *zero=pool+35;
  vcfg_set_valu(vcfg_valu_target::imm0, (half)1.0f);
  vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);

#define ID(B,C) ((B) * ((B) + 1) / 2 + (C))
#define LOAD(B,C) vldExecute_async(mt + ID(B,C), m, ((B) * 16 + threadId) * 64 + (C) * 16)
  LOAD(0,0);
  LOAD(1,0); LOAD(1,1);
  LOAD(2,0); LOAD(2,1); LOAD(2,2);
  LOAD(3,0); LOAD(3,1); LOAD(3,2); LOAD(3,3);
  fenceVld(0);

// Update one strict-lower row of the diagonal tile at a time.  At iteration r, rows < r already contain the inverse recurrence,
// while row r still contains M, so X[r,:] = M[r,:] + (X*X)[r,:].
#define COPY_DIAG(B) \
  valuExecute_async(xt + ID(B,B), mt + ID(B,B), (half)0.0f, __ValuOpc::add)
  COPY_DIAG(0); COPY_DIAG(1); COPY_DIAG(2); COPY_DIAG(3);
  fenceValu(0);
#define TRS_DIAG(B) \
  vsfuExecute_async(xtt + ID(B,B), xt + ID(B,B), (short)0, __ShuffleMode::Trs)
  TRS_DIAG(0); TRS_DIAG(1); TRS_DIAG(2); TRS_DIAG(3);
  fenceVsfu(0);

  vmatInit(xt, xt, xtt);
  valuInit(xt, xt, mt);
  vsfuInit(xtt, xt);
#pragma rpu_hwloop_ul
  for (short row = 2; row < 16; ++row) {
    // Match unit_tril_inv_fp16: VMAT executes uniformly across the warp,
    // while the per-thread strobe commits only the current recurrence row.
    __builtin_rhvm_set_strbs(threadId == row ? (short)0xffff : (short)0);
#define ROW_MUL(B) \
      vmatExecute_async(xt + ID(B,B), xt + ID(B,B), xtt + ID(B,B), \
                        __OutMode::outon, __ClrMode::clron, __StrbMode::strbs)
      ROW_MUL(0); ROW_MUL(1); ROW_MUL(2); ROW_MUL(3);
      fenceVmat(0);
#define ROW_ADD(B) \
      valuExecute_async(xt + ID(B,B), xt + ID(B,B), mt + ID(B,B), \
                        __ValuOpc::add, __StrbMode::strbs)
      ROW_ADD(0); ROW_ADD(1); ROW_ADD(2); ROW_ADD(3);
      fenceValu(0);
    TRS_DIAG(0); TRS_DIAG(1); TRS_DIAG(2); TRS_DIAG(3);
    fenceVsfu(0);
  }

  __builtin_rhvm_set_strbs((short)(1u << threadId));
#define SEED_DIAG(B) valuExecute_async(xt + ID(B,B), (half)1.0f, (half)0.0f, \
    __ValuOpc::add, __ClrMode::clron, __OutMode::outon, __StrbMode::strbs)
  SEED_DIAG(0); SEED_DIAG(1); SEED_DIAG(2); SEED_DIAG(3);
  fenceValu(0);
  TRS_DIAG(0); TRS_DIAG(1); TRS_DIAG(2);
  fenceVsfu(0);

#undef COPY_DIAG
#undef TRS_DIAG
#undef ROW_MUL
#undef ROW_ADD
#undef SEED_DIAG

// Compute T_bc=sum_k M_bk X_kc into one FP32 accumulator, then X_bc=X_bb T_bc.
#define TERM(B,C,K,MODE) vmatExecute_async(work, mt + ID(B,K), xtt + ID(K,C), __OutMode::MODE)
#define FINISH(B,C) { \
  vsfuExecute(wt, work, (short)0, __ShuffleMode::Trs); \
  vmatInit(xt + ID(B,C), xt + ID(B,B), wt, __AclrMode::aclr); \
  vmatExecute(xt + ID(B,C), xt + ID(B,B), wt, __OutMode::outon); \
  if ((B) < 3) \
    vsfuExecute(xtt + ID(B,C), xt + ID(B,C), (short)0, __ShuffleMode::Trs); \
}
#define SUM1(B,C,K0) { \
  vmatInit(work, mt + ID(B,K0), xtt + ID(K0,C), __AclrMode::aclr); \
  TERM(B,C,K0,outon); fenceVmat(0); FINISH(B,C); \
}
#define SUM2(B,C,K0,K1) { \
  vmatInit(work, mt + ID(B,K0), xtt + ID(K0,C), __AclrMode::aclr); \
  TERM(B,C,K0,outoff); TERM(B,C,K1,outon); fenceVmat(0); FINISH(B,C); \
}
#define SUM3(B,C,K0,K1,K2) { \
  vmatInit(work, mt + ID(B,K0), xtt + ID(K0,C), __AclrMode::aclr); \
  TERM(B,C,K0,outoff); TERM(B,C,K1,outoff); TERM(B,C,K2,outon); fenceVmat(0); FINISH(B,C); \
}

  SUM1(1,0,0);
  SUM2(2,0,0,1);
  SUM1(2,1,1);
  SUM3(3,0,0,1,2);
  SUM2(3,1,1,2);
  SUM1(3,2,2);

#define STORE(B,C) vstExecute_async(xt + ID(B,C), x, ((B) * 16 + threadId) * 64 + (C) * 16)
  STORE(0,0);
  STORE(1,0); STORE(1,1);
  STORE(2,0); STORE(2,1); STORE(2,2);
  STORE(3,0); STORE(3,1); STORE(3,2); STORE(3,3);
  if (m != x) {
    valuExecute(zero, (half)0.0f, (half)0.0f, __ValuOpc::add);
#define ZERO_UP(B,C) vstExecute_async(zero, x, ((B) * 16 + threadId) * 64 + (C) * 16)
    ZERO_UP(0,1); ZERO_UP(0,2); ZERO_UP(0,3);
    ZERO_UP(1,2); ZERO_UP(1,3); ZERO_UP(2,3);
#undef ZERO_UP
  }
  fenceVst(0);

#undef ID
#undef LOAD
#undef COPY_DIAG
#undef SEED_DIAG
#undef ROUND_TRS
#undef ROUND_SQ
#undef ROUND_MUL
#undef ROUND_ADD
#undef ROUND_ALL
#undef TERM
#undef FINISH
#undef SUM1
#undef SUM2
#undef SUM3
#undef STORE
}

GD_INLINE void gd_reset() {
  vcfg_set_valu(vcfg_valu_target::imm0,(half)1.0f);
  vcfg_set_valu(vcfg_valu_target::imm1,(half)1.0f);
  vcfg_set_valu(vcfg_valu_target::imm2,(half)1.0f);
  vcfg_set_vmat(vcfg_vmat_target::acc_init_fp32_l,(short)0);
  vcfg_set_vmat(vcfg_vmat_target::acc_init_fp32_h,(short)0);
  vcfg_set_vmat(vcfg_vmat_target::scale_allch_l,(short)0);
  vcfg_set_vmat(vcfg_vmat_target::scale_allch_h,(short)0x3f80);
}

// Every inline helper completes stores before a later helper reloads scratch.
template <int ELEMS, int OP>
GD_INLINE void gd_binary(half *dst, half *a, half *b, __local__ f16v16 *pool) {
  auto *x=pool,*y=pool+64;
  vldSetTsize32B(ELEMS/256,1); vldSetTstep32B_0(1,256); vldSetTstep32B_1(0,0);
  vldRepeat_async(x,a,(int)threadId*16); vldRepeat_async(y,b,(int)threadId*16); fenceVld(0);
  valuSetTsize(ELEMS/256,1); valuSetTstep0(1,1,1); valuSetTstep1(0,0,0);
  if (OP==0) valuRepeat(x,x,y,__ValuOpc::add);
  if (OP==1) valuRepeat(x,x,y,__ValuOpc::sub);
  if (OP==2) valuRepeat(x,x,y,__ValuOpc::mul);
  vstSetTsize32B(ELEMS/256,1); vstSetTstep32B_0(1,256); vstSetTstep32B_1(0,0);
  vstRepeat(x,dst,(int)threadId*16); fenceVst(0);
}

template <int ROWS, int COLS>
GD_INLINE void gd_bcast_mul(half *dst, half *src, half *scalars, __local__ f16v16 *pool) {
  auto *x=pool,*s=pool+16,*v=pool+17;
  LoopStrbInfo one((short)1); LoopStrbLPAttr mask(&one);
#pragma rpu_hwloop_ul
  for (short base=0;base<ROWS;base+=16) {
    const int row=base+threadId;
    vldSetTsize32B(COLS/16,1); vldSetTstep32B_0(1,16); vldSetTstep32B_1(0,0);
    vldRepeat(x,src,row*COLS); vldExecute(s,scalars,row,&mask);
    vsfuExecute(v,s,(short)0,__ShuffleMode::Psel);
    valuSetTsize(COLS/16,1); valuSetTstep0(1,1,0); valuSetTstep1(0,0,0);
    valuRepeat(x,x,v,__ValuOpc::mul);
    vstSetTsize32B(COLS/16,1); vstSetTstep32B_0(1,16); vstSetTstep32B_1(0,0);
    vstRepeat(x,dst,row*COLS);
  }
  fenceVst(0);
}

GD_INLINE void gd_scalar_mul(half *dst, half *src, half *scalar,
                             __local__ f16v16 *pool) {
  auto *x=pool,*s=pool+64,*v=pool+65;
  LoopStrbInfo one((short)1); LoopStrbLPAttr mask(&one);
  vldExecute(s,scalar,0,&mask); vsfuExecute(v,s,(short)0,__ShuffleMode::Psel);
  vldSetTsize32B(64,1); vldSetTstep32B_0(1,256); vldSetTstep32B_1(0,0);
  vldRepeat(x,src,(int)threadId*16);
  valuSetTsize(64,1); valuSetTstep0(1,1,0); valuSetTstep1(0,0,0);
  valuRepeat(x,x,v,__ValuOpc::mul);
  vstSetTsize32B(64,1); vstSetTstep32B_0(1,256); vstSetTstep32B_1(0,0);
  vstRepeat(x,dst,(int)threadId*16); fenceVst(0);
}

// Row-wise normalized q/k and unnormalized v go directly into head-major scratch.
GD_INLINE void gd_prepare_qk(half *src,half *dst,short head,short start,
                             short valid,bool scale_q,__local__ f16v16 *pool) {
  auto *x=pool,*gram=pool+8,*norm=pool+9,*scratch=pool+10;
  LoopOutInfo oi[1]={{(short)0}}; LoopClrInfo ci[1]={{(short)0}};
  LoopOutLPAttr out(oi); LoopClrLPAttr clr(ci);
#pragma rpu_hwloop_ul
  for (short base=0;base<64;base+=16) {
    const short row=base+threadId;
    vldSetTsize32B(8,1); vldSetTstep32B_0(1,16); vldSetTstep32B_1(0,0);
    vldRepeat(x,src,((int)start+row)*256+(int)head*128);
    vmatSetTsize(8,1); vmatSetTstep0(0,1,1); vmatSetTstep1(0,0,0);
    vmatRepeat(gram,x,x,&out,&clr);
    half sum_square;
    vcfg_set_valu(vcfg_valu_target::imm2,gram,(short)threadId);
    valuExecute(scratch,(half)1.0f,(half)0.0f,__ValuOpc::add);
    vcfg_set_valu(vcfg_valu_target::imm2,(half)1.0f);
    vsfuScalar(sum_square,scratch,(short)0,__ShuffleMode::Psel);
    const unsigned short bits=reinterpret_cast<unsigned short&>(sum_square);
    const unsigned short small=sumint(bits);
    const unsigned short large=sumint((unsigned short)(0xffffu-bits));
    valuSetTsize(8,1); valuSetTstep0(1,1,0); valuSetTstep1(0,0,0);
    half epsilon=(half)1e-6f;
    if (small<0x0400u || large<=0x83ffu) {
      half factor=(half)1.0f;
      if (bits<0x0400u) { factor=(half)128.0f; epsilon=(half)0.016384f; }
      // Match standalone C128 L2Norm: 1/256 may overflow the SUM again.
      // 1/4096 bounds every finite FP16 row's scaled sum below 32737.
      else if(bits>=0x7c00u) { factor=(half)0.000244140625f; epsilon=(half)0.0f; }
      valuRepeat(x,x,factor,__ValuOpc::mul);
      vmatRepeat(gram,x,x,&out,&clr);
      vcfg_set_valu(vcfg_valu_target::imm2,gram,(short)threadId);
      valuExecute(scratch,(half)1.0f,(half)0.0f,__ValuOpc::add);
      vcfg_set_valu(vcfg_valu_target::imm2,(half)1.0f);
      vsfuScalar(sum_square,scratch,(short)0,__ShuffleMode::Psel);
    }
    valuExecute(norm,sum_square,epsilon,__ValuOpc::add);
    valuExecute(norm,norm,(half)0.0f,__ValuOpc::sqrt_reciprocal);
    valuRepeat(x,x,norm,__ValuOpc::mul);
    if (scale_q) valuRepeat(x,x,(half)0.08838834764831845f,__ValuOpc::mul);
    // Padding repeats scalar zeros. Advance only the vector destination.
    valuSetTstep0(1,0,0);
    if(start+row>=valid) valuRepeat(x,(half)0.0f,(half)0.0f,__ValuOpc::add);
    vstSetTsize32B(8,1); vstSetTstep32B_0(1,16); vstSetTstep32B_1(0,0);
    vstRepeat(x,dst,(int)row*128); fenceVst(0);
  }
}

GD_INLINE void gd_prepare_v(half *src,half *dst,short head,short start,
                            short valid,__local__ f16v16 *pool) {
#pragma rpu_hwloop_ul
  for(short base=0;base<64;base+=16) {
    const short row=base+threadId;
    vldSetTsize32B(8,1); vldSetTstep32B_0(1,16); vldSetTstep32B_1(0,0);
    vldRepeat(pool,src,((int)start+row)*256+(int)head*128);
    if(start+row>=valid) {
      valuSetTsize(8,1); valuSetTstep0(1,0,0); valuSetTstep1(0,0,0);
      valuRepeat(pool,(half)0.0f,(half)0.0f,__ValuOpc::add);
    }
    vstSetTsize32B(8,1); vstSetTstep32B_0(1,16); vstSetTstep32B_1(0,0);
    vstRepeat(pool,dst,(int)row*128); fenceVst(0);
  }
}

GD_INLINE void gd_prepare_bg(half *a,half *b,half *dt,half *negA,half *beta,half *g,
                             short head,short start,short valid,__local__ f16v16 *pool) {
  auto *av=pool,*bv=pool+1,*dv=pool+2,*nv=pool+3,*mx=pool+4,*tmp=pool+5;
  LoopStrbInfo one((short)1); LoopStrbLPAttr mask(&one);
  vldExecute(dv,dt,(int)head,&mask); vldExecute(nv,negA,(int)head,&mask);
#pragma rpu_hwloop_ul
  for(short base=0;base<64;base+=16) {
    const short row=base+threadId;
    vldExecute(av,a,((int)start+row)*16+head,&mask);
    vldExecute(bv,b,((int)start+row)*16+head,&mask);
    valuExecute(bv,bv,(half)0.0f,__ValuOpc::sigmoid);
    valuExecute(av,av,dv,__ValuOpc::add);
    valuExecute(mx,av,(half)0.0f,__ValuOpc::max);
    vcfg_set_valu(vcfg_valu_target::imm0,(half)2.0f);
    valuExecute(tmp,mx,av,__ValuOpc::sub);
    vcfg_set_valu(vcfg_valu_target::imm0,(half)1.0f);
    valuExecute(tmp,tmp,(half)0.0f,__ValuOpc::sigmoid);
    valuExecute(tmp,tmp,(half)0.0f,__ValuOpc::log);
    valuExecute(av,mx,tmp,__ValuOpc::sub);
    valuExecute(av,av,nv,__ValuOpc::mul);
    if(start+row>=valid) {
      valuExecute(av,(half)0.0f,(half)0.0f,__ValuOpc::add);
      valuExecute(bv,(half)0.0f,(half)0.0f,__ValuOpc::add);
    }
    vstExecute(av,g,(int)row,&mask); vstExecute(bv,beta,(int)row,&mask);
  }
  fenceVst(0);
}

GD_INLINE void gd_scan(half *src,half *dst,__local__ f16v16 *pool) {
  auto *input=pool,*expanded=pool+4,*prefix=pool+68;
  vldSetTsize32B(4,1); vldSetTstep32B_0(1,16); vldSetTstep32B_1(0,0);
  vldRepeat(input,src,0);
  vsfuSetTsize32B(8,4); vsfuSetTstep32B_0(2,0,1); vsfuSetTstep32B_1(16,1,0);
  vsfuRepeat(expanded,input,(short)0,__ShuffleMode::Psel);
  vsfuRepeat(expanded+1,input,(short)1,__ShuffleMode::Psel);
  valuExecute(prefix,expanded,(half)0.0f,__ValuOpc::acc,__ClrMode::clron,__OutMode::outon);
  valuSetTsize(63,1); valuSetTstep0(1,1,0); valuSetTstep1(0,0,0);
  valuRepeat(prefix+1,expanded+1,(half)0.0f,__ValuOpc::acc,__ClrMode::clroff,__OutMode::outon);
  __builtin_rhvm_set_strbs((short)1);
  vstSetTsize32B(64,1); vstSetTstep32B_0(1,1); vstSetTstep32B_1(0,0);
  if(threadId==0) vstRepeat(prefix,dst,0,__StrbMode::strbs);
  fenceVst(0);
}

// lower-triangular exp(g_i-g_j), retaining the subtraction/exp FP16 boundaries.
GD_INLINE void gd_decay(half *gc,half *decay,__local__ f16v16 *pool) {
  auto *col=pool,*row=pool+4,*s=pool+5,*x=pool+6;
  LoopStrbInfo one((short)1); LoopStrbLPAttr mask(&one);
  vldSetTsize32B(4,1); vldSetTstep32B_0(1,16); vldSetTstep32B_1(0,0); vldRepeat(col,gc,0);
#pragma rpu_hwloop_ul
  for(short base=0;base<64;base+=16) {
    const short ri=base+threadId;
    vldExecute(s,gc,(int)ri,&mask); vsfuExecute(row,s,(short)0,__ShuffleMode::Psel);
    valuSetTsize(4,1); valuSetTstep0(1,0,1); valuSetTstep1(0,0,0);
    valuRepeat(x,row,col,__ValuOpc::sub);
    valuSetTstep0(1,1,0); valuRepeat(x,x,(half)0.0f,__ValuOpc::exp);
#pragma rpu_hwloop_ul
    for(short block=0;block<4;++block) {
      const short count=ri-block*16+1;
      if(count<16) {
        const short live=count>0?count:0;
        __builtin_rhvm_set_strbs((short)~bmsk(0,live));
        valuExecute(x+block,(half)0.0f,(half)0.0f,__ValuOpc::add,__ClrMode::clron,__OutMode::outon,__StrbMode::strbs);
      }
    }
    vstSetTsize32B(4,1); vstSetTstep32B_0(1,16); vstSetTstep32B_1(0,0);
    vstRepeat(x,decay,(int)ri*64); fenceVst(0);
  }
}

GD_INLINE void gd_strict_neg(half *attn,half *decay,__local__ f16v16 *pool) {
  auto *x=pool,*d=pool+4;
#pragma rpu_hwloop_ul
  for(short base=0;base<64;base+=16) {
    const short ri=base+threadId;
    vldSetTsize32B(4,1); vldSetTstep32B_0(1,16); vldSetTstep32B_1(0,0);
    vldRepeat(x,attn,(int)ri*64); vldRepeat(d,decay,(int)ri*64);
    valuSetTsize(4,1); valuSetTstep0(1,1,0); valuSetTstep1(0,0,0);
    valuRepeat(x,x,(half)-1.0f,__ValuOpc::mul);
    valuSetTstep0(1,1,1); valuRepeat(x,x,d,__ValuOpc::mul);
#pragma rpu_hwloop_ul
    for(short block=0;block<4;++block) {
      const short count=ri-block*16;
      if(count<16) {
        const short live=count>0?count:0;
        __builtin_rhvm_set_strbs((short)~bmsk(0,live));
        valuExecute(x+block,(half)0.0f,(half)0.0f,__ValuOpc::add,__ClrMode::clron,__OutMode::outon,__StrbMode::strbs);
      }
    }
    vstSetTsize32B(4,1); vstSetTstep32B_0(1,16); vstSetTstep32B_1(0,0);
    vstRepeat(x,attn,(int)ri*64); fenceVst(0);
  }
}

GD_INLINE void gd_expg(half *gc,half *ge,bool last_minus,__local__ f16v16 *pool) {
  auto *x=pool,*s=pool+4,*last=pool+5;
  vldSetTsize32B(4,1); vldSetTstep32B_0(1,16); vldSetTstep32B_1(0,0); vldRepeat(x,gc,0);
  valuSetTsize(4,1); valuSetTstep1(0,0,0);
  if(last_minus) {
    LoopStrbInfo one((short)1); LoopStrbLPAttr mask(&one);
    vldExecute(s,gc,63,&mask); vsfuExecute(last,s,(short)0,__ShuffleMode::Psel);
    valuSetTstep0(1,0,1); valuRepeat(x,last,x,__ValuOpc::sub);
  }
  valuSetTstep0(1,1,0); valuRepeat(x,x,(half)0.0f,__ValuOpc::exp);
  vstSetTsize32B(4,1); vstSetTstep32B_0(1,16); vstSetTstep32B_1(0,0);
  if(threadId==0) vstRepeat(x,ge,0); fenceVst(0);
}

GD_INLINE void gd_norm_gate(half *src,half *z,half *gamma,half *dst,short head,
                            short start,half *raw_out,half *norm_out,bool debug,__local__ f16v16 *pool) {
  auto *x=pool,*w=pool+8,*gate=pool+16,*sg=pool+24,*gram=pool+32,*norm=pool+33,*scratch=pool+34;
  vcfg_set_vmat(vcfg_vmat_target::scale_allch_l,(short)0);
  vcfg_set_vmat(vcfg_vmat_target::scale_allch_h,(short)0x3c00); // 1/128
  LoopOutInfo oi[1]={{(short)0}}; LoopClrInfo ci[1]={{(short)0}};
  LoopOutLPAttr out(oi); LoopClrLPAttr clr(ci);
  vldSetTsize32B(8,1); vldSetTstep32B_0(1,16); vldSetTstep32B_1(0,0); vldRepeat(w,gamma,0);
#pragma rpu_hwloop_ul
  for(short base=0;base<64;base+=16) {
    const short row=base+threadId;
    const int token=((int)start+row)*256+(int)head*128;
    vldSetTsize32B(8,1); vldSetTstep32B_0(1,16); vldSetTstep32B_1(0,0);
    vldRepeat(x,src,(int)row*128); vldRepeat(gate,z,token);
    if(debug) {
      vstSetTsize32B(8,1); vstSetTstep32B_0(1,16); vstSetTstep32B_1(0,0);
      vstRepeat(x,raw_out,token); fenceVst(0);
    }
    valuSetTsize(8,1); valuSetTstep0(1,1,0); valuSetTstep1(0,0,0);
    valuRepeat(x,x,(half)128.0f,__ValuOpc::mul);
    vmatSetTsize(8,1); vmatSetTstep0(0,1,1); vmatSetTstep1(0,0,0); vmatRepeat(gram,x,x,&out,&clr);
    half mean_square;
    vcfg_set_valu(vcfg_valu_target::imm2,gram,(short)threadId);
    valuExecute(scratch,(half)1.0f,(half)0.0f,__ValuOpc::add);
    vcfg_set_valu(vcfg_valu_target::imm2,(half)1.0f);
    vsfuScalar(mean_square,scratch,(short)0,__ShuffleMode::Psel);
    valuExecute(norm,mean_square,(half)0.016384f,__ValuOpc::add);
    valuExecute(norm,norm,(half)0.0f,__ValuOpc::sqrt_reciprocal);
    valuRepeat(x,x,norm,__ValuOpc::mul);
    valuSetTstep0(1,1,1); valuRepeat(x,x,w,__ValuOpc::mul);
    if(debug) {
      vstSetTsize32B(8,1); vstSetTstep32B_0(1,16); vstSetTstep32B_1(0,0);
      vstRepeat(x,norm_out,token); fenceVst(0);
    }
    valuSetTstep0(1,1,0); valuRepeat(sg,gate,(half)0.0f,__ValuOpc::sigmoid);
    valuSetTstep0(1,1,1); valuRepeat(gate,gate,sg,__ValuOpc::mul);
    valuRepeat(x,gate,x,__ValuOpc::mul);
    vstSetTsize32B(8,1); vstSetTstep32B_0(1,16); vstSetTstep32B_1(0,0);
    vstRepeat(x,dst,token); fenceVst(0);
  }
}
