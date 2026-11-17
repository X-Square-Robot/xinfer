# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""GDN-VENDOR kernels: gates-chain fuse for the 2B GDN vendor path.

gdrprep_bf16: one pass over the strided zba rows [T, 2080] writing
  beta[t,h] = RNE_bf16(1/(1+exp(-f32(zba[t, 2048+h]))))   (bitwise vs aclnn
              SigmoidAiCore: probe_sig2 v0, n_diff=0 over all 65536 bf16)
  gpre[t,h] = f32(zba[t, 2064+h]) + dt[h]                  (IEEE-exact)
Softplus and the (-exp(A_log)) multiply stay on aclnn: SoftplusV2 has a
log1p tail that vln(1+exp(x)) cannot reproduce bitwise (probe_sp2), and the
g-amp advisory (R/gdn-k3-r2) forbids non-bitwise g inputs.
"""

import tilelang.ascend.language as T

CORES = 64
COMPILE_OPTIONS = ["-ffp-contract=off"]

def gdrprep_bf16(hv=16, ldz=2080, cores=CORES):
    """One pass over the strided zba rows: beta + the g-chain prefix.

    beta[t, h] = RNE_bf16(1 / (1 + exp(-f32(zba[t, hv*128 + h]))))
    gpre[t, h] = f32(zba[t, hv*128 + hv + h]) + dt[h]
    dt is pre-duplicated into a 32-lane pad inside the kernel. The sigmoid
    composition is bitwise against aclnn SigmoidAiCore over the whole bf16
    domain (results/probe-tr-bitwise.json v0: n_diff=0/65536); the add/cast
    ops are IEEE-exact. Softplus and the premul multiply stay on aclnn
    (SoftplusV2's log1p tail is NOT bitwise reproducible with vln).
    """
    boff = hv * 128
    aoff = boff + hv

    @T.prim_func
    def main(zh: T.handle, dth: T.handle, bh: T.handle, gh: T.handle, tokens: T.int64):
        z = T.match_buffer(zh, (tokens * ldz,), "bfloat16")
        dt = T.match_buffer(dth, (hv,), "float32")
        beta = T.match_buffer(bh, (tokens * hv,), "bfloat16")
        gpre = T.match_buffer(gh, (tokens * hv,), "float32")
        tiles = T.ceildiv(tokens, 2)
        with T.Kernel(cores) as core:
            # 64-wide UB slots: one vld/vsts pair moves 64 lanes (UNPK_B16
            # 128B); 32-wide slots would be overrun by the wider store.
            b_ub = T.alloc_shared((64,), "bfloat16")
            a_ub = T.alloc_shared((64,), "bfloat16")
            dt_ub = T.alloc_shared((64,), "float32")
            beta_ub = T.alloc_shared((64,), "bfloat16")
            g_ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions(
                {b_ub: 2, a_ub: 2, dt_ub: 2, beta_ub: 2, g_ub: 2})
            T.copy(dt, dt_ub[0:hv])
            T.copy(dt, dt_ub[hv:2 * hv])
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    r0 = tile * 2
                    z0 = T.cast(r0 * ldz + boff, "int32")
                    za0 = T.cast(r0 * ldz + aoff, "int32")
                    T.copy(z[z0:z0 + hv], b_ub[0:hv])
                    T.copy(z[za0:za0 + hv], a_ub[0:hv])
                    if r0 + 1 < tokens:
                        z1 = T.cast((r0 + 1) * ldz + boff, "int32")
                        za1 = T.cast((r0 + 1) * ldz + aoff, "int32")
                        T.copy(z[z1:z1 + hv], b_ub[hv:2 * hv])
                        T.copy(z[za1:za1 + hv], a_ub[hv:2 * hv])
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        one = T.simd.vdup(1.0, "float32", m)
                        bv = T.simd.vcvt(
                            T.simd.vld(b_ub[0], dist="UNPK_B16"),
                            "float32", part=0)
                        e0 = T.simd.vexp(T.simd.vneg(bv, m), m)
                        s0 = T.simd.vdiv(
                            one, T.simd.vadd(e0, one, m), m, precision="exact")
                        T.simd.vsts(
                            beta_ub[0],
                            T.simd.vcvt(s0, "bfloat16", round="ROUND_R"),
                            dist="PK_B32")
                        av = T.simd.vcvt(
                            T.simd.vld(a_ub[0], dist="UNPK_B16"),
                            "float32", part=0)
                        T.simd.vsts(g_ub[0], T.simd.vadd(av, T.simd.vld(dt_ub[0]), m))
                    bo = T.cast(r0 * hv, "int32")
                    T.copy(beta_ub[0:hv], beta[bo:bo + hv])
                    T.copy(g_ub[0:hv], gpre[bo:bo + hv])
                    if r0 + 1 < tokens:
                        T.copy(beta_ub[hv:2 * hv], beta[bo + hv:bo + 2 * hv])
                        T.copy(g_ub[hv:2 * hv], gpre[bo + hv:bo + 2 * hv])
    return main


def build_programs():
    yield "gdrprep_bf16", gdrprep_bf16()
