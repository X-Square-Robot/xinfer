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

"""Compute beta and the gate pre-activation from strided Gated DeltaNet Z/B/A rows.

Beta is the BF16-rounded FP32 sigmoid of its input. The gate pre-activation
adds the per-head dt bias in FP32. The caller uses ACLNN softplus and
multiplies by the negative exponential of A_log."""

import tilelang.ascend.language as T

CORES = 64
COMPILE_OPTIONS = ["-ffp-contract=off"]

def gdrprep_bf16(hv=16, ldz=2080, cores=CORES):
    """Read beta logits and gate logits from strided Z/B/A rows; store BF16 sigmoid and FP32 gate plus dt bias."""
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
