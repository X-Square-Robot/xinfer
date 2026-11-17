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

"""GQA-GLUE knives 1+3: BHSD partial rope (rotary_dim 64) with in-kernel
angle duplication.

Two changes vs the in-service rope_half_d64 (v2.1), both value-preserving:

1. BHSD addressing (knife 1): q/k are read and written head-major
   ``[H, rows, 256]`` -- per-head contiguous staging blocks, pure index
   permutation of the same values, so the qkv BHSD kernel's outputs feed
   straight in and the FIA BNSD inputs / KV-cache slots read straight out.
2. In-kernel angle duplication (knife 3): the wrapper's fresh ``cat(cos, cos)``
   disappears; the kernel consumes the compact width-32 tables and duplicates
   them to 64 lanes itself. The 2026-11-15 v3 attempt failed because
   ``vmul`` was fed a vector_bool as an *operand* (asc_mul has no such
   overload); here the bool vector is the *predicate* of a multiply by
   vdup(1.0) -- exactly the ``vmul(w_v, s_v, hi_m)`` pattern the shipped v2.1
   body already proves -- so active lanes keep exact values and masked lanes
   are zero; the duplicate is the sum of the two half-masked products.

The rotation semantics (u/w/z neighbor loads, per-product rounding, one RNE
store round, exact bf16->f32->bf16 pass-through of channels 64..255) are
verbatim from the shipped v2.1 kernel.
"""

import tilelang.ascend.language as T

H = 8
KV = 2
D = 256
CW = 32  # compact cos/sin width; duplicated to 64 lanes in-kernel
CORES = 64
ROWS = 8
GUARD = 64


def _head_lines(buf, out, heads, R):
    lines = []
    for h in range(heads):
        lines.append("base = (%d * %d + r) * %d" % (h, R, D))
        lines.append('u_v = T.simd.vcvt(T.simd.vld(%s[gb + base], dist="UNPK_B16"), "float32", part=0)' % buf)
        lines.append('w_v = T.simd.vcvt(T.simd.vld(%s[gb + base - 32], dist="UNPK_B16"), "float32", part=0)' % buf)
        lines.append('z_v = T.simd.vcvt(T.simd.vld(%s[gb + base + 32], dist="UNPK_B16"), "float32", part=0)' % buf)
        lines.append('t1_v = T.simd.vmul(u_v, c_v, m)')
        lines.append('res_v = T.simd.vsub(T.simd.vadd(t1_v, T.simd.vmul(w_v, s_v, hi_m), m),'
                     ' T.simd.vmul(z_v, s_v, lo_m), m)')
        lines.append('T.simd.vsts(%s[base], T.simd.vcvt(res_v, "bfloat16", round="ROUND_R"), dist="PK_B32")' % out)
        for j in (64, 128, 192):
            lines.append('p_v = T.simd.vcvt(T.simd.vld(%s[gb + base + %d], dist="UNPK_B16"), "float32", part=0)' % (buf, j))
            lines.append('T.simd.vsts(%s[base + %d], T.simd.vcvt(p_v, "bfloat16", round="ROUND_R"), dist="PK_B32")' % (out, j))
    return lines


def rope_half_d64_bhsd():
    R = ROWS
    q_text = "\n".join("                            " + ln for ln in _head_lines("q_ub", "qo_ub", H, R))
    ko_text = "\n".join("                            " + ln for ln in _head_lines("k_ub", "ko_ub", KV, R))

    # per-head contiguous staging copies (BHSD global <-> head-major UB);
    # every R/GUARD below is already the literal number.
    q_in = "\n".join(
        "                    T.copy(q[({h} * rows + r0) * 256 : ({h} * rows + r0 + {R}) * 256],"
        " q_ub[{G} + {h} * {R256} : {G} + {h} * {R256} + {R256}])".format(h=h, R=R, R256=R * 256, G=GUARD)
        for h in range(H))
    k_in = "\n".join(
        "                    T.copy(k[({h} * rows + r0) * 256 : ({h} * rows + r0 + {R}) * 256],"
        " k_ub[{G} + {h} * {R256} : {G} + {h} * {R256} + {R256}])".format(h=h, R=R, R256=R * 256, G=GUARD)
        for h in range(KV))
    q_out = "\n".join(
        "                    T.copy(qo_ub[{h} * {R256} : {h} * {R256} + {R256}],"
        " qo[({h} * rows + r0) * 256 : ({h} * rows + r0 + {R}) * 256])".format(h=h, R=R, R256=R * 256)
        for h in range(H))
    k_out = "\n".join(
        "                    T.copy(ko_ub[{h} * {R256} : {h} * {R256} + {R256}],"
        " ko[({h} * rows + r0) * 256 : ({h} * rows + r0 + {R}) * 256])".format(h=h, R=R, R256=R * 256)
        for h in range(KV))

    template = """
    @T.prim_func
    def main(
        qh: T.handle,
        kh: T.handle,
        ch: T.handle,
        sh: T.handle,
        qoh: T.handle,
        koh: T.handle,
        rows: T.int64,
    ):
        q = T.match_buffer(qh, (rows * QW,), "bfloat16")
        k = T.match_buffer(kh, (rows * KW,), "bfloat16")
        cosine = T.match_buffer(ch, (rows * CWD,), "float32")
        sine = T.match_buffer(sh, (rows * CWD,), "float32")
        qo = T.match_buffer(qoh, (rows * QW,), "bfloat16")
        ko = T.match_buffer(koh, (rows * KW,), "bfloat16")
        with T.Kernel(CORES) as core:
            q_ub = T.alloc_shared((QW * R + GUARD_E,), "bfloat16")
            k_ub = T.alloc_shared((KW * R + GUARD_E,), "bfloat16")
            c_ub = T.alloc_shared((CWD * R + 192,), "float32")
            s_ub = T.alloc_shared((CWD * R + 192,), "float32")
            qo_ub = T.alloc_shared((QW * R,), "bfloat16")
            ko_ub = T.alloc_shared((KW * R,), "bfloat16")
            T.annotate_buffer_versions({q_ub: 2, k_ub: 2, c_ub: 2, s_ub: 2, qo_ub: 1, ko_ub: 1})
            for step in T.Pipelined(T.ceildiv(rows, CORES * R), num_stages=2):
                r0 = (step * CORES + core) * R
                if r0 < rows:
                    cbase = r0 * CWD
                    T.copy(cosine[cbase : cbase + R * CWD], c_ub[64 : 64 + R * CWD])
                    T.copy(sine[cbase : cbase + R * CWD], s_ub[64 : 64 + R * CWD])
Q_IN
K_IN
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        lane = T.simd.vci(0, "float32")
                        lo_m = T.simd.vcmps(lane, 32.0, m, op="lt")
                        hi_m = T.simd.vcmps(lane, 31.5, m, op="gt")
                        one = T.simd.vdup(1.0, "float32")
                        for r in range(R):
                            c_v = T.simd.vadd(T.simd.vmul(T.simd.vld(c_ub[64 + r * CWD]), one, lo_m),
                                              T.simd.vmul(T.simd.vld(c_ub[64 + r * CWD - 32]), one, hi_m), m)
                            s_v = T.simd.vadd(T.simd.vmul(T.simd.vld(s_ub[64 + r * CWD]), one, lo_m),
                                              T.simd.vmul(T.simd.vld(s_ub[64 + r * CWD - 32]), one, hi_m), m)
                            gb = GUARD_E
Q_BODY
K_BODY
Q_OUT
K_OUT
"""
    # template-level tokens first (word-boundary R only; CORES must survive);
    # the inserted bodies carry no tokens.
    template = template.replace("QW", str(H * D)).replace("KW", str(KV * D))
    template = template.replace("CWD", str(CW)).replace("GUARD_E", str(GUARD))
    import re
    template = re.sub(r"\bR\b", str(R), template)
    template = template.replace("Q_IN", q_in).replace("K_IN", k_in)
    template = template.replace("Q_BODY", q_text).replace("K_BODY", ko_text)
    template = template.replace("Q_OUT", q_out).replace("K_OUT", k_out)
    template = chr(10).join(
        line[4:] if line.startswith("    ") else line for line in template.split(chr(10))
    )
    namespace = {"T": T, "CORES": CORES, "R": R}
    source_name = "<rope-vec-d64-bhsd>"
    import linecache
    linecache.cache[source_name] = (
        len(template), None,
        [line + chr(10) for line in template.split(chr(10))], source_name
    )
    exec(compile(template, source_name, "exec"), namespace)
    return namespace["main"]


COMPILE_OPTIONS = ()


def build_programs():
    yield "rope_half_d64_bhsd", rope_half_d64_bhsd()
