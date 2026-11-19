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

"""Split prefill QKV, apply Q/K normalization and emit head-major Q/K/V.
Apply the zero-centered factor before the supplied reciprocal standard
deviation in FP32 and round once to BF16. The gate remains token-major.
Caller-supplied output buffers allow K/V to be written into cache slots."""

import tilelang.ascend.language as T

W = 5120
QW = 2048
HEADS = 8
KHEADS = 2
D = 256
CW = 64  # vector chunk width
CORES = 64
ROWS = 8


def qkv_split_vec_bhsd():
    R = ROWS
    lines = []
    # q/gate: per (row, head) -- q lands head-major in UB, gate row-major.
    for h in range(HEADS):
        lines.append("inv = T.simd.vdup(rq_ub[r * 8 + %d], \"float32\")" % h)
        for c in range(4):
            off = "r * %d + %d" % (W, h * 512 + c * CW)
            dst = "((%d * RD + r) * %d + %d)" % (h, D, c * CW)
            lines.append('xa = T.simd.vcvt(T.simd.vld(qp_ub[%s], dist="UNPK_B16"), "float32", part=0)' % off)
            lines.append('t1 = T.simd.vmul(xa, T.simd.vld(qf_ub[%d]), m)' % (c * CW))
            lines.append('zz = T.simd.vmul(t1, inv, m)')
            lines.append('T.simd.vsts(q_ub[%s], T.simd.vcvt(zz, "bfloat16", round="ROUND_R"), dist="PK_B32")' % dst)
        for c in range(4):
            off = "r * %d + %d" % (W, h * 512 + 256 + c * CW)
            dst = "r * %d + %d" % (QW, h * D + c * CW)
            lines.append('pa = T.simd.vcvt(T.simd.vld(qp_ub[%s], dist="UNPK_B16"), "float32", part=0)' % off)
            lines.append('T.simd.vsts(g_ub[%s], T.simd.vcvt(pa, "bfloat16", round="ROUND_R"), dist="PK_B32")' % dst)
    for h in range(KHEADS):
        lines.append("kinv = T.simd.vdup(rk_ub[r * 2 + %d], \"float32\")" % h)
        for c in range(4):
            off = "r * %d + %d" % (W, 4096 + h * 256 + c * CW)
            dst = "((%d * RD + r) * %d + %d)" % (h, D, c * CW)
            lines.append('ka = T.simd.vcvt(T.simd.vld(qp_ub[%s], dist="UNPK_B16"), "float32", part=0)' % off)
            lines.append('kt1 = T.simd.vmul(ka, T.simd.vld(kf_ub[%d]), m)' % (c * CW))
            lines.append('kz = T.simd.vmul(kt1, kinv, m)')
            lines.append('T.simd.vsts(ko_ub[%s], T.simd.vcvt(kz, "bfloat16", round="ROUND_R"), dist="PK_B32")' % dst)
    # v: two heads x 256; the eight 64-lane chunks cover head c//4.
    for c in range(8):
        off = "r * %d + %d" % (W, 4608 + c * CW)
        dst = "((%d * RD + r) * %d + %d)" % (c // 4, D, (c % 4) * CW)
        lines.append('va = T.simd.vcvt(T.simd.vld(qp_ub[%s], dist="UNPK_B16"), "float32", part=0)' % off)
        lines.append('T.simd.vsts(vo_ub[%s], T.simd.vcvt(va, "bfloat16", round="ROUND_R"), dist="PK_B32")' % dst)

    text = "\n".join("                            " + ln for ln in lines)

    # copy-out: per head, contiguous RD*256 blocks on both sides (BHSD global
    # [H, rows, D]; head-major UB). v is staged in its own head-major region.
    q_copies = "\n".join(
        "                    T.copy(q_ub[{h} * RD * 256 : ({h} + 1) * RD * 256],"
        " q[({h} * rows + r0) * 256 : ({h} * rows + r0 + RD) * 256])".format(h=h)
        for h in range(HEADS))
    k_copies = "\n".join(
        "                    T.copy(ko_ub[{h} * RD * 256 : ({h} + 1) * RD * 256],"
        " k[({h} * rows + r0) * 256 : ({h} * rows + r0 + RD) * 256])".format(h=h)
        for h in range(KHEADS))
    v_copies = "\n".join(
        "                    T.copy(vo_ub[{h} * RD * 256 : ({h} + 1) * RD * 256],"
        " v[({h} * rows + r0) * 256 : ({h} * rows + r0 + RD) * 256])".format(h=h)
        for h in range(KHEADS))

    template = """
    @T.prim_func
    def main(
        ph: T.handle,
        rqh: T.handle,
        rkh: T.handle,
        qfh: T.handle,
        kfh: T.handle,
        qh: T.handle,
        kh: T.handle,
        vh: T.handle,
        gh: T.handle,
        rows: T.int64,
    ):
        packed = T.match_buffer(ph, (rows * WID,), "bfloat16")
        rq = T.match_buffer(rqh, (rows * 8,), "float32")
        rk = T.match_buffer(rkh, (rows * 2,), "float32")
        qf = T.match_buffer(qfh, (256,), "float32")
        kf = T.match_buffer(kfh, (256,), "float32")
        q = T.match_buffer(qh, (rows * HD8,), "bfloat16")
        k = T.match_buffer(kh, (rows * HD2,), "bfloat16")
        v = T.match_buffer(vh, (rows * HD2,), "bfloat16")
        gate = T.match_buffer(gh, (rows * QWD,), "bfloat16")
        with T.Kernel(CORES) as core:
            qp_ub = T.alloc_shared((RD * WID,), "bfloat16")
            rq_ub = T.alloc_shared((RD * 8,), "float32")
            rk_ub = T.alloc_shared((RD * 2,), "float32")
            qf_ub = T.alloc_shared((256,), "float32")
            kf_ub = T.alloc_shared((256,), "float32")
            q_ub = T.alloc_shared((HD8 * RD,), "bfloat16")
            ko_ub = T.alloc_shared((HD2 * RD,), "bfloat16")
            vo_ub = T.alloc_shared((HD2 * RD,), "bfloat16")
            g_ub = T.alloc_shared((RD * QWD,), "bfloat16")
            T.annotate_buffer_versions({qp_ub: 2, rq_ub: 2, rk_ub: 2, qf_ub: 1, kf_ub: 1, q_ub: 1, ko_ub: 1, vo_ub: 1, g_ub: 1})
            T.copy(qf, qf_ub)
            T.copy(kf, kf_ub)
            for step in T.Pipelined(T.ceildiv(rows, CORES * RD), num_stages=2):
                r0 = (step * CORES + core) * RD
                if r0 < rows:
                    pbase = r0 * WID
                    T.copy(packed[pbase : pbase + RD * WID], qp_ub)
                    T.copy(rq[r0 * 8 : r0 * 8 + RD * 8], rq_ub)
                    T.copy(rk[r0 * 2 : r0 * 2 + RD * 2], rk_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for r in range(RD):
BODY
QCOPIES
KCOPIES
VCOPIES
                    T.copy(g_ub, gate[r0 * QWD : r0 * QWD + RD * QWD])
"""
    template = template.replace("WID", str(W)).replace("QWD", str(QW))
    template = template.replace("HD8", str(HEADS * D)).replace("HD2", str(KHEADS * D))
    template = template.replace("BODY", text)
    template = template.replace("QCOPIES", q_copies)
    template = template.replace("KCOPIES", k_copies)
    template = template.replace("VCOPIES", v_copies)
    template = template.replace("RD", str(R))
    template = chr(10).join(
        line[4:] if line.startswith("    ") else line for line in template.split(chr(10))
    )
    namespace = {"T": T, "CORES": CORES, "R": R}
    source_name = "<qkv-split-vec-bhsd>"
    import linecache
    linecache.cache[source_name] = (
        len(template), None,
        [line + chr(10) for line in template.split(chr(10))], source_name
    )
    exec(compile(template, source_name, "exec"), namespace)
    return namespace["main"]


COMPILE_OPTIONS = ()


def build_programs():
    yield "qkv_split_vec_bhsd_h256", qkv_split_vec_bhsd()
