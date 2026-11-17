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

"""SIMD-EPILOGUE R1 kernels: buf2/f32direct scheduling variants of the
released ubv plans (h1280 vision/merger + h2048 prefill), for the
fused-after route (GEMM + back-to-back SIMD add_rms in one stream).

All arithmetic statements are the released ubv plan (imported through the
r5 generator, itself verbatim from the walloss lib source); the variants
change only scheduling/staging:
  buf2   -- sum_ub/out_ub buffer versions 1 -> 2 (MTE3 store of step N
            overlaps vector work of step N+1); zero arithmetic touched.
  f32direct -- FP32 weight row staged once (identical f32 values to the
            in-kernel UNPK upcast of the BF16 staging); kills the per-call
            fold16 scalar unpack prologue.
add_residual=1, write_output=1, no stats, not centered -- exactly the
inference add_rms_norm call.
"""
import tilelang.ascend.language as T  # noqa: E402

from . import rms_consistent as _orig  # noqa: E402  (the vendored baselines)
from . import rmsfuse_r5 as _r5  # noqa: E402


def build_programs():
    # references (release arithmetic, verbatim)
    yield "se1_ubv1280_base", _r5.ubv5()
    yield "se1_ubv2048_orig", _orig.ubv(2048)
    # fused-after knives
    yield "se1_ubv1280_buf2f", _r5.ubv5(buf2=True, weight_mode="f32direct")
    yield "se1_ubv2048_buf2", _r5.ubv5(width=2048, buf2=True)
    yield "se1_ubv2048_buf2f", _r5.ubv5(width=2048, buf2=True, weight_mode="f32direct")
