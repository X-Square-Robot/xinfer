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

"""Add-RMSNorm plans using double-buffered stores and direct FP32 weights
for the 1280-wide vision/merger and 2048-wide prefill streams. BF16 rounding
of the residual sum precedes the FP32 sum of squares."""
import tilelang.ascend.language as T  # noqa: E402

from . import rms_consistent as _orig  # noqa: E402  (the vendored baselines)
from . import buffered_rmsnorm as _buffered_rmsnorm  # noqa: E402


def build_programs():
    # references (release arithmetic, verbatim)
    yield "add_rmsnorm_buffered_h1280_baseline", _buffered_rmsnorm.buffered_ub_rmsnorm()
    yield "add_rmsnorm_buffered_h2048_baseline", _orig.ubv(2048)
    # Overlap the residual and normalized output stores with the next row tile.
    yield "add_rmsnorm_h1280_double_buffer_fp32_weight", _buffered_rmsnorm.buffered_ub_rmsnorm(buf2=True, weight_mode="f32direct")
    yield "add_rmsnorm_h2048_double_buffer", _buffered_rmsnorm.buffered_ub_rmsnorm(width=2048, buf2=True)
    yield "add_rmsnorm_h2048_double_buffer_fp32_weight", _buffered_rmsnorm.buffered_ub_rmsnorm(width=2048, buf2=True, weight_mode="f32direct")
