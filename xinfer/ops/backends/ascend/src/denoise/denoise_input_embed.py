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

"""Small FP32 projection with the source's ordered FMA and BF16 store."""

import tilelang.ascend.language as T

from gemm.silu_and_mul import _store_rne


def input_embed():
    @T.prim_func
    def main(
        oh: T.handle, xh: T.handle, eh: T.handle, wh: T.handle, hh: T.handle,
        mh: T.handle, a0h: T.handle, a1h: T.handle, a2h: T.handle, vh: T.handle,
        vmh: T.handle, rows: T.int64, width: T.int64, columns: T.int64,
        heads: T.int64, flags: T.int64, addends: T.int64,
    ):
        inner = width + width * (flags & 1)
        count = rows * columns
        out = T.match_buffer(oh, (count,), "bfloat16")
        x = T.match_buffer(xh, (rows, width), "float32")
        extra = T.match_buffer(eh, (rows, width), "float32")
        weight = T.match_buffer(wh, (heads * columns, inner), "float32")
        head_index = T.match_buffer(hh, (rows,), "int32")
        input_mask = T.match_buffer(mh, (rows,), "uint8")
        add0 = T.match_buffer(a0h, (count,), "float32")
        add1 = T.match_buffer(a1h, (count,), "float32")
        add2 = T.match_buffer(a2h, (count,), "float32")
        override = T.match_buffer(vh, (count,), "float32")
        override_mask = T.match_buffer(vmh, (rows,), "uint8")
        blocks = T.min(T.ceildiv(count, 256), T.int64(128))

        with T.Kernel(blocks) as block:
            with T.SimtVF(threads=256):
                for step in T.serial(T.ceildiv(count, blocks * 256)):
                    for lane in T.Parallel(256):
                        index = (step * blocks + T.cast(block, "int64")) * 256 + lane
                        if index < count:
                            row = index // columns
                            col = index % columns
                            use_override = T.alloc_var("bool", init=False)
                            if (flags & 8) != 0:
                                use_override = override_mask[row] != 0
                            value = T.alloc_var("float32", init=0)
                            if use_override:
                                value = override[index]
                            else:
                                active = T.alloc_var("bool", init=True)
                                if (flags & 4) != 0:
                                    active = input_mask[row] != 0
                                head = T.alloc_var("int64", init=0)
                                if (flags & 2) != 0:
                                    head = T.cast(head_index[row], "int64")
                                # The source requires in-range head IDs. Guard the
                                # device load even for invalid caller data.
                                if active and head >= 0 and head < heads:
                                    feature = head * columns + col
                                    for k in T.serial(width):
                                        value = T.call_extern(
                                            "float32", "fmaf", x[row, k],
                                            weight[feature, k], value,
                                        )
                                    if (flags & 1) != 0:
                                        for k in T.serial(width):
                                            value = T.call_extern(
                                                "float32", "fmaf", extra[row, k],
                                                weight[feature, width + k], value,
                                            )
                                if addends >= 1:
                                    value = value + add0[index]
                                if addends >= 2:
                                    value = value + add1[index]
                                if addends >= 3:
                                    value = value + add2[index]
                            out[index] = _store_rne(value, "bfloat16")

    return main
