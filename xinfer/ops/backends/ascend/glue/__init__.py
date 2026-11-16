"""The unified Ascend kernel glue (feats/ascend-kernels-unified).

Python dispatch over the in-tree operator sources (``../kernels``) and the
vendor CANN compositions the models' routed>0 surface needs. Library loading
is exclusively ``.._provider`` (pure ctypes); nothing here imports or loads
the former paired kernel package — that architecture lives on in the old fastpaths branch.

Module map (the 25-row ARCH-DESIGN table, landed):
``cann_projection/cann_action/cann_attention/cann_gated_delta_net`` — the
CANN bridge quartet over the core library's static bridges;
``_cann_primitives/_cann_action_scope/_capture`` — bridge support code;
``rmsnorm/adarmsnorm/double_adarmsnorm/gated_residual_adarmsnorm/
denoise_output`` — class-style entries behind ``core``;
``soup2/soupform/soupkill/soupunlock/finishbf16/simdepi_r1/rmsfuse_r4/
decodeattn_r4/ubfinish_r5/g2112_tilelang/smallm_r3/move_r2/conv_gemm/
glu_epi/aa_resample/gdn_vendor/copykill_r2`` — the knife mounts (one flag,
one library, one stats receipt each);
``qkv_split_vec/qkv_split_bhsd/rope_d64/rope_d64_bhsd`` — the GQA-GLUE
kernel mounts over the core library; ``core`` — the provider namespace the
binding layers probe.
"""

from xinfer.ops.backends.ascend.glue import core  # noqa: F401
