# Generated AA source

These are the original generated Ascend programs used by the validated AA
downsample kernels. Each file contains its device program and host launcher;
there is no embedded precompiled object or library. CMake compiles this source
directly with bisheng, then renames the launcher's original symbol prefix to the
in-tree ABI. No scheduling or numerical changes are applied.

| Source | SHA-256 |
|---|---|
| `aa_v_720_1280_128_224.asc` | `03133b01d8eab61a975a712d58adb9461ff67f3ae5a797eb94c10b01e8629512` |
| `aa_h_720_1280_128_224.asc` | `690a3ca8545986cbb7b3f45c7f0ec91aa063f58bd1e91219e405be20373e4c49` |
| `aa_v_480_640_160_224.asc` | `981e275bdd52a96429a6ca253944e7ff88c3c64e143809de27c1fdbfc3f3256e` |
| `aa_h_480_640_160_224.asc` | `1daad93ec5a77db5ad6d69efa352d25dcaa0bcb93ced471bd9f86293a955cff1` |

The six transitive template headers are included under `third_party/tilelang`
from TileLang `0.1.15+cpu.git1e8448fc`, with their original license. Standard
C/C++ and Ascend SDK headers remain compiler-toolchain dependencies.
