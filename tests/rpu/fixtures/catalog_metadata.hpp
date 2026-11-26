// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
// Authored admission metadata. Device payloads are created by the test caller.
namespace xinfer::rpu::metadata {
inline constexpr unsigned abi_version = 11;
struct Asset { const char* name; const char* filename; const char* admission; const char* execution_status; };
inline constexpr Asset assets[] = {
    {"audited", "audited.ref", "", ""},
    {"candidate", "candidate.ref", "candidate", ""},
    {"unknown_admission", "unknown_admission.ref", "future", ""},
    {"unknown_status", "unknown_status.ref", "", "future"},
};
inline constexpr std::size_t count = sizeof(assets) / sizeof(assets[0]);
}
