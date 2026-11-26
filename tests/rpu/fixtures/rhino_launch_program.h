// The rhino_lkn namespace and API signatures correspond to the external
// proprietary Rhino Launch SDK provided by HUIXI-AI. This minimal in-memory
// test double supports host-only tests. SDK licensing is governed by the
// terms provided by HUIXI-AI.
// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

// A deliberately small host-side test double, authored for xinfer tests.
// It is not an SDK header and does not model any board or queue behavior.
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>

namespace rhino_lkn {
class Program_t {
 public:
  int calls = 0;
  int status = 0;
  bool throw_on_load = false;
  std::string path;
  std::string name;
  std::string bytes;
  std::function<void()> before_read;

  int create_with_binary_file(const std::string& asset_path, const char* entry) {
    ++calls;
    if (throw_on_load) throw std::runtime_error("test loader failure");
    path = asset_path;
    name = entry;
    if (before_read) before_read();
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("external asset cannot be read");
    bytes.assign(std::istreambuf_iterator<char>(stream), {});
    return status;
  }
};
}  // namespace rhino_lkn
