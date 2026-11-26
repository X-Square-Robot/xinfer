// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// External-asset loader contracts using authored Launch API doubles.
#include "common/rpu/library.hpp"
#include "rhino_launch_program.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>

namespace rpu = xinfer::rpu;
namespace fs = std::filesystem;
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
struct Directory {
  fs::path path;
  Directory() {
    auto pattern = (fs::temp_directory_path() / "xinfer-rpu-assets-test-XXXXXX").string();
    auto* created = ::mkdtemp(pattern.data());
    if (!created) throw std::runtime_error("cannot create test asset directory");
    path = created;
  }
  ~Directory() { std::error_code ignored; fs::remove_all(path, ignored); }
};
void write(const fs::path& path, const std::string& data) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(data.data(), static_cast<std::streamsize>(data.size()));
  if (!stream) throw std::runtime_error("cannot write test asset");
}
int main() try {
  check(rpu::library_abi_version() == 11 && xinfer_rpu_abi_version_v1() == 11, "ABI mismatch");
  check(rpu::kernel_count() == 4 && rpu::has_kernel("candidate"), "catalog metadata lost");
  check(std::string(rpu::kernel_admission(1)) == "candidate" && !rpu::kernel_name(4),
        "catalog admission or enumeration changed");
  rhino_lkn::Program_t program;
  const char* invalid_names[] = {nullptr, "unknown", "../audited", "/audited"};
  for (const char* name : invalid_names)
    check(rpu::load_program(program, name) == -1, "unknown/traversal name reached SDK");
  check(program.calls == 0, "unknown entry reached SDK");
  check(rpu::load_program(program, "candidate") == -3, "candidate lacks explicit admission");
  for (const char* name : {"unknown_admission", "unknown_status"})
    check(rpu::load_program(program, name, true) == -6, "unknown metadata accepted");
  check(xinfer_rpu_load_program_v1(nullptr, "audited", 0) == -1, "null program accepted");
  check(xinfer_rpu_load_program_v1(&program, "audited", 2) == -5, "invalid admission flag accepted");
  ::unsetenv("XINFER_RPU_ASSETS");
  check(rpu::load_program(program, "audited") == -2 && program.calls == 0, "missing assets root accepted");
  Directory directory;
  ::setenv("XINFER_RPU_ASSETS", directory.path.c_str(), 1);
  const auto asset = directory.path / "audited.ref";
  check(rpu::load_program(program, "audited") == -2, "missing asset accepted");
  write(asset, "");
  check(rpu::load_program(program, "audited") == -2, "empty asset accepted");
  fs::remove(asset); fs::create_directory(asset);
  check(rpu::load_program(program, "audited") == -2, "directory accepted as asset");
  fs::remove(asset);
  const auto outside = directory.path / "unlisted.ref";
  write(outside, "unlisted payload");
  fs::create_symlink(outside, asset);
  check(rpu::load_program(program, "audited") == -2, "symlink asset accepted");
  fs::remove(asset);
  check(::mkfifo(asset.c_str(), 0600) == 0, "cannot create test FIFO");
  check(rpu::load_program(program, "audited") == -2, "nonregular asset accepted");
  fs::remove(asset);
  const std::string payload("test\0asset", 10);
  write(asset, payload);
  write(directory.path / "candidate.ref", payload);
  check(rpu::load_program(program, "audited") == 0 && program.calls == 1, "valid asset not loaded");
  check(program.bytes == payload && program.name == "audited", "asset bytes or entry changed");
  check(program.path.rfind("/proc/self/fd/", 0) == 0, "loader did not keep validated inode pinned");
  check(std::string(rpu::last_error()).empty(), "successful load retained stale error");
  check(xinfer_rpu_load_program_v1(&program, "candidate", 1) == 0 && program.calls == 2,
        "explicit candidate admission ignored");
  program.before_read = [&] {
    fs::rename(asset, directory.path / "saved.ref");
    fs::create_symlink(outside, asset);
  };
  check(rpu::load_program(program, "audited") == 0 && program.bytes == payload,
        "concurrent path replacement changed the validated asset");
  program.before_read = {};
  fs::remove(asset); fs::rename(directory.path / "saved.ref", asset);
  program.status = 73;
  check(rpu::load_program(program, "audited") == 73 &&
            std::string(rpu::last_error()).find("73") != std::string::npos, "SDK failure lost");
  program.status = 0; program.throw_on_load = true;
  check(rpu::load_program(program, "audited") == -2 &&
            std::string(rpu::last_error()).find("test loader failure") != std::string::npos,
        "SDK exception escaped or lost diagnostic");
  check(rpu::load_program(program, "candidate") == -3, "candidate gate changed");
  const std::string main_error = rpu::last_error();
  std::string worker_error;
  std::thread worker([&] {
    rhino_lkn::Program_t other;
    check(rpu::load_program(other, "unknown") == -1, "worker accepted unknown entry");
    worker_error = rpu::last_error();
  });
  worker.join();
  check(!worker_error.empty() && worker_error != main_error && rpu::last_error() == main_error,
        "diagnostic state crossed threads");
  std::cout << "PASS: external allowlist, admission, regular files, pinned reads and SDK failures\n";
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n'; return 1;
}
