// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/library.hpp"
#include "rhino_launch_program.h"
#include "catalog_metadata.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
thread_local char error_text[512]{};
void error(const char* text) noexcept {
  std::snprintf(error_text, sizeof(error_text), "%s", text);
}
class FileDescriptor {
 public:
  explicit FileDescriptor(int value) : value_(value) {}
  ~FileDescriptor() { if (value_ >= 0) ::close(value_); }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  int get() const { return value_; }
 private:
  int value_;
};
std::size_t find(const char* name) noexcept {
  if (name) for (std::size_t i = 0; i < xinfer::rpu::metadata::count; ++i)
    if (std::strcmp(name, xinfer::rpu::metadata::assets[i].name) == 0) return i;
  return xinfer::rpu::metadata::count;
}
}  // namespace

namespace xinfer::rpu {
std::uint32_t library_abi_version() noexcept { return metadata::abi_version; }
std::size_t kernel_count() noexcept { return metadata::count; }
const char* kernel_name(std::size_t index) noexcept {
  return index < metadata::count ? metadata::assets[index].name : nullptr;
}
const char* kernel_admission(std::size_t index) noexcept {
  return index < metadata::count ? metadata::assets[index].admission : nullptr;
}
const char* kernel_execution_status(std::size_t index) noexcept {
  return index < metadata::count ? metadata::assets[index].execution_status : nullptr;
}
bool has_kernel(const char* name) noexcept { return find(name) < metadata::count; }
const char* last_error() noexcept { return error_text; }
std::int32_t load_program(rhino_lkn::Program_t& program, const char* name,
                          bool allow_candidate) noexcept {
  error_text[0] = '\0';
  const auto index = find(name);
  if (index == metadata::count) { error("unknown XInfer RPU kernel"); return -1; }
  const auto& asset = metadata::assets[index];
  if (asset.execution_status[0] != '\0' ||
      (asset.admission[0] != '\0' && std::strcmp(asset.admission, "candidate") != 0)) {
    error("XInfer RPU kernel has unsupported admission metadata"); return -6;
  }
  if (std::strcmp(asset.admission, "candidate") == 0 && !allow_candidate) {
    error("XInfer RPU candidate requires explicit allow_candidate"); return -3;
  }
  try {
    const char* root = std::getenv("XINFER_RPU_ASSETS");
    if (!root || !*root) throw std::runtime_error("RPU execution requires XINFER_RPU_ASSETS");
    FileDescriptor directory(::open(root, O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (directory.get() < 0) throw std::runtime_error("cannot open XINFER_RPU_ASSETS directory");
    // Catalog filenames are basenames. Pinning the directory and opening an
    // asset without following symlinks prevents traversal and path replacement.
    FileDescriptor file(::openat(directory.get(), asset.filename,
                                 O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    struct stat state {};
    if (file.get() < 0 || ::fstat(file.get(), &state) != 0 ||
        !S_ISREG(state.st_mode) || state.st_size <= 0)
      throw std::runtime_error("RPU asset must be a nonempty regular file without symlinks");
    // Launch reads a path. Keep the verified inode pinned through its read,
    // without embedding bytes or writing a temporary copy of the asset.
    const auto path = std::string("/proc/self/fd/") + std::to_string(file.get());
    const auto rc = program.create_with_binary_file(path, name);
    if (rc) std::snprintf(error_text, sizeof(error_text), "Rhino Launch rejected %s: %d", name, rc);
    return rc;
  } catch (const std::exception& e) {
    error(e.what()); return -2;
  } catch (...) {
    error("unexpected error loading XInfer RPU kernel"); return -2;
  }
}
}  // namespace xinfer::rpu

extern "C" std::uint32_t xinfer_rpu_abi_version_v1() noexcept {
  return xinfer::rpu::library_abi_version();
}
extern "C" std::int32_t xinfer_rpu_load_program_v1(
    rhino_lkn::Program_t* program, const char* name, std::uint32_t allow_candidate) noexcept {
  if (!program) { error("null Program_t"); return -1; }
  if (allow_candidate > 1) { error("allow_candidate must be 0 or 1"); return -5; }
  return xinfer::rpu::load_program(*program, name, allow_candidate != 0);
}
extern "C" const char* xinfer_rpu_last_error_v1() noexcept {
  return xinfer::rpu::last_error();
}
