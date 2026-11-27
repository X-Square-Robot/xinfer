// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/tp_memory.hpp"
#include <iostream>

namespace {
using namespace xinfer::rpu::tp;
unsigned checks = 0;
void expect(bool ok) { if (!ok) throw std::runtime_error("memory contract assertion failed"); ++checks; }
template<class F> void rejects(F&& f) {
  try { f(); } catch (const std::invalid_argument&) { ++checks; return; }
  throw std::runtime_error("memory contract accepted an invalid descriptor");
}
Tensor tensor(std::vector<std::int64_t> dims, const std::string& storage,
              const std::string& role, std::uint64_t id) {
  Tensor t;
  t.shape = std::move(dims); t.storage = storage; t.distribution = role;
  t.identity = id; t.address = storage == "spm" ? id * 65536 : 0x10000000 + id * 65536;
  t.bytes = elements(t) * 2;
  return t;
}
void copies() {
  auto src = tensor({16,16}, "spm", "local", 1);
  auto dst = tensor({16,16}, "spm", "local", 2);
  auto run = [](const Tensor& x, const Tensor& y, int mode) {
    Request r; r.op = "copy_f16"; r.tensors = {{"input", x}, {"out", y}};
    r.integers["mode"] = mode; Lowered out;
    expect(lower_copy(r, out)); expect(out.transfer_mode == mode);
    expect(out.writes == std::vector<std::string>{"out"});
  };
  run(src, dst, 0);
  auto ddr = tensor({8,16,16}, "ddr", "shared", 3);
  run(src, ddr, 0); run(ddr, dst, 0);
  auto shared = tensor({16,16}, "ddr", "shared", 4);
  auto replicated = dst; replicated.distribution = "replicated";
  run(shared, replicated, 1); run(replicated, shared, 2);
  auto single = dst; single.distribution = "single";
  run(shared, single, 2); run(single, shared, 2);
  auto other_ddr = tensor({16,16}, "ddr", "shared", 5);
  run(shared, other_ddr, 3);
  rejects([&] { run(ddr, replicated, 0); });
  rejects([&] { run(shared, dst, 1); });
  rejects([&] { run(src, ddr, 2); });
  rejects([&] { run(single, dst, 0); });
  rejects([&] { run(src, dst, 3); });
  rejects([&] { run(src, dst, 4); });
  auto overlap = src; overlap.address += 8;
  rejects([&] { run(src, overlap, 0); });
  auto immutable = dst; immutable.immutable = true;
  rejects([&] { run(src, immutable, 0); });
  auto wrong_shape = dst; wrong_shape.shape = {16,8}; wrong_shape.bytes /= 2;
  rejects([&] { run(src, wrong_shape, 0); });
  auto unaligned = dst; ++unaligned.address;
  rejects([&] { run(src, unaligned, 0); });
}
void views() {
  auto arena = tensor({1024}, "spm", "local", 1);
  auto view = contiguous_view(arena, {16,16}, 256, "replicated", -1);
  expect(view.address == arena.address + 256 && view.bytes == 512);
  expect(view.identity == arena.identity && !view.immutable);
  auto shard = contiguous_view(arena, {16,16}, 0, "sharded", 1);
  expect(shard.shard_axis == 1 && shard.distribution == "sharded");
  rejects([&] { contiguous_view(shard, {16,16}, 0, "replicated", -1); });
  rejects([&] { contiguous_view(arena, {16}, 0, "single", -1); });
  rejects([&] { contiguous_view(arena, {1024}, 8, "local", -1); });
  rejects([&] { contiguous_view(arena, {16}, -8, "local", -1); });
  rejects([&] { contiguous_view(arena, {16}, 2, "local", -1); });
  rejects([&] { contiguous_view(arena, {1}, 0, "local", -1); });
  rejects([&] { contiguous_view(arena, {16}, 0, "sharded", 1); });
  auto table = tensor({8,128}, "ddr", "shared", 2); table.immutable = true;
  expect(contiguous_view(table, {128}, 256, "shared", -1).immutable);
  rejects([&] { contiguous_view(table, {16}, 0, "local", -1); });
  table.layout = "tp8_nk16";
  rejects([&] { contiguous_view(table, {16}, 0, "shared", -1); });
  arena.state_kind = "history";
  rejects([&] { contiguous_view(arena, {16}, 0, "local", -1); });
}
}  // namespace
int main() {
  try { copies(); views(); std::cout << checks << " memory contract checks passed\n"; }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
