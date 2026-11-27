// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Board-free descriptors only. Does not execute SDK or emulate device math.
#include "common/rpu/tp_contract.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace tp=xinfer::rpu::tp;
namespace abi=xinfer::rpu::abi;
unsigned checks=0;
void check(bool v){++checks;if(!v)throw std::runtime_error("contract check failed");}
template<class F>void rejects(F f){++checks;try{f();}catch(const std::invalid_argument&){return;}throw std::runtime_error("expected rejection");}
struct Fixture{
 tp::Request r;uint64_t at=0x10000,identity=1;
 void add(const std::string& n,std::vector<int64_t> shape,std::string dist){
  tp::Tensor t;t.shape=std::move(shape);t.dtype="float16";t.storage="spm";t.layout="row_major";
  t.distribution=dist;t.shard_axis=dist=="sharded"?1:-1;t.identity=identity++;
  t.bytes=tp::elements(t)*2;t.address=at;at+=(t.bytes+255)&~uint64_t(255);
  t.values_known=true;t.finite=true;t.minimum=0;t.maximum=0;r.tensors.emplace(n,t);
 }
 explicit Fixture(std::string op,unsigned m){
  r.op=std::move(op);bool zba=r.op.find("zba")!=std::string::npos,gate=r.op.find("gateup")!=std::string::npos;
  add("src",{m,2048},"replicated");
  if(gate){add("out_gate",{m,768},"sharded");add("out_up",{m,768},"sharded");}
  else if(zba){for(auto n:{"out_q","out_k","out_v","out_z"})add(n,{m,256},"sharded");add("out_b",{m,16},"sharded");add("out_a",{m,16},"sharded");}
  else{add("out_q",{m,256},"sharded");add("out_qgate",{m,256},"sharded");add("out_k",{m,256},"local");add("out_v",{m,256},"local");}
  tp::Tensor w;w.shape={8,gate?1536:zba?1056:1024,2048};w.dtype="float16";w.storage="ddr";w.layout="tp8_nk16";
  w.distribution="shared";w.shard_axis=-1;w.identity=identity++;w.bytes=tp::elements(w)*2;w.address=0x100000000ULL;w.immutable=true;
  r.tensors.emplace("weight",w);
 }
 tp::Lowered lower(){tp::Lowered d;check(tp::lower_operator(r,d));return d;}
};
int main(){
 try{
  // Catalog names: stem_base + variant; m512/m448_k128 variants follow each
  // family. Generic stems (no size suffix) dispatch via tp_family resolver.
  struct Stem {
    const char* full;     // catalog name for full-precision un-shaped entry
    const char* m512;     // _m512 catalog name
    const char* m512k128; // _m512_k128 catalog name
    const char* m448k128; // _m448_k128 catalog name
    const char* gateup_key; // matches resolver dispatch signal
  };
  const std::array<Stem,3> stems{
    Stem{"qkv_zba_n1056k2048", "qkv_zba_m512", "qkv_zba_m512_k128", "qkv_zba_m448_k128", "qkv_zba"},
    Stem{"qkv_qgate_n1024k2048", "qkv_qgate_m512", "qkv_qgate_m512_k128", "qkv_qgate_m448_k128", "qkv"},
    Stem{"gateup_n1536k2048", "gateup_m512", "gateup_m512_k128", "gateup_m448_k128", "gateup"},
  };
  auto full_name = [](const Stem& s) { return std::string(s.full); };
  auto m512_name = [](const Stem& s) { return std::string(s.m512); };
  auto mKX_k128_name = [](const Stem& s, unsigned m) {
    return std::string(m == 512 ? s.m512k128 : s.m448k128);
  };
  for(unsigned m:{448U,512U})for(const auto&stem:stems){
   const auto name=mKX_k128_name(stem,m);
   Fixture f(name,m);auto d=f.lower();const auto shape=abi::streamed_gemm_shape(d.plan.entry);
   check(std::string(abi::name(d.plan.entry))==name);check(shape.k==2048&&shape.tile_n==64);
   check(d.plan.grid[0]==(shape.columns+63)/64&&d.plan.grid[1]==1&&d.plan.grid[2]==1&&d.plan.cores==8);
   check(d.plan.regs[4+2*shape.output_count]==m);check(d.plan.register_count==5+2*shape.output_count);
   check(d.plan.regs[2]==0&&d.plan.regs[3]==0x100); // DDR address >>8 =0x01000000
   check(d.writes.size()==shape.output_count);
   check(f.at<7*1024*1024);check(d.zero_counters.empty());
   for(unsigned bad:{0U,1U,64U,128U,192U,256U,320U,384U,447U,449U,513U,m==448?512U:448U})
    rejects([&]{Fixture x(name,bad);x.lower();});
   rejects([&]{Fixture x(name,m);x.r.tensors.at("weight").layout="row_major";x.lower();});
   rejects([&]{Fixture x(name,m);x.r.tensors.at("weight").immutable=false;x.lower();});
   rejects([&]{Fixture x(name,m);x.r.tensors.at("weight").address+=16;x.lower();});
   rejects([&]{Fixture x(name,m);x.r.tensors.at("src").distribution="partial";x.lower();});
   const auto port=std::string(stem.gateup_key)=="gateup"?"out_gate":"out_q";
   rejects([&]{Fixture x(name,m);x.r.tensors.at(port).address=x.r.tensors.at("src").address;x.lower();});
   rejects([&]{Fixture x(name,m);x.r.tensors.at(port).immutable=true;x.lower();});
   rejects([&]{Fixture x(name,m);x.r.tensors.at(port).shape[1]-=16;x.lower();});
   rejects([&]{Fixture x(name,m);x.r.integers["valid_rows"]=447;x.lower();});
  }
  // Default projection dispatch must have exactly the same native packet, writes and
  // state semantics as the qualified explicit K128 entry. Direct ABI and
  // explicit legacy M512 retain K64 for diagnostic/compatibility callers.
  for(unsigned m:{448U,512U})for(const auto&stem:stems){
   Fixture generic(full_name(stem),m), explicit_new(mKX_k128_name(stem,m),m);
   auto a=generic.lower(), b=explicit_new.lower();
   check(a.plan.entry==b.plan.entry && a.plan.regs==b.plan.regs &&
         a.plan.grid==b.plan.grid && a.plan.cores==b.plan.cores &&
         a.plan.register_count==b.plan.register_count && a.writes==b.writes);
   check(a.state_accesses.empty() && b.state_accesses.empty() &&
         !a.collective && !b.collective && a.zero_counters==b.zero_counters);
   Fixture legacy(m512_name(stem),m);auto c=legacy.lower();
   check(std::string(abi::name(c.plan.entry))==m512_name(stem));
   std::vector<std::string> ports = std::string(stem.gateup_key)=="gateup" ? std::vector<std::string>{"out_gate","out_up"} :
     std::string(stem.gateup_key)=="qkv_zba" ? std::vector<std::string>{"out_q","out_k","out_v","out_z","out_b","out_a"} :
       std::vector<std::string>{"out_q","out_qgate","out_k","out_v"};
   std::array<uint32_t,6> dst{};for(unsigned i=0;i<ports.size();++i)dst[i]=generic.r.tensors.at(ports[i]).address;
   auto base=std::string(stem.gateup_key)=="gateup"?abi::Entry::GateUp:std::string(stem.gateup_key)=="qkv_zba"?abi::Entry::QkvZba:abi::Entry::Qkv;
   auto direct=abi::streamed_gemm({base,dst,uint32_t(generic.r.tensors.at("src").address),m,generic.r.tensors.at("weight").address});
   check(direct.entry==c.plan.entry && direct.regs==c.plan.regs && direct.grid==c.plan.grid);
  }
  // Small generic dispatches and explicit low-M *_m512 remain unchanged.
  for(unsigned m:{64U,128U,192U,256U,320U,384U,448U,512U})for(const auto&stem:stems){
   Fixture f(full_name(stem),m);auto d=f.lower();
   check(std::string(abi::name(d.plan.entry))==(m>384?mKX_k128_name(stem,m):full_name(stem)));
  }
  for(unsigned m:{128U,192U,256U,320U,384U})for(const auto&stem:stems){
   auto name=m512_name(stem);Fixture f(name,m);auto d=f.lower();check(std::string(abi::name(d.plan.entry))==name);
  }
  std::cout<<"PASS "<<checks<<" host descriptor checks; no SDK context or device math\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL after "<<checks<<": "<<e.what()<<"\n";return 1;}
}
