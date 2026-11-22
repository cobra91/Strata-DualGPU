// Based on Project Maya (https://github.com/mw00/project-maya)
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Licensed under the MIT License.

// src/ik2/ik2_engine.cu - strata-ik2: Inkling 2 (thinkingmachines/Inkling-2-Small, 594B / ~17B active) on one or two
// GPUs, from an `inkling2` GGUF in llama.cpp PR #32187's layout (docs/INKLING2.md; unsloth/Inkling-2-Small-GGUF's
// files: NVFP4, IQ3_XXS / IQ4_XS and Q*_K experts).
//
//   * The dense weights (attention, router, head) decoded once to FP16 in VRAM (the router in F32); cuBLAS' FP16 GEMMs
//     with F32 sums.  The attention (relative bias, sinks, sliding window, L2 clips) fused: an mma.sync kernel, and a
//     wgmma one on Hopper (sm_90a); the KV cache in int8 with a scale per 64 (IK2_KV=f16, fp8: E4M3).
//   * The routed experts as the GGUF holds them, all in pinned host memory (~323 GiB for UD-Q4_K_XL), and a cache of
//     them in VRAM (--cache-gb, at most what the GPU has room for).  A prompt goes through layer by layer in passes of
//     up to 16384 tokens, so each pass streams every expert it misses once, the next layer's prefetched while this one
//     runs; a scan-resistant eviction keeps a pass from flushing the cache.  IK2_WARM fills the cache at load with the
//     experts tokens pick most (a profile written with IK2_POP_OUT).
//   * Decode a token at a time; with IK2_CPU=1 a missing expert is computed on the CPU (AVX-512) when that is sooner
//     than its copy.  Two GPUs (--gpus 2): IK2_EP=1 deals each layer's experts to both (expert parallel - each with its
//     own cache and PCIe link), else the layers are split between them.
//
//   strata-ik2 --model FIRST-SHARD.gguf --serve [--max-context 262144] [--cache-gb 60] [--gpus 1] [--threads 8]
//              [--eos a,b]
//   strata-ik2 --model FIRST-SHARD.gguf --tokens 1,2,3|@FILE [--gen N] [--chunk 2048] [--dump DIR]
// --serve: the engine protocol on stdin / stdout (serve/server.py's StrataEngine: INFO / READY, GEN, PP, T, DONE,
// STOP); --serve-ik2 the older @-line one.  --dump: each layer's output (blk.L.out) and the log-probabilities of every
// prompt position, F32 raw files ([T][C], [T][n_vocab]) - what tools/ik2_compare.py holds against ref/inkling2.py's
// seams.  Settings (IK2_*): docs/SETTINGS.md.
#include "ik2_cpu.hpp"
#include "ik2_draft.hpp"
#include "ik2_kernels.cuh"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/ik2/engine.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#ifdef O_RDONLY
#undef O_RDONLY
#endif
#define O_RDONLY (_O_RDONLY | _O_BINARY)
#define open _open
#define close _close
typedef SSIZE_T ssize_t;
#define off_t int64_t
static inline ssize_t pread(int fd, void* buf, size_t count, int64_t offset) {
    if (_lseeki64(fd, offset, SEEK_SET) == -1) return -1;
    return _read(fd, buf, (unsigned int)count);
}
#define posix_fadvise(fd, off, len, adv) ((void)0)
#define MADV_WILLNEED 0
static inline int madvise(void*, size_t, int) { return 0; }
static inline int munmap(void*, size_t) { return 0; }
#ifndef SIGUSR1
#define SIGUSR1 30
#endif
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
static inline cudaError_t cudaMemPrefetchAsync_compat(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream = 0) {
    cudaMemLocation loc{};
    if (dstDevice < 0) {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    } else {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = dstDevice;
    }
    return cudaMemPrefetchAsync(devPtr, count, loc, 0, stream);
}
#define cudaMemPrefetchAsync cudaMemPrefetchAsync_compat

static inline cudaError_t cudaMemAdvise_compat(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) {
    cudaMemLocation loc{};
    if (device < 0) {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    } else {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = device;
    }
    return cudaMemAdvise(devPtr, count, advice, loc);
}
#define cudaMemAdvise cudaMemAdvise_compat
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <fstream>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <csignal>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

namespace k = ik2::k;

#define CU(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "strata-ik2: %s: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)
#define CB(x)                                                                                              \
    do {                                                                                                   \
        cublasStatus_t e_ = (x);                                                                           \
        if (e_ != CUBLAS_STATUS_SUCCESS) {                                                                 \
            std::fprintf(stderr, "strata-ik2: %s: cuBLAS status %d (line %d)\n", #x, (int) e_, __LINE__);    \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "strata-ik2: %s\n", m.c_str());
    std::exit(1);
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- the files

struct Shards {
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::vector<std::string> paths;
    std::map<std::string, std::pair<int, const strata::TensorInfo*>> by_name;

    explicit Shards(const std::string& first) {
        std::vector<std::string> ps{first};
        const size_t at = first.find("-00001-of-");
        if (at != std::string::npos) {
            const int n = std::atoi(first.c_str() + at + 10);
            for (int i = 2; i <= n; ++i) {
                char num[16];
                std::snprintf(num, sizeof num, "%05d", i);
                std::string p = first;
                p.replace(at + 1, 5, num);
                ps.push_back(p);
            }
        }
        for (const auto& p : ps) {
            files.emplace_back(new strata::GgufFile(p));
            paths.push_back(p);
            for (const auto& t : files.back()->tensors()) by_name[t.name] = {(int) files.size() - 1, &t};
        }
    }
    const strata::TensorInfo* info(const std::string& n) const {
        auto it = by_name.find(n);
        return it == by_name.end() ? nullptr : it->second.second;
    }
    const uint8_t* data(const std::string& n) const {
        auto it = by_name.find(n);
        if (it == by_name.end()) die("no tensor " + n);
        return files[it->second.first]->tensor_data(*it->second.second);
    }
    uint64_t file_offset(const std::string& n) const {
        auto it = by_name.find(n);
        return files[it->second.first]->data_start() + it->second.second->offset;
    }
    int file_of(const std::string& n) const { return by_name.at(n).first; }
    double num(const std::string& key) const {
        const auto* v = files[0]->get(key);
        if (!v) die("no key " + key);
        return v->num();
    }
    const strata::MetaValue* get(const std::string& key) const { return files[0]->get(key); }
};

size_t tensor_bytes(const strata::TensorInfo& t) {
    // (the blocks decoded here first: strata's block_geometry names NVFP4 but has no geometry for it)
    if (k::block_values((int) t.type)) return (size_t) (t.elements() / k::block_values((int) t.type)) * k::block_bytes((int) t.type);
    int e = 0, b = 0;
    if (!strata::block_geometry(t.type, e, b)) die(std::string("type ") + t.type_name() + " of " + t.name);
    return (size_t) (t.elements() / e) * b;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- the model

struct Config {
    int n_layer, C, H, KV, D, E, K, F, window, d_rel, extent, extent_swa, block, n_vocab, n_rows;
    // A looped model (Inkling-2-Mini): the layers core0 .. core1 - 1 (whole blocks) run n_pass times a token, so a
    // token is n_visit layer visits (n_layer without a loop).  rank: the rank of the loop's message gates.
    int n_visit = 0, n_pass = 1, core0 = 0, core1 = 0, rank = 0;
    float eps, route_scale, qk_clip, rel_clip, alpha, n_floor, gate_cap, up_cap, logit_cap;
    std::vector<bool> sliding;   // by layer
};

// a dense matrix as the GGUF holds it, in VRAM: its blocks [N][K]
struct Mat {
    // (IK2_DENSE=q8 with Q8_0 blocks and K a multiple of 128: the blocks taken apart - the int8 values [N][K] and the
    //  blocks' FP16 scales [N][K / 32] - which a warp reads 128 values an instruction; raw is then null)
    int8_t* q8 = nullptr;
    uint16_t* q8d = nullptr;
    uint8_t* raw = nullptr;     // IK2_DENSE=q8: the blocks, one token's products straight over them
    uint16_t* f16 = nullptr;    // else (the default): an FP16 copy, every product a cuBLAS GEMM - measured faster on an
    int type = 0, N = 0, K = 0; // RTX PRO 6000 (16 against 21 ms a token), for 4 GiB more VRAM
};

struct Layer {
    int dev = 0;   // the GPU it runs on (--gpus: the layers in equal runs)
    bool sliding;
    // FP16 [out][in] (fused projections on !g_dense_q8, or when quant types match on g_dense_q8)
    Mat wq, wk, wv, wr, wgate, wo;
    Mat wqkvrg, wqrg, wk_dyn, wv_dyn;
    uint16_t *sink16 = nullptr;                    // F16 [H][C] (used in g_dense_q8; fused in !g_dense_q8)
    uint16_t *router16 = nullptr;                  // F16 [E][C]
    float *probs_b;                                // F32 [E]
    float *attn_norm, *q_in_norm, *k_in_norm, *v_in_norm, *ffn_norm;
    float *q_clip, *r_clip, *rel_proj;             // [H][D], [H][d_rel], [extent][d_rel]
    float *res_score, *ffn_res_score;              // [nq][C], [C]
    int n_res_q;
    float *sc_k, *sc_v, *sc_attn, *sc_mlp;         // [C'][4]
    uint16_t *sc_k_dyn16 = nullptr, *sc_v_dyn16 = nullptr;   // F16 [KV * 4][C] (used in g_dense_q8; fused in !g_dense_q8)
    // the experts: types, bytes an expert, host blocks, scales (host)
    int ty[3];
    size_t eb[3];            // bytes of a role's part of an expert
    size_t stride[3];        // bytes between two experts' parts in host[r] (eb, or 2 eb in a fused gate_up, or slot_bytes)
    const uint8_t* host[3];
    bool host_contig = false;
    size_t slot_bytes = 0;
    std::vector<float> scale[3];                   // empty: 1
    // a looped core's layer: its later passes' message gates by module (0 q's, 1 k's, 2 v's, 3 the MoE's) -
    // down [rank][2 C], forget and gain [rank][C] - and the four messages' queries [4][C] in that order (F32)
    uint16_t *lp_down[4] = {}, *lp_forget[4] = {}, *lp_gain[4] = {};   // (FP16)
    float* lp_query = nullptr;
    // the resident token path (every expert in VRAM): where each expert's blocks are, and the experts' scales, on the
    // device ([E] each; a scale's null: 1)
    const uint8_t** tab_blob = nullptr;
    float* tab_sc[3] = {};
    std::vector<const uint8_t*> tab_h;             // (tab_blob on the host: a short prompt's groups)
    // multi-GPU EP resident tables: local expert pointers on device d (null if owned by peer)
    const uint8_t** ep_tab_blob[2] = {nullptr, nullptr};
    float* ep_tab_sc[2][3] = {};
    // the state: of the visit being run (bind_visit)
    uint16_t *kc = nullptr, *vc = nullptr;         // FP16 [kv_cap][KV * D] (IK2_KV=f16)
    uint8_t *kc8 = nullptr, *vc8 = nullptr;        // FP8 E4M3 [kv_cap][KV * D] (IK2_KV=fp8), int8 rows (the default)
    int kv_cap = 0;
    float *h_k, *h_v, *h_attn, *h_mlp;             // conv histories [3][C']
    float *rec_old[4] = {}, *rec_x[4] = {};        // (speculative) the convs' histories and inputs of a verify window
};

// A layer visit and its own state.  Without a loop a visit is a layer; a looped core's layer is visited once a pass,
// each time with that visit's K / V cache and conv histories (the weights, and the experts in the cache, are the
// layer's).  bind_visit() points the layer's state at a visit's before it runs.
struct Visit {
    int l = 0;        // the layer
    int pass = 0;     // the core's pass (0: its first, or a layer outside the core)
    uint16_t *kc = nullptr, *vc = nullptr;
    uint8_t *kc8 = nullptr, *vc8 = nullptr;
    int kv_cap = 0;
    float *h_k = nullptr, *h_v = nullptr, *h_attn = nullptr, *h_mlp = nullptr;
    float *rec_old[4] = {}, *rec_x[4] = {};
};

struct Model {
    Config c;
    std::vector<Layer> L;
    std::vector<Visit> V;   // the visits of a token, in order
    Mat out_w;            // [n_rows][C]
    float *out_norm, *out_res_score, *embd_norm;
    float* lp_query_all = nullptr;   // a looped core's message queries, layer by layer ([n_core][4][C]: Layer::lp_query's)
    const uint8_t* embd;  // host Q8_0 [n_rows][C]
    int embd_type;
};

inline Layer& bind_visit(Model& M, int v) {
    const Visit& Vi = M.V[v];
    Layer& Ly = M.L[Vi.l];
    Ly.kc = Vi.kc, Ly.vc = Vi.vc, Ly.kc8 = Vi.kc8, Ly.vc8 = Vi.vc8, Ly.kv_cap = Vi.kv_cap;
    Ly.h_k = Vi.h_k, Ly.h_v = Vi.h_v, Ly.h_attn = Vi.h_attn, Ly.h_mlp = Vi.h_mlp;
    for (int i = 0; i < 4; ++i) Ly.rec_old[i] = Vi.rec_old[i], Ly.rec_x[i] = Vi.rec_x[i];
    return Ly;
}

// ---- the device helpers

cublasHandle_t g_blas;
cudaStream_t g_s;
cudaStream_t g_cs;          // the experts' copies of the token path, beside the compute stream
cudaEvent_t g_ev;
bool g_tg_capturing = false;   // IK2_TOKGRAPH: a decode token's graph being captured
// the device plan's counters set back once this many visits were planned (a token past it runs without a graph)
const long long g_dp_reset_at = getenv("IK2_DP_RESET_AT") ? std::atoll(getenv("IK2_DP_RESET_AT")) : (1LL << 30);
bool g_overlap = true;      // IK2_OVERLAP=0: the copies on the compute stream, before the experts
// The CPU lane (IK2_CPU=1: on): a missing NVFP4 expert computed in host memory by the pool while the others' blocks
// are copied; its own blocks follow on the copy stream without being waited for ("lazy"), so it is in VRAM next time.
ik2::cpu::Pool* g_pool = nullptr;
double g_t_cpu = 0.5e-3, g_t_up = 0.36e-3;   // seconds an expert: the pool's (measured as it runs), the link's
uint64_t g_n_cpu = 0, g_n_wait = 0;
bool g_cpu_lazy = true;   // IK2_CPU_LAZY=0: a CPU-computed expert is not copied in afterwards (the link carries less)
// IK2_LAZY_MAX: lazy copy batches in flight at most; past it a CPU-computed expert is not copied in (its copies would
// queue ahead of the next layers' urgent ones on the one host-to-device copy engine)
int g_lazy_max = 1 << 30;
constexpr int kRing = 256;
cudaEvent_t g_ring[kRing];                   // a lazy batch's end on the copy stream, by sequence number
uint64_t g_seq_next = 1, g_seq_done = 0;
void lazy_poll() {
    while (g_seq_done + 1 < g_seq_next && cudaEventQuery(g_ring[(g_seq_done + 1) % kRing]) == cudaSuccess) ++g_seq_done;
}

// IK2_PROF=1: the stream waited for at each phase's end, its time added to the phase (slower, but where it goes)
bool g_prof = false;
bool g_mv = true;   // IK2_MV=0: a few tokens' experts through the FP16 GEMMs too
bool g_verify_cpu = !getenv("IK2_VERIFY_CPU") || std::atoi(getenv("IK2_VERIFY_CPU"));   // (see multi below)
int g_verify_copies = getenv("IK2_VERIFY_COPIES") ? std::atoi(getenv("IK2_VERIFY_COPIES")) : 1;
// How often each expert is picked by a token (prefill and decode), [n_layer * E]: IK2_POP_OUT=FILE writes it at
// the end (int32 n_layer, int32 E, then uint64 counts); IK2_WARM=FILE fills the expert cache with the most picked
// experts at load, so the first requests do not start from an empty cache.
std::vector<uint64_t> g_pop;
int g_pop_E = 0;
void pop_count(int l, const int* top, int S) {
    if (g_pop.empty()) return;
    for (int i = 0; i < S; ++i) ++g_pop[(size_t) l * g_pop_E + top[i]];
}
void pop_dump() {
    const char* f = getenv("IK2_POP_OUT");
    if (!f || g_pop.empty()) return;
    FILE* o = std::fopen(f, "wb");
    if (!o) return;
    const int32_t hdr[2] = {(int32_t) (g_pop.size() / g_pop_E), (int32_t) g_pop_E};
    std::fwrite(hdr, 4, 2, o);
    std::fwrite(g_pop.data(), 8, g_pop.size(), o);
    std::fclose(o);
}
int g_pop_E_ep = 0;
// IK2_EP=1 with --gpus 2: expert parallel. Every layer's attention and dense weights (and the KV cache) on GPU 0;
// expert e of every layer belongs to GPU e & 1, each GPU with its own cache over all layers and its own PCIe link.
// In decode a layer's picks go to their GPUs at once, both links copying the misses side by side.
bool g_ep = false;
bool g_ep_fast = false;
cudaEvent_t g_ep_x, g_ep_cp[2], g_ep_done;
struct EpStaging {
    void* h_xq = nullptr;
    void* h_route = nullptr;
    void* h_y = nullptr;
    float* d_xq[2] = {};
    int* d_route[2] = {};
    float* d_y[2] = {};
};
EpStaging g_ep_stage;
// IK2_TP=1 with --gpus 2: the experts tensor-parallel instead. GPU d holds half of every expert: rows [d F/2, ..) of
// gate and up and rows [d C/2, ..) of down, so every miss is half on each link. A layer: each GPU its half of h
// (gate/up/SwiGLU), the halves swapped (P2P), each GPU its half of the output rows (down), concatenated on GPU 0.
// Attention, dense weights and the KV cache on GPU 0 as with IK2_EP.
bool g_tp = false;
cudaEvent_t g_tp_h[2];
int g_tp_F = 0, g_tp_C = 0;
inline bool dual() { return g_ep || g_tp; }
// which GPU an expert belongs to, [n_layer * E]: IK2_EP_SPLIT = GPU 1's share of the picks (0.5; GPU 0 also holds the
// attention, the dense weights and the KV cache). With IK2_WARM's profile each layer's experts are dealt by how often
// they are picked, so the two GPUs' shares come out as asked; without it by a hash of the expert.
std::vector<uint8_t> g_ep_own;
inline int ep_owner(int l, int e) { return g_ep_own[(size_t) l * g_pop_E_ep + e]; }
bool g_flash = true;   // IK2_FLASH=0: the prefill's attention (FP16 KV) through the logits in memory and cuBLAS, not fused
int g_n_sm = 0;
bool g_group = true;   // IK2_GROUP=0: 2 .. 8 tokens' experts a slot at a time, their attention a head at a time
bool g_group_exp = true, g_group_attn = true;   // (IK2_GROUP=1: the experts grouped only, 2: the attention only)
FILE* g_trace = nullptr;   // IK2_TRACE=FILE: each layer's picks, records of int32 (layer, T, K, T x K experts)
struct Prof {
    double attn = 0, route = 0, upload = 0, experts = 0, rest = 0;
    double t = 0;
    void start() {
        if (g_prof) cudaStreamSynchronize(g_s), t = now();
    }
    void lap(double& acc) {
        if (!g_prof) return;
        cudaStreamSynchronize(g_s);
        const double n = now();
        acc += n - t;
        t = n;
    }
} g_pf;

template <class T>
T* dalloc(size_t n) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    return p;
}

// Y [T][N] (ldy) = X [T][K] FP16 . W [N][K]^T FP16, F32 sums
void gemm16(const uint16_t* W, int N, int Kd, const uint16_t* X, int T, float* Y, int ldy, float beta = 0.0f) {
    const float one = 1.0f;
    CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, Kd, &one, W, CUDA_R_16F, Kd, X, CUDA_R_16F, Kd, &beta, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

uint8_t* g_stage = nullptr;   // a raw tensor on its way to its FP16 decode
size_t g_stage_bytes = 0;

bool g_dense_q8 = false;
// The KV cache in FP16 (IK2_KV=fp8: E4M3, half the bytes).  Against ref/inkling2.py over 48 tokens FP16 kept every
// top-1 token (KL mean 8.5e-4) and FP8 changed 5 (KL 8.3e-3): Inkling's router is full of near-ties, so FP8's 1.5% in
// layer 0's attention output already moves 7 of 48 tokens' experts.  The full layers' cache is 32 KB a token in FP16.
bool g_kv_fp8 = false;
bool g_kv_i8 = false;   // IK2_KV=i8: kc8 / vc8 hold rows of k::kv_row_bytes_i8(KD) bytes (int8 codes, a scale per 64)
inline size_t kv_tok_b(size_t KD) { return g_kv_i8 ? k::kv_row_bytes_i8((int) KD) : KD * (g_kv_fp8 ? 1 : 2); }
inline bool kv_b8() { return g_kv_fp8 || g_kv_i8; }   // (the cache in kc8 / vc8)
template <class L>
void kv_unpack(const L& Ly, int cap, int k0, int nk, size_t KD, float k_inv, float v_inv, uint16_t* ko, uint16_t* vo,
               cudaStream_t s) {
    if (g_kv_i8) k::unpack_kv_i8_to_f16((const int8_t*) Ly.kc8, (const int8_t*) Ly.vc8, cap, k0, nk, (int) KD, ko, vo, s);
    else k::unpack_kv_fp8_to_f16(Ly.kc8, Ly.vc8, cap, k0, nk, (int) KD, k_inv, v_inv, ko, vo, s);
}
// Q8_0 blocks (2 bytes of FP16 scale, 32 int8 values) taken apart into m.q8 / m.q8d on the device; false if the
// matrix cannot be kept so (another type, or K not a multiple of 128)
bool upload_planar(Mat& m, const std::vector<const uint8_t*>& parts, const std::vector<size_t>& part_values) {
    if (m.type != k::kQ8_0 || m.K % 128) return false;
    size_t n = 0;
    for (size_t v : part_values) n += v;
    std::vector<int8_t> q(n);
    std::vector<uint16_t> d(n / 32);
    size_t b = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        const uint8_t* p = parts[i];
        for (size_t j = 0; j < part_values[i] / 32; ++j, ++b, p += 34) {
            std::memcpy(&d[b], p, 2);
            std::memcpy(&q[b * 32], p + 2, 32);
        }
    }
    m.q8 = dalloc<int8_t>(n);
    m.q8d = dalloc<uint16_t>(n / 32);
    CU(cudaMemcpy(m.q8, q.data(), n, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(m.q8d, d.data(), n / 32 * 2, cudaMemcpyHostToDevice));
    return true;
}

Mat load_mat(const Shards& S, const std::string& n, int N, int Kd) {
    const auto* t = S.info(n);
    if (!t) die("no tensor " + n);
    if (t->elements() != (uint64_t) N * Kd) die(n + ": " + std::to_string(t->elements()) + " values");
    if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
        die(n + ": type " + t->type_name() + " not read here");
    Mat m;
    m.type = (int) t->type, m.N = N, m.K = Kd;
    const size_t nb = tensor_bytes(*t);
    if (g_dense_q8) {
        if (upload_planar(m, {S.data(n)}, {(size_t) N * Kd})) return m;
        m.raw = dalloc<uint8_t>(nb);
        CU(cudaMemcpy(m.raw, S.data(n), nb, cudaMemcpyHostToDevice));
        return m;
    }
    if (nb > g_stage_bytes) {
        if (g_stage) cudaFree(g_stage);
        g_stage = dalloc<uint8_t>(g_stage_bytes = nb);
    }
    CU(cudaMemcpy(g_stage, S.data(n), nb, cudaMemcpyHostToDevice));
    m.f16 = dalloc<uint16_t>((size_t) N * Kd);
    k::decode_f16(m.type, g_stage, (size_t) N * Kd, m.f16, g_s);
    CU(cudaStreamSynchronize(g_s));
    return m;
}

struct SegPack {
    float* dst[6];
    int off[7];
    int n;
};
// (float4 a thread when every width is a multiple of 4 - the segments' lookup unrolled, so the parameters stay in
// the constant bank instead of being copied to local memory for a dynamic index)
__global__ void unpack_segments4_kernel(const float4* __restrict__ src, int T, SegPack p) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    const int stride4 = p.off[p.n] >> 2;
    if (c4 >= stride4 || t >= T) return;
    const float4 v = src[(size_t) t * stride4 + c4];
    const int col = c4 << 2;
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < p.n && col >= p.off[s] && col < p.off[s + 1]) {
            const int w = p.off[s + 1] - p.off[s];
            *(float4*) (p.dst[s] + (size_t) t * w + (col - p.off[s])) = v;
        }
    }
}
__global__ void unpack_segments_kernel(const float* __restrict__ src, int T, SegPack p) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int stride = p.off[p.n];
    if (col >= stride || t >= T) return;
    const float v = src[(size_t) t * stride + col];
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < p.n && col >= p.off[s] && col < p.off[s + 1]) {
            const int w = p.off[s + 1] - p.off[s];
            p.dst[s][(size_t) t * w + (col - p.off[s])] = v;
        }
    }
}
void unpack_segments(const float* src, int T, const std::initializer_list<std::pair<float*, int>>& segs, cudaStream_t s) {
    SegPack p{};
    p.n = (int) segs.size();
    int off = 0, idx = 0;
    bool vec = ((uintptr_t) src & 15) == 0;
    p.off[0] = 0;
    for (const auto& sg : segs) {
        p.dst[idx] = sg.first;
        vec = vec && (sg.second & 3) == 0 && ((uintptr_t) sg.first & 15) == 0;
        off += sg.second;
        p.off[++idx] = off;
    }
    if (vec) {
        const dim3 grid((unsigned) ((off / 4 + 255) / 256), (unsigned) T);
        unpack_segments4_kernel<<<grid, 256, 0, s>>>((const float4*) src, T, p);
    } else {
        const dim3 grid((unsigned) ((off + 255) / 256), (unsigned) T);
        unpack_segments_kernel<<<grid, 256, 0, s>>>(src, T, p);
    }
}

struct MatPart {
    std::string name;
    int N = 0;
    std::vector<float> f32;
};

void ensure_stage(size_t nb) {
    if (nb > g_stage_bytes) {
        if (g_stage) cudaFree(g_stage);
        g_stage = dalloc<uint8_t>(g_stage_bytes = nb);
    }
}

Mat load_fused_f16(const Shards& S, const std::vector<MatPart>& parts, int Kd) {
    int N_tot = 0;
    for (const auto& pt : parts) N_tot += pt.N;
    Mat m;
    m.N = N_tot;
    m.K = Kd;
    m.f16 = dalloc<uint16_t>((size_t) N_tot * Kd);
    int row = 0;
    for (const auto& pt : parts) {
        if (pt.f32.empty()) {
            const auto* t = S.info(pt.name);
            if (!t) die("no tensor " + pt.name);
            if (t->elements() != (uint64_t) pt.N * Kd) die(pt.name + ": " + std::to_string(t->elements()) + " values");
            if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
                die(pt.name + ": type " + t->type_name() + " not read here");
            const size_t nb = tensor_bytes(*t);
            ensure_stage(nb);
            CU(cudaMemcpy(g_stage, S.data(pt.name), nb, cudaMemcpyHostToDevice));
            k::decode_f16((int) t->type, g_stage, (size_t) pt.N * Kd, m.f16 + (size_t) row * Kd, g_s);
        } else {
            const size_t nb = pt.f32.size() * 4;
            ensure_stage(nb);
            CU(cudaMemcpy(g_stage, pt.f32.data(), nb, cudaMemcpyHostToDevice));
            k::to_f16((const float*) g_stage, pt.f32.size(), m.f16 + (size_t) row * Kd, g_s);
        }
        row += pt.N;
    }
    CU(cudaStreamSynchronize(g_s));
    return m;
}

bool try_load_fused_raw(const Shards& S, const std::vector<MatPart>& parts, int Kd, Mat& out) {
    int ty = -1, N_tot = 0;
    size_t nb_tot = 0;
    for (const auto& pt : parts) {
        if (!pt.f32.empty()) return false;
        const auto* t = S.info(pt.name);
        if (!t) die("no tensor " + pt.name);
        if (t->elements() != (uint64_t) pt.N * Kd) die(pt.name + ": " + std::to_string(t->elements()) + " values");
        if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
            die(pt.name + ": type " + t->type_name() + " not read here");
        if (ty < 0) ty = (int) t->type;
        else if (ty != (int) t->type) return false;
        N_tot += pt.N;
        nb_tot += tensor_bytes(*t);
    }
    out = Mat{};
    out.type = ty;
    out.N = N_tot;
    out.K = Kd;
    {
        std::vector<const uint8_t*> ps;
        std::vector<size_t> vs;
        for (const auto& pt : parts) ps.push_back(S.data(pt.name)), vs.push_back((size_t) pt.N * Kd);
        if (upload_planar(out, ps, vs)) return true;
    }
    out.raw = dalloc<uint8_t>(nb_tot);
    size_t off = 0;
    for (const auto& pt : parts) {
        const size_t nb = tensor_bytes(*S.info(pt.name));
        CU(cudaMemcpy(out.raw + off, S.data(pt.name), nb, cudaMemcpyHostToDevice));
        off += nb;
    }
    return true;
}

// one token's product over a dense matrix kept as the GGUF's blocks, or planar
void mv1(const Mat& m, const float* x, float* y) {
    if (m.q8) k::mv_q8p(m.q8, m.q8d, m.N, m.K, x, y, g_s);
    else k::mv_blocks(m.type, m.raw, m.N, m.K, x, y, g_s);
}
// ... with up to two short convs on its rows: the planar Q8 product has them as its epilogue (a launch less each;
// those rows of y are then not written), any other matrix gets the product and then the convs
void mv1c(const Mat& m, const float* x, float* y, const k::MvConv& c0, const k::MvConv& c1 = k::MvConv{}) {
    if (m.q8) {
        k::mv_q8p_conv(m.q8, m.q8d, m.N, m.K, x, y, c0, c1, g_s);
        return;
    }
    k::mv_blocks(m.type, m.raw, m.N, m.K, x, y, g_s);
    for (const k::MvConv* cv : {&c0, &c1})
        if (cv->n) k::conv4_res(y + cv->r0, cv->hist, cv->w, cv->dyn, cv->heads, 1, cv->n, cv->res, cv->y, g_s);
}

bool g_q8_t = !getenv("IK2_Q8_T") || std::atoi(getenv("IK2_Q8_T"));
uint16_t* g_w16 = nullptr;   // a matrix's FP16 decode for the prompt path's GEMMs (the largest of a layer's)
size_t g_w16_n = 0;          // values it holds
// Y [T][N] = X [T][K] . W^T: one token straight over the blocks (x32), more through an FP16 decode and a GEMM (x16).
// A matrix larger than the decode buffer (the head: its few rows of logits) goes a token at a time over its blocks.
void matmul(const Mat& m, const float* x32, const uint16_t* x16, int T, float* Y, int ldy) {
    if (m.f16) {
        gemm16(m.f16, m.N, m.K, x16, T, Y, ldy);
        return;
    }
    if (m.q8 && T > 1 && T <= 8 && m.K % 128 == 0 && g_q8_t) {   // (a short chunk: the weights read once for all T)
        k::mv_q8p_t(m.q8, m.q8d, m.N, m.K, x32, T, Y, ldy, g_s);
        return;
    }
    if (m.K <= 8192 && (T == 1 || (size_t) m.N * m.K > g_w16_n)) {
        for (int t = 0; t < T; ++t) mv1(m, x32 + (size_t) t * m.K, Y + (size_t) t * ldy);
        return;
    }
    if (m.q8) k::decode_q8p_f16(m.q8, m.q8d, (size_t) m.N * m.K, g_w16, g_s);
    else k::decode_f16(m.type, m.raw, (size_t) m.N * m.K, g_w16, g_s);
    gemm16(g_w16, m.N, m.K, x16, T, Y, ldy);
}

std::vector<float> host_f32(const Shards& S, const std::string& n, size_t expect) {
    const auto* t = S.info(n);
    if (!t) die("no tensor " + n);
    if (t->elements() != expect) die(n + ": " + std::to_string(t->elements()) + " values, expected " + std::to_string(expect));
    std::vector<float> v(expect);
    const uint8_t* p = S.data(n);
    if (t->type == 0) {
        std::memcpy(v.data(), p, expect * 4);
    } else if (t->type == 30) {
        for (size_t i = 0; i < expect; ++i) v[i] = bf16_to_f32((uint16_t) (p[2 * i] | (p[2 * i + 1] << 8)));
    } else if (k::block_values((int) t->type)) {
        k::decode_host((int) t->type, p, expect, v.data());
    } else {
        die(n + ": type " + t->type_name());
    }
    return v;
}
// F32 values as FP16 on the device (round to nearest)
uint16_t* dev_f16_of(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __half_as_ushort(__float2half_rn(v[i]));
    uint16_t* p = dalloc<uint16_t>(h.size());
    CU(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    return p;
}
float* dev_f32(const std::vector<float>& v) {
    float* p = dalloc<float>(v.size());
    CU(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return p;
}
uint16_t* dev_f16(const std::vector<float>& v) {
    const size_t nb = v.size() * 4;
    ensure_stage(nb);
    CU(cudaMemcpy(g_stage, v.data(), nb, cudaMemcpyHostToDevice));
    uint16_t* p = dalloc<uint16_t>(v.size());
    k::to_f16((const float*) g_stage, v.size(), p, g_s);
    CU(cudaStreamSynchronize(g_s));
    return p;
}
float* load_f32(const Shards& S, const std::string& n, size_t expect) { return dev_f32(host_f32(S, n, expect)); }
uint16_t* load_f16(const Shards& S, const std::string& n, size_t expect) { return dev_f16(host_f32(S, n, expect)); }

// ---- loading the experts into pinned host memory: pread from the files, several at once

struct Pinned {
    struct Block {
        void* ptr = nullptr;
        size_t bytes = 0;
        bool mapped = false;
        bool registered = false;
    };
    std::vector<Block> blocks;
    ~Pinned() {
        for (const auto& b : blocks) {
            if (b.mapped) {
                if (b.registered) cudaHostUnregister(b.ptr);
                ::munmap(b.ptr, b.bytes);
            } else if (b.ptr) {
                cudaFreeHost(b.ptr);
            }
        }
    }
};

void read_into_strided(const std::string& path, uint64_t off, int n_exp, size_t exp_bytes, size_t dst_stride,
                       uint8_t* dst) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open " + path);
    const size_t chunk = 64ull << 20;
    for (int e = 0; e < n_exp; ++e) {
        size_t done = 0;
        uint8_t* ep = dst + (size_t) e * dst_stride;
        const uint64_t eoff = off + (uint64_t) e * exp_bytes;
        while (done < exp_bytes) {
            const ssize_t r = ::pread(fd, ep + done, std::min(chunk, exp_bytes - done), (off_t) (eoff + done));
            if (r <= 0) die("read failed: " + path);
            done += (size_t) r;
        }
    }
    posix_fadvise(fd, (off_t) off, (off_t) ((uint64_t) n_exp * exp_bytes), POSIX_FADV_DONTNEED);
    ::close(fd);
}

// ---- the expert cache in VRAM: per-size-class pools, O(1) S3-FIFO with ghost history

// GPU d's half of role r (0 gate, 1 up, 2 down) of an expert: offset in the role's part, bytes (IK2_TP)
inline void tp_part(const Layer& Ly, int d, int r, size_t& off, size_t& bytes) {
    const size_t rows = r < 2 ? (size_t) g_tp_F : (size_t) g_tp_C, rb = Ly.eb[r] / rows, half = rows / 2;
    off = (size_t) d * half * rb, bytes = half * rb;
}
inline size_t tp_bytes(const Layer& Ly, int d) {
    size_t o, b, t = 0;
    for (int r = 0; r < 3; ++r) tp_part(Ly, d, r, o, b), t += b;
    return t;
}
struct ExpertCache {
    static constexpr uint8_t kMaxFreq = 7;
    struct Pool {
        size_t slot_bytes = 0;
        int n_slots = 0, prob_cap = 0, n_layers = 0;
        int n_active = 0;               // slots in use as cache; [n_active, n_slots) lent out (lend)
        int kv_res = 0;                 // the last kv_res slots: the full layers' K / V (--kv-grow), never given back
        uint8_t* data = nullptr;
        std::vector<int> owner;         // slot -> expert idx (l * E + e), -1 if free
        std::vector<uint64_t> used;     // slot -> last tick
        std::vector<uint64_t> sseq;     // slot -> lazy batch seq (0: none)
        std::vector<uint8_t> qid;       // 0=free, 1=prob, 2=main
        std::vector<uint8_t> freq;      // saturating access counter [0..kMaxFreq]
        std::vector<int> prev, next;
        int prob_head = -1, prob_tail = -1, prob_sz = 0;
        int main_head = -1, main_tail = -1, main_sz = 0;
        int free_head = -1;
        uint64_t ghost_seq = 1;
    };

    int n_layers = 0, E = 0;
    std::vector<Pool> pools;
    std::vector<int> layer_pool;       // layer l -> pool index (-1 if on another GPU)
    std::vector<int> loc;              // l * E + e -> slot in pools[layer_pool[l]], or -1
    std::vector<uint64_t> ghost_at;    // l * E + e -> ghost_seq when evicted in decode (0: none)
    int last_handle = -1;
    bool scan = false;   // a long prefill's pass (evict_scan; IK2_SCAN=0: off)
    int tp = -1;         // IK2_TP: this GPU's half of every expert (its index), not whole experts
    uint64_t tick = 0, hits = 0, misses = 0, bytes_up = 0;
    double t_up = 0;

    static void push_back(Pool& P, int q, int s) {
        P.qid[s] = (uint8_t) q;
        P.prev[s] = -1;
        P.next[s] = -1;
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        if (tl < 0) hd = tl = s;
        else P.next[tl] = s, P.prev[s] = tl, tl = s;
        ++sz;
    }
    static int pop_front(Pool& P, int q) {
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        const int s = hd;
        if (s < 0) return -1;
        hd = P.next[s];
        if (hd >= 0) P.prev[hd] = -1;
        else tl = -1;
        P.prev[s] = P.next[s] = -1;
        P.qid[s] = 0;
        --sz;
        return s;
    }
    static int evict_from_prob(Pool& P, uint64_t t, bool& saw_busy_lazy) {
        const int n = P.prob_sz;
        for (int i = 0; i < n; ++i) {
            const int s = pop_front(P, 1);
            if (P.used[s] >= t) {
                push_back(P, 1, s);
                continue;
            }
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) {
                    saw_busy_lazy = true;
                    push_back(P, 1, s);
                    continue;
                }
            }
            if (P.freq[s] > 0) {
                P.freq[s] = std::min<uint8_t>(P.freq[s] - 1, 1);
                push_back(P, 2, s);
                continue;
            }
            return s;
        }
        return -1;
    }
    static int evict_from_main(Pool& P, uint64_t t, bool& saw_busy_lazy) {
        int pinned_run = 0;
        while (P.main_sz > 0 && pinned_run < P.main_sz) {
            const int s = pop_front(P, 2);
            if (P.used[s] >= t) {
                ++pinned_run;
                push_back(P, 2, s);
                continue;
            }
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) {
                    saw_busy_lazy = true;
                    ++pinned_run;
                    push_back(P, 2, s);
                    continue;
                }
            }
            pinned_run = 0;
            if (P.freq[s] > 0) {
                --P.freq[s];
                push_back(P, 2, s);
                continue;
            }
            return s;
        }
        return -1;
    }
    // A pass that streams nearly every expert of every layer in order (a long prefill), once the cache is full:
    // the victim is the most recently taken probation slot that is free to go (in practice an expert of the layer
    // just computed), not the oldest. Under the usual order such a pass, larger than the cache, evicts the experts of
    // the layers it is about to reach, and the next pass finds none of them (32k after other prompts: 9% hits);
    // like this, the streamed experts take turns in a few slots and the rest of the cache stays for the next pass.
    static int evict_scan(Pool& P, uint64_t t) {
        for (int s = P.prob_tail; s >= 0; s = P.prev[s]) {
            if (P.used[s] >= t) continue;
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) continue;
            }
            unlink(P, s);
            return s;
        }
        return -1;
    }
    static int alloc_slot(Pool& P, uint64_t t, bool scan = false) {
        if (P.free_head >= 0) {
            const int s = P.free_head;
            P.free_head = P.next[s];
            P.next[s] = P.prev[s] = -1;
            return s;
        }
        if (scan) {
            const int s = evict_scan(P, t);
            if (s >= 0) return s;
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool saw_busy_lazy = false;
            int s = -1;
            if (P.prob_sz > P.prob_cap || P.main_sz == 0) {
                s = evict_from_prob(P, t, saw_busy_lazy);
                if (s < 0) s = evict_from_main(P, t, saw_busy_lazy);
            } else {
                s = evict_from_main(P, t, saw_busy_lazy);
                if (s < 0) {
                    const int prev_main = P.main_sz;
                    s = evict_from_prob(P, t, saw_busy_lazy);
                    if (s < 0 && P.main_sz > prev_main) s = evict_from_main(P, t, saw_busy_lazy);
                }
            }
            if (s >= 0) return s;
            if (attempt == 0 && saw_busy_lazy) {
                CU(cudaStreamSynchronize(g_cs));
                g_seq_done = g_seq_next - 1;
                continue;
            }
            break;
        }
        die("expert cache: no slot free");
    }

    void init(const std::vector<Layer>& L, int dev, int n_layers_in, int E_in, size_t budget) {
        n_layers = n_layers_in;
        E = E_in;
        layer_pool.assign(n_layers, -1);
        loc.assign((size_t) n_layers * E, -1);
        ghost_at.assign((size_t) n_layers * E, 0);
        struct Group {
            size_t sb;
            std::vector<int> layers;
        };
        std::map<size_t, std::vector<int>> by_sb;
        for (int l = 0; l < n_layers; ++l) {
            if (dev >= 0 && L[l].dev != dev) continue;   // (dev -1: every layer, an expert-parallel GPU)
            const size_t eb = tp >= 0 ? tp_bytes(L[l], tp) : L[l].eb[0] + L[l].eb[1] + L[l].eb[2];
            const size_t sb = (eb + 255) / 256 * 256;
            by_sb[sb].push_back(l);
        }
        std::vector<Group> grps;
        for (auto& kv : by_sb) grps.push_back({kv.first, std::move(kv.second)});
        auto min_req_bytes = [&]() {
            size_t s = 0;
            for (const auto& g : grps) s += (size_t) std::min(E, (int) g.layers.size() * E) * g.sb;
            return s;
        };
        while (grps.size() > 1 && (min_req_bytes() > budget || grps.size() > 4)) {
            size_t best_waste = SIZE_MAX;
            size_t best_i = 0;
            for (size_t i = 0; i + 1 < grps.size(); ++i) {
                const size_t w = grps[i].layers.size() * (grps[i + 1].sb - grps[i].sb);
                if (w < best_waste) best_waste = w, best_i = i;
            }
            grps[best_i + 1].layers.insert(grps[best_i + 1].layers.end(), grps[best_i].layers.begin(),
                                           grps[best_i].layers.end());
            grps.erase(grps.begin() + (ptrdiff_t) best_i);
        }
        const size_t hard_min = min_req_bytes();
        if (hard_min > budget)
            die("--cache-gb too small: need at least " + std::to_string((hard_min + (1 << 20) - 1) >> 20) + " MiB");
        size_t soft_min = 0;
        for (const auto& g : grps) soft_min += (size_t) std::min(E + 16, (int) g.layers.size() * E) * g.sb;
        const int min_base = soft_min <= budget ? E + 16 : E;

        pools.resize(grps.size());
        std::vector<int> pmin(grps.size()), pmax(grps.size());
        for (size_t p = 0; p < grps.size(); ++p) {
            pools[p].slot_bytes = grps[p].sb;
            pools[p].n_layers = (int) grps[p].layers.size();
            for (int l : grps[p].layers) layer_pool[l] = (int) p;
            pmax[p] = pools[p].n_layers * E;
            pmin[p] = std::min(min_base, pmax[p]);
        }
        auto used_for = [&](double spl) {
            size_t u = 0;
            for (size_t p = 0; p < pools.size(); ++p) {
                const int ns = std::clamp((int) std::floor(pools[p].n_layers * spl), pmin[p], pmax[p]);
                u += (size_t) ns * pools[p].slot_bytes;
            }
            return u;
        };
        double lo = 0.0, hi = (double) E;
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (used_for(mid) <= budget) lo = mid;
            else hi = mid;
        }
        size_t used = 0;
        for (size_t p = 0; p < pools.size(); ++p) {
            pools[p].n_slots = std::clamp((int) std::floor(pools[p].n_layers * lo), pmin[p], pmax[p]);
            used += (size_t) pools[p].n_slots * pools[p].slot_bytes;
        }
        size_t rem = budget - used;
        bool grew = true;
        while (grew) {
            grew = false;
            for (size_t p = 0; p < pools.size(); ++p) {
                if (pools[p].n_slots < pmax[p] && pools[p].slot_bytes <= rem) {
                    ++pools[p].n_slots;
                    rem -= pools[p].slot_bytes;
                    grew = true;
                }
            }
        }
        for (size_t p = 0; p < pools.size(); ++p) {
            Pool& P = pools[p];
            P.prob_cap = std::max(1, P.n_slots / 5);
            P.data = dalloc<uint8_t>((size_t) P.n_slots * P.slot_bytes);
            P.owner.assign(P.n_slots, -1);
            P.used.assign(P.n_slots, 0);
            P.sseq.assign(P.n_slots, 0);
            P.qid.assign(P.n_slots, 0);
            P.freq.assign(P.n_slots, 0);
            P.prev.assign(P.n_slots, -1);
            P.next.assign(P.n_slots, -1);
            for (int s = 0; s < P.n_slots - 1; ++s) P.next[s] = s + 1;
            P.free_head = P.n_slots > 0 ? 0 : -1;
            P.n_active = P.n_slots;
        }
    }
    int total_slots() const {
        int s = 0;
        for (const auto& P : pools) s += P.n_slots;
        return s;
    }
    size_t total_bytes() const {
        size_t b = 0;
        for (const auto& P : pools) b += (size_t) P.n_slots * P.slot_bytes;
        return b;
    }
    uint64_t& slot_sseq(int handle) { return pools[handle >> 20].sseq[handle & 0xfffff]; }
    // the slot of expert e of layer l, its blocks copied in if missing; pinned for this tick (not evicted in it)
    // st: the stream the copy is queued on; *missed (given): whether it was copied in now
    // copy false: the slot is taken but its blocks are the caller's to copy in (copy_in)
    uint8_t* get(const Layer& Ly, int l, int e, uint64_t t, cudaStream_t st = nullptr, bool* missed = nullptr,
                 bool copy = true, bool is_prefill = false) {
        if (!st) st = g_s;
        if (missed) *missed = false;
        const int idx = l * E + e;
        const int p = layer_pool[l];
        Pool& P = pools[p];
        if (loc[idx] >= 0) {
            const int s = loc[idx];
            ++hits;
            P.used[s] = t;
            if (P.freq[s] < kMaxFreq) ++P.freq[s];
            last_handle = (p << 20) | s;
            return P.data + (size_t) s * P.slot_bytes;
        }
        ++misses;
        if (missed) *missed = true;
        const int best = alloc_slot(P, t, scan);
        if (P.owner[best] >= 0) {
            const int old_idx = P.owner[best];
            loc[old_idx] = -1;
            if (!is_prefill) ghost_at[old_idx] = P.ghost_seq;
        }
        if (!is_prefill) ++P.ghost_seq;
        const bool is_ghost = !is_prefill && ghost_at[idx] != 0 && (P.ghost_seq - ghost_at[idx]) <= (uint64_t) P.n_slots;
        P.owner[best] = idx;
        P.used[best] = t;
        P.sseq[best] = 0;
        P.freq[best] = is_ghost ? 1 : 0;
        push_back(P, is_ghost ? 2 : 1, best);
        loc[idx] = best;
        last_handle = (p << 20) | best;
        uint8_t* dst = P.data + (size_t) best * P.slot_bytes;
        if (copy) copy_in(Ly, e, dst, st);
        return dst;
    }
    static void unlink(Pool& P, int s) {
        const int q = P.qid[s];
        if (!q) return;
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        if (P.prev[s] >= 0) P.next[P.prev[s]] = P.next[s];
        else hd = P.next[s];
        if (P.next[s] >= 0) P.prev[P.next[s]] = P.prev[s];
        else tl = P.prev[s];
        P.prev[s] = P.next[s] = -1;
        P.qid[s] = 0;
        --sz;
    }
    static void rebuild_free(Pool& P) {
        P.free_head = -1;
        for (int s = P.n_active - 1; s >= 0; --s)
            if (P.owner[s] < 0 && P.qid[s] == 0) P.next[s] = P.free_head, P.prev[s] = -1, P.free_head = s;
    }
    // Lends the last k cache slots of pool p as one block of scratch (the experts in them are dropped), for a long
    // prefill whose passes stream the experts anyway; give_back() makes them cache slots again. Nothing may be in
    // flight on those slots (the caller synchronizes first).
    uint8_t* lend(int p, int k) {
        Pool& P = pools[p];
        if (k <= 0 || k > P.n_active) return nullptr;
        const int s0 = P.n_active - k;
        for (int s = s0; s < P.n_active; ++s) {
            unlink(P, s);
            if (P.owner[s] >= 0) loc[P.owner[s]] = -1, P.owner[s] = -1;
            P.used[s] = 0, P.sseq[s] = 0, P.freq[s] = 0;
        }
        P.n_active = s0;
        rebuild_free(P);
        return P.data + (size_t) s0 * P.slot_bytes;
    }
    // slot s of pool p holds expert key now (IK2_DEVPLAN's copies written back), at the main queue's tail
    void assign(int p, int s, int key, uint64_t t) {
        Pool& P = pools[p];
        if (P.owner[s] >= 0) loc[P.owner[s]] = -1;
        unlink(P, s);
        P.owner[s] = key;
        loc[key] = s;
        P.used[s] = t, P.sseq[s] = 0, P.freq[s] = 1;
        push_back(P, 2, s);
    }
    void refree(int p) { rebuild_free(pools[p]); }
    void give_back(int p) {
        Pool& P = pools[p];
        P.n_active = P.n_slots - P.kv_res;
        rebuild_free(P);
    }
    bool has(int l, int e) const { return loc[l * E + e] >= 0; }
    void copy_in(const Layer& Ly, int e, uint8_t* dst, cudaStream_t st) {
        if (tp >= 0) {   // (gate's half, up's half, down's half, packed)
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                size_t o, b;
                tp_part(Ly, tp, r, o, b);
                CU(cudaMemcpyAsync(dst + off, Ly.host[r] + (size_t) e * Ly.stride[r] + o, b, cudaMemcpyHostToDevice, st));
                off += b, bytes_up += b;
            }
            return;
        }
        if (Ly.host_contig) {
            CU(cudaMemcpyAsync(dst, Ly.host[0] + (size_t) e * Ly.slot_bytes, Ly.slot_bytes, cudaMemcpyHostToDevice, st));
            bytes_up += Ly.slot_bytes;
            return;
        }
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            CU(cudaMemcpyAsync(dst + off, Ly.host[r] + (size_t) e * Ly.stride[r], Ly.eb[r], cudaMemcpyHostToDevice, st));
            off += Ly.eb[r];
            bytes_up += Ly.eb[r];
        }
    }
};

// ---- the forward

struct Work {
    static constexpr int kPackRing = 4;
    static constexpr int kMissBatch = 24;
    int Tm, ctx, nk_max, T_sup = 0;
    size_t blk_ld = 0, lg_cap = 0;
    float *h, *res, *r, *a, *tmp, *xq, *xk, *xv, *q, *kk, *vv, *rr, *gate, *sink, *dyn, *kconv, *vconv, *o, *lg;
    float *qkv;         // fused projection staging (aliases gu32)
    float *blocks;      // [n_blk][blk_ld][C]
    uint16_t *x16, *xk16, *xv16, *xq16_sup = nullptr;
    uint16_t *kv_k16 = nullptr, *kv_v16 = nullptr;
    int kv_tmp_cap = 0;
    int* pos;
    float* rlog;
    int* route_d;       // contiguous [S] top (int) followed by [S] weights (float)
    // the experts
    uint16_t *ex16, *w16, *h16;
    float *gu32, *y32, *sg, *su, *wt;
    int *gidx, *inv;
    const uint8_t** blob;
    int* tok;
    uint8_t *pack_h[kPackRing] = {}, *pack_d[kPackRing] = {};   // ring-buffered token path tables
    uint16_t* tp_h = nullptr;   // IK2_TP: the whole h, [S][F] FP16 (each GPU fills its half, then the other's)
    uint8_t* tp_pk = nullptr;   // (the decode's tables of this layer on the device: canonical blobs at +1536)
    float *y_h[kPackRing] = {};                                 // ring-buffered CPU lane outputs [64][C]
    int pack_slot = 0;
    float *x_h = nullptr;                                       // CPU lane inputs [8][C], pinned
    float *emb_h = nullptr;                                     // pinned host staging
    int *pos_h = nullptr, *route_h = nullptr;
    int *gidx_h[2] = {}, *inv_h[2] = {};
    float *sg_h[2] = {}, *su_h[2] = {}, *wt_h[2] = {};
    int *argmax_d = nullptr, *argmax_h = nullptr;
    std::vector<cudaEvent_t> miss_ev;
    // prefill: the next layer's experts copied in while this one runs (prefetch_layer)
    // the prefetches in flight, a ring by layer: their batches' events, expert -> its batch
    // (-1: not prefetched), the layer each slot is for
    static constexpr int kPfRing = 4;
    std::vector<cudaEvent_t> pf_evs[kPfRing];
    std::vector<int> pf_bs[kPfRing];
    int pf_layers[kPfRing] = {-1, -1, -1, -1};
    std::vector<int> miss_pos;   // expert -> its index among this layer's misses (-1: not one)
    std::vector<uint32_t> use;   // [layer * E + e] how many tokens each expert ran in prefills so far
    std::vector<uint64_t> use_tot;   // [layer] their sum
    float* wtr = nullptr;        // [S] the routing weight of each expert-sorted row
    float* wtr_h[2] = {nullptr, nullptr};
    // prefill attention: the relative bias of up to kRelbRows queries x heads, [rows * H][extent] (IK2_RELB=0: off)
    static constexpr int kRelbRows = 512;
    float* relb = nullptr;
    int relb_ext = 0;
    uint16_t* lg16 = nullptr;   // the prefill attention's probabilities (FP16, lg_cap of them; aliases gu32)
    size_t gu32_bytes = 0;   // (gu32's size: a lent scratch only replaces it when it needs more)
    int T_host = 0;          // tokens the pinned host staging (pos_h, route_h, gidx_h, ..) holds
    std::vector<int> cnt, start, fill, hit_exps, miss_exps;
    std::vector<uint8_t*> slot_ptrs;
    std::vector<uint32_t> seen_stamp, gi_stamp;
    std::vector<char> seen_late;
    std::vector<int> gi_idx;
    uint32_t stamp = 0;
    float* logits;
    // A looped core's passes ([Tm][C] each): the stream the core was first given; the bank of the pass before (the
    // blocks it added after its first, then its final stream); a layer's four messages (q's, k's, v's, the MoE's);
    // its three attention inputs before the messages.
    float *lp_pre = nullptr, *lp_bank = nullptr, *lp_msg = nullptr, *lp_in = nullptr;
    // a looped core in a lent layer-major pass (forward_prefill_sup): the stream kept, the bank, the messages [4][T][C]
    // and a sub-chunk's of them [4][512][C] (null outside such a pass)
    float *lp_pre_s = nullptr, *lp_bank_s = nullptr, *lp_msg_s = nullptr, *lp_sub = nullptr;
    float* lp_msg_all = nullptr;   // a token's messages of a pass, every core layer's ([n_core][4][C]): one launch a pass
    float* lp_g = nullptr;   // the message gates' scratch [3][Tm][2 rank + 2]
    float *lp_a = nullptr, *lp_f = nullptr;   // ... of a chunk's: [3][Tm][rank] and [2][Tm][C]
    float* logits_h = nullptr;   // the resident token path's logits, pinned [n_vocab]
    int *cand_d = nullptr, *cand_h = nullptr, *cand_part = nullptr;   // ... its candidates (k::cand_topk's out), scratch
    // a short prompt's experts by groups, every expert in VRAM (forward): the tables, pinned and on the device -
    // [rg_groups] pointers | n [rg_groups] | tok, row [rg_groups][kGroupMax] | sg, su, wt, id [rg_slots]
    uint8_t *rg_h = nullptr, *rg_d = nullptr;
    int rg_slots = 0, rg_groups = 0;
    size_t rg_bytes = 0;
};

// --serve: SIGUSR1 asks the request running to stop (its client went away) at its next token or prompt chunk
volatile std::sig_atomic_t g_cancel = 0;
// IK2_STATS=1: per request, how many tokens the prefill's experts each ran ([0] 1, [1] 2, [2] 3-7, [3] 8+)
uint64_t g_cnt_hist[4] = {0, 0, 0, 0};
// IK2_STATS: a request's experts by where they came from: used and already cached / used and prefetched for this
// pass / used and copied after the routing / prefetched and not used
uint64_t g_src_hist[4] = {0, 0, 0, 0};
extern "C" void on_usr1(int) { g_cancel = 1; }

// Speculative decoding: g_taps [g_taps_n][g_taps_ld][C] gets the drafter's taps of every position a forward call runs (the
// stream at the start of layers 5, 11, .., 41, the final normed hidden); with g_rec each short conv's history and
// inputs are kept, so that a verify window's rejected part can be taken back (rollback()).
float* g_taps = nullptr;
int g_taps_ld = 0;
int g_taps_n = 0;
bool g_rec = false;
void conv_rec(Layer& Ly, int which, const float* x, const float* hist, int T, int Cc) {
    if (!g_rec) return;
    CU(cudaMemcpyAsync(Ly.rec_old[which], hist, (size_t) 3 * Cc * 4, cudaMemcpyDeviceToDevice, g_s));
    CU(cudaMemcpyAsync(Ly.rec_x[which], x, (size_t) T * Cc * 4, cudaMemcpyDeviceToDevice, g_s));
}

// The pictures of the request running (GENI): the image encoder's rows (strata-vision's SVE1 records, [n][C] F32), each
// in place of one image token of the prompt - the model reads them as they are, without the embedding's norm.
struct Pictures {
    std::vector<float> rows;
    std::vector<int> row_at;   // a prompt position's row (-1: a token)
    int n = 0;
    void clear() { rows.clear(), row_at.clear(), n = 0; }
    int at(int pos) const { return n && pos < (int) row_at.size() ? row_at[pos] : -1; }
};
Pictures g_img;
// the rows of `path` for the image tokens of ids in order; sig[i]: a hash of position i's row (0: a token), the
// kept state's key beside the ids.  -> an error's text, or empty
std::string load_pictures(const std::string& path, const std::vector<int>& ids, int image_token, int C, Pictures& P,
                          std::vector<uint64_t>& sig) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot read the pictures' file";
    for (;;) {
        int32_t h[5];
        if (std::fread(h, 4, 5, f) != 5) break;
        if (h[0] != 0x31455653 || h[1] <= 0 || h[4] != C) {
            std::fclose(f);
            return "the pictures' file is not the image encoder's for this model";
        }
        const size_t at = P.rows.size(), n = (size_t) h[1] * C;
        P.rows.resize(at + n);
        if (std::fread(P.rows.data() + at, 4, n, f) != n) {
            std::fclose(f);
            return "the pictures' file is cut short";
        }
    }
    std::fclose(f);
    const int n_rows = (int) (P.rows.size() / C);
    P.row_at.assign(ids.size(), -1);
    int k = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != image_token) continue;
        if (k >= n_rows) return "the prompt has more image tokens than its pictures have rows";
        P.row_at[i] = k;
        uint64_t hsh = 1469598103934665603ull;   // FNV-1a over the row's bytes
        const uint8_t* b = (const uint8_t*) (P.rows.data() + (size_t) k * C);
        for (size_t j = 0; j < (size_t) C * 4; ++j) hsh = (hsh ^ b[j]) * 1099511628211ull;
        sig[i] = hsh | 1;
        ++k;
    }
    if (k != n_rows) return "the prompt's image tokens and its pictures' rows do not match";
    P.n = n_rows;
    return "";
}
// the pictures' rows of positions pos0 .. pos0 + T - 1 over their tokens' embeddings in h [T][C]
void place_pictures(float* h, int T, int pos0, int C, cudaStream_t s) {
    if (!g_img.n) return;
    for (int t = 0; t < T;) {
        const int r = g_img.at(pos0 + t);
        if (r < 0) {
            ++t;
            continue;
        }
        int n = 1;
        while (t + n < T && g_img.at(pos0 + t + n) == r + n) ++n;
        CU(cudaMemcpyAsync(h + (size_t) t * C, g_img.rows.data() + (size_t) r * C, (size_t) n * C * 4,
                           cudaMemcpyHostToDevice, s));
        t += n;
    }
}

// a GPU's own: its streams and cuBLAS handle, work buffers and expert cache; use_dev() makes it the current one (the
// globals g_s, g_cs, g_ev, g_blas the kernels and copies go to)
struct DevCtx {
    cudaStream_t s = nullptr, cs = nullptr;
    cudaEvent_t ev = nullptr;
    cublasHandle_t blas = nullptr;
    Work W{};
    ExpertCache EC;
};
std::vector<DevCtx> g_dev;
int g_cur = -1;
Work& dev_work(int d) { return g_dev[d].W; }
ExpertCache& dev_cache(int d) { return g_dev[d].EC; }
void use_dev(int d) {
    if (d == g_cur) return;
    if (g_stage) cudaFree(g_stage), g_stage = nullptr, g_stage_bytes = 0;   // (the previous device's)
    CU(cudaSetDevice(d));
    g_cur = d;
    g_s = g_dev[d].s, g_cs = g_dev[d].cs, g_ev = g_dev[d].ev, g_blas = g_dev[d].blas;
}

// (kCand: sampling's candidates - the resident token path finds the largest logits on the device, g_cand; any other
// path gives the logits as kLogits does and leaves g_cand.n at -1)
enum class OutMode { kNone, kArgmax, kLogits, kCand };
constexpr int kCandK = 1000, kCandCap = 4096;   // the largest logits sampling looks at; the candidates a token keeps
struct Cand {
    int n = -1;                    // candidates of the last kCand token (-1: its logits were given instead)
    const int* idx = nullptr;      // [n] token ids, in no order
    const float* val = nullptr;    // [n] their logits
} g_cand;

__global__ void gather_f16_16_kernel(const uint16_t* __restrict__ x16, const int* __restrict__ idx, int n, int C,
                                     uint16_t* __restrict__ y16) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const uint4* src = (const uint4*) (x16 + (size_t) idx[row] * C);
    uint4* dst = (uint4*) (y16 + (size_t) row * C);
    const int n4 = C >> 3;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dst[i] = src[i];
}

// out[idx[row]] += wr[row] * y[row] over rows (one expert's tokens: each token at most once, so no two rows collide)
__global__ void scatter_add_rows_kernel(const float* __restrict__ y, const int* __restrict__ idx,
                                        const float* __restrict__ wr, int C, float* __restrict__ out, int ld) {
    const int row = blockIdx.x;
    const float w = wr[row];
    const float4* src = (const float4*) (y + (size_t) row * C);
    float4* dst = (float4*) (out + (size_t) idx[row] * ld);
    for (int i = threadIdx.x; i < (C >> 2); i += blockDim.x) {
        const float4 a = src[i];
        float4 b = dst[i];
        b.x += w * a.x, b.y += w * a.y, b.z += w * a.z, b.w += w * a.w;
        dst[i] = b;
    }
}

// Y [T] rows (ldy) = rows [r0, r0 + n) of a dense matrix over X: a fused projection's part that takes its own input
Mat mat_rows(const Mat& m, int r0, int n) {   // the rows [r0, r0 + n) of a matrix
    Mat p = m;
    p.N = n;
    if (m.f16) p.f16 = m.f16 + (size_t) r0 * m.K;
    else if (m.q8) p.q8 = m.q8 + (size_t) r0 * m.K, p.q8d = m.q8d + (size_t) r0 * (m.K >> 5);
    else p.raw = m.raw + (size_t) r0 * (m.K / k::block_values(m.type)) * k::block_bytes(m.type);
    return p;
}
void matmul_rows(const Mat& m, int r0, int n, const float* x32, const uint16_t* x16, int T, float* Y, int ldy) {
    matmul(mat_rows(m, r0, n), x32, x16, T, Y, ldy);
}

// A looped layer's message gates over a chunk of tokens.  A gate is two matrix products - g = D [rms(x); rms(m)],
// then F g and G g - and k::loop_gates' kernels take them a token at a time: over a prefill chunk that was a quarter
// of the prefill.  Here the products go through cuBLAS and the kernels are what is left around them.
constexpr int kLoopGemmMin = 32;   // tokens from which the products are cuBLAS's
void loop_gates_chunk(Work& W, const k::LoopGate* gt, int n, int T, int C, int R, float eps) {
    if (T < kLoopGemmMin) {
        k::loop_gates(gt, n, T, C, R, eps, W.lp_g, g_s);
        return;
    }
    const float one = 1.0f, zero = 0.0f;
    float *a1 = W.lp_a, *a2 = a1 + (size_t) T * R, *g = a2 + (size_t) T * R;
    float *f = W.lp_f, *gn = f + (size_t) T * C;
    // (the gates' FP16 weights as F32 for cuBLAS's SGEMMs: D [R][2 C], F, G [R][C], a gate at a time)
    static float* w32 = nullptr;
    static size_t w32_n = 0;
    const size_t wn = (size_t) R * 2 * C + (size_t) 2 * R * C;
    if (wn > w32_n) {
        if (w32) cudaFree(w32);
        w32 = dalloc<float>(w32_n = wn);
    }
    float *Dw = w32, *Fw = Dw + (size_t) R * 2 * C, *Gw = Fw + (size_t) R * C;
    for (int j = 0; j < n; ++j) {
        k::f16_to_f32(gt[j].D, (size_t) R * 2 * C, Dw, g_s);
        k::f16_to_f32(gt[j].F, (size_t) R * C, Fw, g_s);
        k::f16_to_f32(gt[j].G, (size_t) R * C, Gw, g_s);
        // a1 [T][R] = x D_x^T, a2 = m D_m^T (D [R][2 C]: its rows' halves)
        CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, Dw, 2 * C, gt[j].x, C, &zero, a1, R));
        CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, Dw + C, 2 * C, gt[j].m, C, &zero, a2, R));
        k::loop_gate_mix(gt[j].x, gt[j].m, a1, a2, T, C, R, eps, g, g_s);
        // f [T][C] = g F, gn = g G (F, G [R][C])
        CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, C, T, R, &one, Fw, C, g, R, &zero, f, C));
        CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, C, T, R, &one, Gw, C, g, R, &zero, gn, C));
        k::loop_gate_blend(gt[j].x, gt[j].m, f, gn, (size_t) T * C, gt[j].y, g_s);
    }
}

// One layer's attention + post-attention conv + residual into r_out ([T][C]).
// h_in ([T][C]) is unmodified and acts as the residual input; when l % c.block == 0, it is saved to blk(nb).
// msg: a looped core's later pass - the layer's messages [4][T][C] (q's, k's, v's, the MoE's), which gate its q, k and
// v inputs before their norms (so a sliding layer too projects each from its own input).
int g_attn_mma = 1;         // a token's attention on the tensor cores (IK2_ATTN_MMA=0: the scalar kernels; 2: the
                            // layers without a window only)
int g_mma_split = 1;        // its blocks a K / V head (k::attn_decode_mma_blocks: the device filled once)
void run_layer_attn(const Model& M, Layer& Ly, int l, int T, int pos0, const float* h_in, const int* pos_p, int nb,
                    size_t blk_off, float* r_out, Work& W, const float* msg = nullptr) {
    const Config& c = M.c;
    const int C = c.C;
    const size_t TC = (size_t) T * C;
    auto blk = [&](int i) { return W.blocks + ((size_t) i * W.blk_ld + blk_off) * C; };

    const bool need_x32 = g_dense_q8 && (T == 1 || (T <= 8 && g_q8_t));   // (matmul's planar Q8 products take F32)
    const uint16_t* xk16 = W.x16;
    const uint16_t* xv16 = W.x16;
    const float *in_q = h_in, *in_k = h_in, *in_v = h_in;
    if (msg) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = h_in;
        k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, W.lp_in, g_s);
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
        float* y32[3] = {W.xq, W.xk, W.xv};
        k::LoopGate gt[3];
        for (int j = 0; j < 3; ++j)
            gt[j] = {W.lp_in + (size_t) (Ly.n_res_q == 3 ? j : 0) * TC, msg + (size_t) j * TC, Ly.lp_down[j],
                     Ly.lp_forget[j], Ly.lp_gain[j], y32[j]};
        loop_gates_chunk(W, gt, 3, T, C, c.rank, 1e-6f);
        for (int j = 0; j < 3; ++j)
            k::rms_f16(y32[j], nw[j], T, C, c.eps, y16[j], need_x32 ? y32[j] : nullptr, g_s);
        in_q = W.xq, in_k = W.xk, in_v = W.xv;
        xk16 = W.xk16, xv16 = W.xv16;
    } else if (l > 0) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = h_in;
        k::RmsOuts outs{};
        if (Ly.sliding) {
            outs.w[0] = Ly.attn_norm;
            outs.y16[0] = W.x16;
            outs.y32[0] = need_x32 ? W.xq : nullptr;
            k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            in_q = in_k = in_v = W.xq;
        } else {
            outs.w[0] = Ly.q_in_norm, outs.w[1] = Ly.k_in_norm, outs.w[2] = Ly.v_in_norm;
            outs.y16[0] = W.x16, outs.y16[1] = W.xk16, outs.y16[2] = W.xv16;
            if (need_x32) outs.y32[0] = W.xq, outs.y32[1] = W.xk, outs.y32[2] = W.xv;
            k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            in_q = W.xq, in_k = W.xk, in_v = W.xv;
            xk16 = W.xk16, xv16 = W.xv16;
        }
    } else if (Ly.sliding) {
        k::rms_f16(in_q, Ly.attn_norm, T, C, c.eps, W.x16, need_x32 ? W.xq : nullptr, g_s);
        in_q = in_k = in_v = W.xq;
    } else {
        k::rms_f16(in_q, Ly.q_in_norm, T, C, c.eps, W.x16, need_x32 ? W.xq : nullptr, g_s);
        k::rms_f16(in_k, Ly.k_in_norm, T, C, c.eps, W.xk16, need_x32 ? W.xk : nullptr, g_s);
        k::rms_f16(in_v, Ly.v_in_norm, T, C, c.eps, W.xv16, need_x32 ? W.xv : nullptr, g_s);
        in_q = W.xq, in_k = W.xk, in_v = W.xv;
        xk16 = W.xk16, xv16 = W.xv16;
    }
    const int HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel;
    float *q_p = W.q, *k_p = W.kk, *v_p = W.vv, *r_p = W.rr, *gate_p = W.gate, *sink_p = W.sink;
    float *kd_p = W.dyn, *vd_p = W.dyn;
    if (Ly.sliding) {
        if (Ly.wqkvrg.N > 0) {
            if (msg) {   // the fused rows by the input they take: q | k | v | r, gate (and the sink)
                const int N = Ly.wqkvrg.N;
                matmul_rows(Ly.wqkvrg, 0, HD, in_q, W.x16, T, W.qkv, N);
                matmul_rows(Ly.wqkvrg, HD, KD, in_k, xk16, T, W.qkv + HD, N);
                matmul_rows(Ly.wqkvrg, HD + KD, KD, in_v, xv16, T, W.qkv + HD + KD, N);
                matmul_rows(Ly.wqkvrg, HD + 2 * KD, N - HD - 2 * KD, in_q, W.x16, T, W.qkv + HD + 2 * KD, N);
            } else {
                matmul(Ly.wqkvrg, in_q, W.x16, T, W.qkv, Ly.wqkvrg.N);
            }
            const bool has_sink = !g_dense_q8;
            if (T == 1) {
                q_p = W.qkv;
                k_p = q_p + HD;
                v_p = k_p + KD;
                r_p = v_p + KD;
                gate_p = r_p + HR;
                if (has_sink) sink_p = gate_p + HD;
            } else {
                if (has_sink)
                    unpack_segments(W.qkv, T,
                                    {{W.q, HD}, {W.kk, KD}, {W.vv, KD}, {W.rr, HR}, {W.gate, HD}, {W.sink, c.H}}, g_s);
                else
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.kk, KD}, {W.vv, KD}, {W.rr, HR}, {W.gate, HD}}, g_s);
            }
            if (!has_sink) gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        } else {
            matmul(Ly.wq, in_q, W.x16, T, q_p, HD);
            matmul(Ly.wk, in_k, xk16, T, k_p, KD);
            matmul(Ly.wv, in_v, xv16, T, v_p, KD);
            matmul(Ly.wr, in_q, W.x16, T, r_p, HR);
            matmul(Ly.wgate, in_q, W.x16, T, gate_p, HD);
            gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        }
        conv_rec(Ly, 0, k_p, Ly.h_k, T, KD);
        conv_rec(Ly, 1, v_p, Ly.h_v, T, KD);
        k::conv4(k_p, Ly.h_k, Ly.sc_k, nullptr, 0, T, KD, W.kconv, g_s);
        k::conv4(v_p, Ly.h_v, Ly.sc_v, nullptr, 0, T, KD, W.vconv, g_s);
    } else {
        if (Ly.wqrg.N > 0) {
            matmul(Ly.wqrg, in_q, W.x16, T, W.qkv, Ly.wqrg.N);
            const bool has_sink = !g_dense_q8;
            if (T == 1) {
                q_p = W.qkv;
                r_p = q_p + HD;
                gate_p = r_p + HR;
                if (has_sink) sink_p = gate_p + HD;
            } else {
                if (has_sink)
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.rr, HR}, {W.gate, HD}, {W.sink, c.H}}, g_s);
                else
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.rr, HR}, {W.gate, HD}}, g_s);
            }
            if (!has_sink) gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        } else {
            matmul(Ly.wq, in_q, W.x16, T, q_p, HD);
            matmul(Ly.wr, in_q, W.x16, T, r_p, HR);
            matmul(Ly.wgate, in_q, W.x16, T, gate_p, HD);
            gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        }
        if (!g_dense_q8) {
            const int off_k = T == 1 ? Ly.wqrg.N : 0;
            const int off_v = T == 1 ? Ly.wqrg.N + Ly.wk_dyn.N : 0;
            matmul(Ly.wk_dyn, in_k, xk16, T, W.qkv + off_k, Ly.wk_dyn.N);
            if (T == 1) {
                k_p = W.qkv + off_k;
                kd_p = k_p + KD;
            } else {
                unpack_segments(W.qkv + off_k, T, {{W.kk, KD}, {W.dyn, c.KV * 4}}, g_s);
            }
            conv_rec(Ly, 0, k_p, Ly.h_k, T, KD);
            k::conv4(k_p, Ly.h_k, Ly.sc_k, kd_p, c.KV, T, KD, W.kconv, g_s);

            matmul(Ly.wv_dyn, in_v, xv16, T, W.qkv + off_v, Ly.wv_dyn.N);
            if (T == 1) {
                v_p = W.qkv + off_v;
                vd_p = v_p + KD;
            } else {
                unpack_segments(W.qkv + off_v, T, {{W.vv, KD}, {W.dyn, c.KV * 4}}, g_s);
            }
            conv_rec(Ly, 1, v_p, Ly.h_v, T, KD);
            k::conv4(v_p, Ly.h_v, Ly.sc_v, vd_p, c.KV, T, KD, W.vconv, g_s);
        } else {
            matmul(Ly.wk, in_k, xk16, T, k_p, KD);
            gemm16(Ly.sc_k_dyn16, c.KV * 4, C, xk16, T, kd_p, c.KV * 4);
            conv_rec(Ly, 0, k_p, Ly.h_k, T, KD);
            k::conv4(k_p, Ly.h_k, Ly.sc_k, kd_p, c.KV, T, KD, W.kconv, g_s);

            matmul(Ly.wv, in_v, xv16, T, v_p, KD);
            gemm16(Ly.sc_v_dyn16, c.KV * 4, C, xv16, T, vd_p, c.KV * 4);
            conv_rec(Ly, 1, v_p, Ly.h_v, T, KD);
            k::conv4(v_p, Ly.h_v, Ly.sc_v, vd_p, c.KV, T, KD, W.vconv, g_s);
        }
    }
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f;
    const float v_scale = 32.0f;
    const float k_inv = 1.0f / k_scale, v_inv = 1.0f / v_scale;
    if (g_kv_i8) {
        k::prep_qkv_r_i8(q_p, Ly.q_clip, W.kconv, (int8_t*) Ly.kc8, W.vconv, (int8_t*) Ly.vc8, r_p, Ly.r_clip, T, c.H,
                         c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha,
                         c.n_floor, pos0, Ly.kv_cap, g_s);
    } else if (g_kv_fp8) {
        k::prep_qkv_r_fp8(q_p, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, Ly.r_clip, T, c.H, c.KV, c.D, c.d_rel,
                          c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos0,
                          Ly.kv_cap, k_scale, v_scale, g_s);
    } else {
        k::prep_qkv_r_f16(q_p, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, Ly.r_clip, T, c.H, c.KV, c.D, c.d_rel,
                          c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos0,
                          Ly.kv_cap, g_s);
    }

    const int extent = Ly.sliding ? c.extent_swa : c.extent;
    const int window = Ly.sliding ? c.window : (1 << 30);
    const int k0 = Ly.sliding ? std::max(0, pos0 - c.window + 1) : 0;
    const int nk = pos0 + T - k0;
    const float one = 1.0f, zero = 0.0f;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const int attn_kv_cap = Ly.sliding ? Ly.kv_cap : 0;
    const bool mma1 = T == 1 && (g_attn_mma == 1 || (g_attn_mma == 2 && !Ly.sliding));
    if (mma1) {   // a token's attention on the tensor cores (q in FP16; a K / V head's query heads share its tiles)
        const int ns = Ly.sliding ? std::min(g_mma_split, std::max(1, (c.window + 63) / 64)) : g_mma_split;
        k::to_f16(q_p, (size_t) HD, W.x16, g_s);
        if (g_kv_i8)
            k::attn_decode_mma_i8(W.x16, r_p, sink_p, (const int8_t*) Ly.kc8, (const int8_t*) Ly.vc8, attn_kv_cap, c.H,
                                  c.KV, nullptr, pos0, ns, qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o,
                                  nullptr, nullptr, g_s);
        else if (g_kv_fp8)
            k::attn_decode_mma_fp8(W.x16, r_p, sink_p, Ly.kc8, Ly.vc8, attn_kv_cap, c.H, c.KV, nullptr, pos0, ns,
                                   qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, nullptr,
                                   nullptr, g_s);
        else
            k::attn_decode_mma_f16(W.x16, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, nullptr, pos0, ns, qk_scale,
                                   Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, nullptr, nullptr, g_s);
    } else if (T == 1) {
        if (g_kv_i8) {
            k::attn_decode_i8(q_p, r_p, sink_p, (const int8_t*) Ly.kc8, (const int8_t*) Ly.vc8, attn_kv_cap, c.H, c.KV,
                              c.D, k0, nk, pos0, qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else if (g_kv_fp8) {
            k::attn_decode_fp8(q_p, r_p, sink_p, Ly.kc8, Ly.vc8, attn_kv_cap, c.H, c.KV, c.D, k0, nk, pos0,
                               qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else {
            k::attn_decode_f16(q_p, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, k0, nk, pos0, qk_scale,
                               Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
    } else if (T <= 8 && g_group && g_group_attn) {
        if (g_kv_i8) {
            k::attn_multi_i8(q_p, r_p, sink_p, (const int8_t*) Ly.kc8, (const int8_t*) Ly.vc8, attn_kv_cap, T, c.H, c.KV,
                             k0, nk, pos0, qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else if (g_kv_fp8) {
            k::attn_multi_fp8(q_p, r_p, sink_p, Ly.kc8, Ly.vc8, attn_kv_cap, T, c.H, c.KV, k0, nk, pos0,
                              qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else {
            k::attn_multi_f16(q_p, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, T, c.H, c.KV, k0, nk, pos0, qk_scale,
                              Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
    } else {
        k::to_f16(q_p, (size_t) T * HD, W.x16, g_s);
        uint16_t* lg16 = W.lg16;
        const int GQA = c.H / c.KV;
        // softmax rows (with the relative bias and the sink) -> lg16: the bias as one small GEMM (K = d_rel) first
        auto attn_probs = [&](int Tq, int nk_s, int ts, int k0_s) {
            if (W.relb && extent <= W.relb_ext) {
                const float one_f = 1.0f, zero_f = 0.0f;
                CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, extent, Tq * c.H, c.d_rel, &one_f, Ly.rel_proj,
                                CUDA_R_32F, c.d_rel, r_p + (size_t) ts * HR, CUDA_R_32F, c.d_rel, &zero_f, W.relb,
                                CUDA_R_32F, extent, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                k::attn_probs_bias_f16(W.lg, lg16, c.H, Tq, nk_s, pos0 + ts, k0_s, qk_scale, W.relb, extent, window,
                                       sink_p + (size_t) ts * c.H, g_s);
            } else {
                k::attn_probs_f16(W.lg, lg16, c.H, Tq, nk_s, pos0 + ts, k0_s, qk_scale, r_p + (size_t) ts * HR,
                                  c.d_rel, Ly.rel_proj, extent, window, sink_p + (size_t) ts * c.H, g_s);
            }
        };
        if (g_flash && c.D == 128 && W.relb && extent <= W.relb_ext) {
            if (!g_n_sm) CU(cudaDeviceGetAttribute(&g_n_sm, cudaDevAttrMultiProcessorCount, Ly.dev));
            for (int ts = 0; ts < T; ts += Work::kRelbRows) {
                const int Tq = std::min(Work::kRelbRows, T - ts);
                const float one_f = 1.0f, zero_f = 0.0f;
                CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, extent, Tq * c.H, c.d_rel, &one_f, Ly.rel_proj,
                                CUDA_R_32F, c.d_rel, r_p + (size_t) ts * HR, CUDA_R_32F, c.d_rel, &zero_f, W.relb,
                                CUDA_R_32F, extent, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                const int k_lo = Ly.sliding ? std::max(0, pos0 + ts - c.window + 1) : 0;
                const int ns = k::attn_flash_splits(Tq, c.H, pos0 + ts + Tq - k_lo, g_n_sm);
                if (g_kv_i8)
                    k::attn_flash_i8(W.x16 + (size_t) ts * HD, HD, (const int8_t*) Ly.kc8, (const int8_t*) Ly.vc8,
                                     attn_kv_cap, c.H, c.KV, c.D, Tq, pos0 + ts, qk_scale, W.relb, extent, window,
                                     sink_p + (size_t) ts * c.H, W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
                else if (g_kv_fp8)
                    k::attn_flash_fp8(W.x16 + (size_t) ts * HD, HD, Ly.kc8, Ly.vc8, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale * k_inv, v_inv, W.relb, extent, window,
                                      sink_p + (size_t) ts * c.H, W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
                else
                    k::attn_flash_f16(W.x16 + (size_t) ts * HD, HD, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale, W.relb, extent, window, sink_p + (size_t) ts * c.H,
                                      W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
            }
        } else
        for (int ts = 0; ts < T;) {
            const int k0_s = Ly.sliding ? std::max(0, pos0 + ts - c.window + 1) : 0;
            const int nk_ub = pos0 + T - k0_s;
            int Tq = std::max(1, std::min(T - ts, (int) (W.lg_cap / ((size_t) c.H * nk_ub))));
            if (W.relb) Tq = std::min(Tq, Work::kRelbRows);
            const int nk_s = pos0 + ts + Tq - k0_s;
            const long long qn = (long long) Tq * nk_s;
            if (kv_b8()) {
                for (int k_off = 0; k_off < nk_s; k_off += W.kv_tmp_cap) {
                    const int nk_c = std::min(W.kv_tmp_cap, nk_s - k_off);
                    kv_unpack(Ly, attn_kv_cap, k0_s + k_off, nk_c, KD, k_inv, v_inv, W.kv_k16,
                                            W.kv_v16, g_s);
                    for (int g = 0; g < c.KV; ++g) {
                        const uint16_t* Kp = W.kv_k16 + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_T, CUBLAS_OP_N, nk_c, Tq, c.D, &one, Kp, CUDA_R_16F, KD, 0,
                            W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                            W.lg + (size_t) g * GQA * qn + k_off, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                }
                attn_probs(Tq, nk_s, ts, k0_s);
                for (int k_off = 0; k_off < nk_s; k_off += W.kv_tmp_cap) {
                    const int nk_c = std::min(W.kv_tmp_cap, nk_s - k_off);
                    if (nk_s > W.kv_tmp_cap) {
                        kv_unpack(Ly, attn_kv_cap, k0_s + k_off, nk_c, KD, k_inv, v_inv,
                                                W.kv_k16, W.kv_v16, g_s);
                    }
                    const float beta = k_off == 0 ? zero : one;
                    for (int g = 0; g < c.KV; ++g) {
                        const uint16_t* Vp = W.kv_v16 + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, nk_c, &one, Vp, CUDA_R_16F, KD, 0,
                            lg16 + (size_t) g * GQA * qn + k_off, CUDA_R_16F, nk_s, qn, &beta,
                            W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    }
                }
                if (g_kv_i8 && k::g_v_rot && c.D == 128)   // (k8v8: the rows' output rotated back)
                    k::wht_rows(W.o + (size_t) ts * HD, Tq * c.H, g_s);
            } else {
                const int start = attn_kv_cap > 0 ? (k0_s % attn_kv_cap) : k0_s;
                const int n1 = attn_kv_cap > 0 ? std::min(nk_s, attn_kv_cap - start) : nk_s;
                const int n2 = nk_s - n1;
                for (int g = 0; g < c.KV; ++g) {
                    const uint16_t* Kp1 = Ly.kc + (size_t) start * KD + (size_t) g * c.D;
                    CB(cublasGemmStridedBatchedEx(
                        g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n1, Tq, c.D, &one, Kp1, CUDA_R_16F, KD, 0,
                        W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                        W.lg + (size_t) g * GQA * qn, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT));
                    if (n2 > 0) {
                        const uint16_t* Kp2 = Ly.kc + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n2, Tq, c.D, &one, Kp2, CUDA_R_16F, KD, 0,
                            W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                            W.lg + (size_t) g * GQA * qn + n1, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                }
                attn_probs(Tq, nk_s, ts, k0_s);
                for (int g = 0; g < c.KV; ++g) {
                    const uint16_t* Vp1 = Ly.vc + (size_t) start * KD + (size_t) g * c.D;
                    CB(cublasGemmStridedBatchedEx(
                        g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, n1, &one, Vp1, CUDA_R_16F, KD, 0,
                        lg16 + (size_t) g * GQA * qn, CUDA_R_16F, nk_s, qn, &zero,
                        W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    if (n2 > 0) {
                        const uint16_t* Vp2 = Ly.vc + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, n2, &one, Vp2, CUDA_R_16F, KD, 0,
                            lg16 + (size_t) g * GQA * qn + n1, CUDA_R_16F, nk_s, qn, &one,
                            W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    }
                }
            }
            ts += Tq;
        }
    }
    k::gate_mul_f16(W.o, gate_p, (size_t) T * HD, W.x16, g_dense_q8 && (T == 1 || (T <= 8 && g_q8_t)), g_s);
    matmul(Ly.wo, W.o, W.x16, T, W.tmp, C);
    conv_rec(Ly, 2, W.tmp, Ly.h_attn, T, C);
    if (l % c.block == 0) {
        CU(cudaMemcpyAsync(blk(nb), h_in, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        k::conv4(W.tmp, Ly.h_attn, Ly.sc_attn, nullptr, 0, T, C, r_out, g_s);
    } else {
        k::conv4_res(W.tmp, Ly.h_attn, Ly.sc_attn, nullptr, 0, T, C, h_in, r_out, g_s);
    }
}

// A layer-major prefill pass of many tokens uses nearly every expert of a layer, so the ones not in VRAM can be
// copied in before the layer's routing is known: while the previous layer's experts and this layer's attention run.
// The copy stream then has no idle gap between layers. Slots are taken at the current tick, which keeps the previous
// layer's slots (still being computed) from being evicted; only done when the pool holds both layers with room.
void prefetch_layer(Model& M, int l, Work& W, ExpertCache& EC, int T, const Work* Wpop = nullptr, int own = -1, int ahead = 1) {
    const int pr = ((l % Work::kPfRing) + Work::kPfRing) % Work::kPfRing;
    W.pf_layers[pr] = -1;
    if (getenv("IK2_PREFETCH") && !std::atoi(getenv("IK2_PREFETCH"))) return;
    if (!g_overlap || g_prof || l >= M.c.n_layer || (!dual() && M.L[l].dev != g_cur)) return;
    const Work& U = Wpop ? *Wpop : W;   // (whose picks so far: under EP, GPU 0's)
    const int E = M.c.E, pi = EC.layer_pool[l];
    if (pi < 0 || EC.pools[pi].n_active < (ahead + 1) * E + 64) return;
    Layer& Ly = M.L[l];
    const uint64_t t = EC.tick + (uint64_t) ahead;   // (the tick its layer will run in: kept till then)
    W.pf_bs[pr].assign(E, -1);
    // Only the experts this pass will likely use: P(used) = 1 - (1 - p_e)^(T K) from the prefills so far (p_e the
    // expert's share of the layer's tokens); with little history yet, all of them (IK2_PF_P: the threshold)
    const bool known = !U.use_tot.empty() && U.use_tot[l] >= (uint64_t) 16 * E;
    // (0.999, not 0.5: a request's tokens are alike, so the independent-picks estimate is far too high - at 0.5 a
    // fifth of the prefetched experts went unused; 500 / 800 new tokens 4.2 / 4.3 -> 3.5 / 3.6 s, 21.8k unchanged)
    const double thr = getenv("IK2_PF_P") ? std::atof(getenv("IK2_PF_P")) : 0.999;
    const double picks = (double) T * M.c.K;
    int n = 0;
    for (int e = 0; e < E; ++e) {
        if (EC.has(l, e) || (own >= 0 && ep_owner(l, e) != own)) continue;
        if (known) {
            const double pe = (U.use[(size_t) l * E + e] + 0.5) / (U.use_tot[l] + 0.5 * E);
            if (1.0 - std::pow(1.0 - pe, picks) < thr) continue;
        }
        uint8_t* dst = EC.get(Ly, l, e, t, g_cs, nullptr, false, true);
        EC.copy_in(Ly, e, dst, g_cs);
        W.pf_bs[pr][e] = n / Work::kMissBatch;
        if (++n % Work::kMissBatch == 0) CU(cudaEventRecord(W.pf_evs[pr][n / Work::kMissBatch - 1], g_cs));
    }
    if (n % Work::kMissBatch) CU(cudaEventRecord(W.pf_evs[pr][n / Work::kMissBatch], g_cs));
    W.pf_layers[pr] = l;
}

void run_prefill_moe(const Model& M, Layer& Ly, int l, int T, const uint16_t* xq16, const int* top, const float* w,
                     const float* r_in, float* h_out, Work& W, ExpertCache& EC, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C, F = c.F, S = T * c.K;
    const double te0 = now();
    if (g_seq_done + 1 < g_seq_next) {
        CU(cudaStreamSynchronize(g_cs));
        g_seq_done = g_seq_next - 1;
    }
    std::fill(W.cnt.begin(), W.cnt.end(), 0);
    for (int i = 0; i < S; ++i) ++W.cnt[top[i]];
    if (W.use.empty()) W.use.assign((size_t) c.n_layer * c.E, 0), W.use_tot.assign(c.n_layer, 0);
    const int prl = l % Work::kPfRing;
    const bool pf_here = W.pf_layers[prl] == l && (int) W.pf_bs[prl].size() == c.E;
    for (int e = 0; e < c.E; ++e) {
        W.use[(size_t) l * c.E + e] += W.cnt[e];
        const int k = W.cnt[e];
        if (k) ++g_cnt_hist[k == 1 ? 0 : k == 2 ? 1 : k < 8 ? 2 : 3];
        const bool pfd = pf_here && W.pf_bs[prl][e] >= 0;
        if (k) ++g_src_hist[pfd ? 1 : EC.has(l, e) ? 0 : 2];
        else if (pfd) ++g_src_hist[3];
    }
    W.use_tot[l] += S;
    W.start[0] = 0;
    for (int e = 0; e < c.E; ++e) W.start[e + 1] = W.start[e] + W.cnt[e];
    for (int e = 0; e < c.E; ++e) W.fill[e] = W.start[e];

    const int ps = l & 1;
    int* gidx = W.gidx_h[ps];
    int* inv = W.inv_h[ps];
    float* sg = W.sg_h[ps];
    float* su = W.su_h[ps];
    float* wt = W.wt_h[ps];
    float* wtr = W.wtr_h[ps];
    for (int i = 0; i < S; ++i) {
        const int e = top[i], row = W.fill[e]++;
        gidx[row] = i / c.K;
        inv[i] = row;
        sg[row] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
        su[row] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
        wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
        wtr[row] = wt[i];
    }

    // the experts on a GPU: d -1 all of them (one GPU), else the ones it owns (IK2_EP), into X.tmp
    auto exec = [&](int d, Work& X, ExpertCache& XC, const uint16_t* xin) {
        const uint64_t tick = ++XC.tick;
        const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
        X.hit_exps.clear();
        X.miss_exps.clear();
        const double tu = now();
        for (int e = 0; e < c.E; ++e) {
            if (!W.cnt[e] || (d >= 0 && ep_owner(l, e) != d)) {
                X.slot_ptrs[e] = nullptr;
                continue;
            }
            if (XC.has(l, e)) {
                X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
                X.hit_exps.push_back(e);
            } else {
                X.slot_ptrs[e] = nullptr;
                X.miss_exps.push_back(e);
            }
        }
        for (int e : X.miss_exps) X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);

        // the layer's routing tables first: the host-to-device copies share one copy engine in order, and these small
        // ones queued behind the misses' (megabytes each) held every expert's kernels back until the last miss landed
        CU(cudaMemcpyAsync(X.gidx, gidx, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.inv, inv, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.sg, sg, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.su, su, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.wtr, wtr, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemsetAsync(X.tmp, 0, (size_t) T * C * 4, g_s));
        const int n_miss = (int) X.miss_exps.size();
        const int n_batches = (n_miss + Work::kMissBatch - 1) / Work::kMissBatch;
        for (int b = 0; b < n_batches; ++b) {
            const int b0 = b * Work::kMissBatch, b1 = std::min(n_miss, b0 + Work::kMissBatch);
            for (int m = b0; m < b1; ++m) {
                const int e = X.miss_exps[m];
                XC.copy_in(Ly, e, X.slot_ptrs[e], cst);
            }
            if (cst != g_s) CU(cudaEventRecord(X.miss_ev[b], cst));
        }
        XC.t_up += now() - tu;
        g_pf.lap(g_pf.upload);


        // Each expert: its tokens gathered, gate/up, SwiGLU, down, then added into tmp with their weights. The experts run
        // in index order whatever is cached (each waiting for its copies), so the sums' order and the result do not
        // depend on the cache. Only [T][C] of scratch: no [T * K][C] copies of the inputs or the outputs.
        auto run_exp = [&](int e) {
            const int cnt_e = W.cnt[e], st_e = W.start[e];
            const uint8_t* sl = X.slot_ptrs[e];
            gather_f16_16_kernel<<<cnt_e, 256, 0, g_s>>>(xin, X.gidx + st_e, cnt_e, C, X.ex16);
            if (Ly.ty[0] == Ly.ty[1]) {
                k::decode_f16(Ly.ty[0], sl, (size_t) 2 * F * C, X.w16, g_s);
            } else {
                k::decode_f16(Ly.ty[0], sl, (size_t) F * C, X.w16, g_s);
                k::decode_f16(Ly.ty[1], sl + Ly.eb[0], (size_t) F * C, X.w16 + (size_t) F * C, g_s);
            }
            gemm16(X.w16, 2 * F, C, X.ex16, cnt_e, X.gu32, 2 * F);
            k::swiglu_cap_f32(X.gu32, cnt_e, F, X.sg + st_e, X.su + st_e, c.gate_cap, c.up_cap, X.h16, g_s);
            k::decode_f16(Ly.ty[2], sl + Ly.eb[0] + Ly.eb[1], (size_t) C * F, X.w16, g_s);
            gemm16(X.w16, C, F, X.h16, cnt_e, X.y32, C);
            scatter_add_rows_kernel<<<cnt_e, 256, 0, g_s>>>(X.y32, X.gidx + st_e, X.wtr + st_e, C, X.tmp, C);
        };
        const bool pf = X.pf_layers[l % Work::kPfRing] == l;
        X.miss_pos.assign(c.E, -1);
        for (int m = 0; m < n_miss; ++m) X.miss_pos[X.miss_exps[m]] = m;
        int pf_waited = -1, miss_waited = -1;
        for (int e = 0; e < c.E; ++e) {
            if (!W.cnt[e] || (d >= 0 && ep_owner(l, e) != d)) continue;
            if (pf && X.pf_bs[l % Work::kPfRing][e] >= 0 && X.pf_bs[l % Work::kPfRing][e] > pf_waited)
                CU(cudaStreamWaitEvent(g_s, X.pf_evs[l % Work::kPfRing][pf_waited = X.pf_bs[l % Work::kPfRing][e]], 0));
            if (X.miss_pos[e] >= 0 && cst != g_s && X.miss_pos[e] / Work::kMissBatch > miss_waited)
                CU(cudaStreamWaitEvent(g_s, X.miss_ev[miss_waited = X.miss_pos[e] / Work::kMissBatch], 0));
            run_exp(e);
        }
        X.pf_layers[l % Work::kPfRing] = -1;
    };
    if (g_ep) {   // GPU 1's experts on GPU 1 (the input over, its sums back), GPU 0's meanwhile
        CU(cudaEventRecord(g_ep_x, g_s));
        use_dev(1);
        Work& X1 = dev_work(1);
        CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
        CU(cudaMemcpyPeerAsync(X1.xq16_sup, 1, xq16, 0, (size_t) T * C * 2, g_s));
        exec(1, X1, dev_cache(1), X1.xq16_sup);
        CU(cudaEventRecord(g_ep_done, g_s));
        use_dev(0);
        exec(0, W, EC, xq16);
        CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
        CU(cudaMemcpyPeerAsync(W.y32, 0, X1.tmp, 1, (size_t) T * C * 4, g_s));
        k::add(W.tmp, W.y32, (size_t) T * C, W.tmp, g_s);
    } else if (g_tp) {
        // tensor-parallel experts: (1) each GPU its half of every expert's h into tp_h (GPU 1 with the input copied
        // over), (2) the halves swapped, (3) each GPU its half of the output rows, GPU 1's columns copied to GPU 0
        const int Fd = F / 2, Cd = C / 2;
        auto tp_gu = [&](int d, Work& X, ExpertCache& XC, const uint16_t* xin) {
            const uint64_t tick = ++XC.tick;
            const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
            X.hit_exps.clear();
            X.miss_exps.clear();
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) {
                    X.slot_ptrs[e] = nullptr;
                    continue;
                }
                if (XC.has(l, e)) {
                    X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
                    X.hit_exps.push_back(e);
                } else {
                    X.slot_ptrs[e] = nullptr;
                    X.miss_exps.push_back(e);
                }
            }
            for (int e : X.miss_exps) X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
            const int n_miss = (int) X.miss_exps.size();
            const int n_batches = (n_miss + Work::kMissBatch - 1) / Work::kMissBatch;
            for (int b = 0; b < n_batches; ++b) {
                const int b0 = b * Work::kMissBatch, b1 = std::min(n_miss, b0 + Work::kMissBatch);
                for (int m = b0; m < b1; ++m) XC.copy_in(Ly, X.miss_exps[m], X.slot_ptrs[X.miss_exps[m]], cst);
                if (cst != g_s) CU(cudaEventRecord(X.miss_ev[b], cst));
            }
            CU(cudaMemcpyAsync(X.gidx, gidx, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.sg, sg, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.su, su, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.wtr, wtr, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            size_t o0, b0g, o1, b1u;
            tp_part(Ly, d, 0, o0, b0g);
            tp_part(Ly, d, 1, o1, b1u);
            const bool pf = X.pf_layers[l % Work::kPfRing] == l;
            X.miss_pos.assign(c.E, -1);
            for (int m = 0; m < n_miss; ++m) X.miss_pos[X.miss_exps[m]] = m;
            int pf_waited = -1, miss_waited = -1;
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) continue;
                if (pf && X.pf_bs[l % Work::kPfRing][e] >= 0 && X.pf_bs[l % Work::kPfRing][e] > pf_waited)
                    CU(cudaStreamWaitEvent(g_s, X.pf_evs[l % Work::kPfRing][pf_waited = X.pf_bs[l % Work::kPfRing][e]], 0));
                if (X.miss_pos[e] >= 0 && cst != g_s && X.miss_pos[e] / Work::kMissBatch > miss_waited)
                    CU(cudaStreamWaitEvent(g_s, X.miss_ev[miss_waited = X.miss_pos[e] / Work::kMissBatch], 0));
                const int cnt_e = W.cnt[e], st_e = W.start[e];
                const uint8_t* sl = X.slot_ptrs[e];
                gather_f16_16_kernel<<<cnt_e, 256, 0, g_s>>>(xin, X.gidx + st_e, cnt_e, C, X.ex16);
                if (Ly.ty[0] == Ly.ty[1]) {
                    k::decode_f16(Ly.ty[0], sl, (size_t) 2 * Fd * C, X.w16, g_s);
                } else {
                    k::decode_f16(Ly.ty[0], sl, (size_t) Fd * C, X.w16, g_s);
                    k::decode_f16(Ly.ty[1], sl + b0g, (size_t) Fd * C, X.w16 + (size_t) Fd * C, g_s);
                }
                gemm16(X.w16, 2 * Fd, C, X.ex16, cnt_e, X.gu32, 2 * Fd);
                k::swiglu_cap_f32(X.gu32, cnt_e, Fd, X.sg + st_e, X.su + st_e, c.gate_cap, c.up_cap, X.h16, g_s);
                CU(cudaMemcpy2DAsync(X.tp_h + (size_t) st_e * F + (size_t) d * Fd, (size_t) F * 2, X.h16, (size_t) Fd * 2,
                                     (size_t) Fd * 2, cnt_e, cudaMemcpyDeviceToDevice, g_s));
            }
            X.pf_layers[l % Work::kPfRing] = -1;
            CU(cudaEventRecord(g_tp_h[d], g_s));
        };
        auto tp_dn = [&](int d, Work& X, const Work& O) {
            // the other GPU's half of h in, then this GPU's rows of down for every expert
            CU(cudaStreamWaitEvent(g_s, g_tp_h[1 - d], 0));
            CU(cudaMemcpy2DAsync(X.tp_h + (size_t) (1 - d) * Fd, (size_t) F * 2, O.tp_h + (size_t) (1 - d) * Fd,
                                 (size_t) F * 2, (size_t) Fd * 2, S, cudaMemcpyDefault, g_s));
            CU(cudaMemsetAsync(X.tmp, 0, (size_t) T * C * 4, g_s));
            size_t o, b0g, b1u;
            tp_part(Ly, d, 0, o, b0g);
            tp_part(Ly, d, 1, o, b1u);
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) continue;
                const int cnt_e = W.cnt[e], st_e = W.start[e];
                k::decode_f16(Ly.ty[2], X.slot_ptrs[e] + b0g + b1u, (size_t) Cd * F, X.w16, g_s);
                gemm16(X.w16, Cd, F, X.tp_h + (size_t) st_e * F, cnt_e, X.y32, Cd);
                scatter_add_rows_kernel<<<cnt_e, 256, 0, g_s>>>(X.y32, X.gidx + st_e, X.wtr + st_e, Cd,
                                                               X.tmp + (size_t) d * Cd, C);
            }
        };
        Work& X1 = dev_work(1);
        CU(cudaEventRecord(g_ep_x, g_s));
        use_dev(1);
        CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
        CU(cudaMemcpyPeerAsync(X1.xq16_sup, 1, xq16, 0, (size_t) T * C * 2, g_s));
        tp_gu(1, X1, dev_cache(1), X1.xq16_sup);
        use_dev(0);
        tp_gu(0, W, EC, xq16);
        use_dev(1);
        tp_dn(1, X1, W);
        CU(cudaEventRecord(g_ep_done, g_s));
        use_dev(0);
        tp_dn(0, W, X1);
        CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
        CU(cudaMemcpy2DAsync(W.tmp + Cd, (size_t) C * 4, X1.tmp + Cd, (size_t) C * 4, (size_t) Cd * 4, T,
                             cudaMemcpyDefault, g_s));
    } else {
        exec(-1, W, EC, xq16);
    }
    g_pf.lap(g_pf.experts);
    t_exp += now() - te0;
    conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
    k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, r_in, h_out, g_s);
}

void run_head(const Model& M, int T, bool all_logits, OutMode out_mode, int nb, const float* h_in, Work& W,
              std::vector<float>& logits_out, std::vector<int>* argmax_out) {
    const Config& c = M.c;
    const int C = c.C;
    const size_t TC = (size_t) T * C;
    g_pf.start();
    if (out_mode != OutMode::kNone || g_taps) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + (size_t) i * W.blk_ld * C;
        st.st[nb] = h_in;
        if (out_mode != OutMode::kNone && (all_logits || T == 1)) {
            k::RmsOuts outs{};
            outs.w[0] = M.out_norm;
            outs.y16[0] = W.x16;
            outs.y32[0] = (g_taps || g_dense_q8) ? W.xq : nullptr;
            k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            if (g_taps)
                CU(cudaMemcpyAsync(g_taps + (size_t) (g_taps_n - 1) * g_taps_ld * C, W.xq, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        } else {
            k::attn_res(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, W.tmp, g_s);
            k::rms(W.tmp, M.out_norm, T, C, c.eps, W.xq, g_s);
            if (g_taps)
                CU(cudaMemcpyAsync(g_taps + (size_t) (g_taps_n - 1) * g_taps_ld * C, W.xq, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            if (out_mode != OutMode::kNone) k::to_f16(W.xq + (size_t) (T - 1) * C, (size_t) C, W.x16, g_s);
        }
        if (out_mode != OutMode::kNone) {
            const int t0 = all_logits ? 0 : T - 1, nt = T - t0;
            if (out_mode == OutMode::kArgmax) {
                argmax_out->resize(nt);
                for (int i = 0; i < nt; i += 16) {
                    const int n = std::min(16, nt - i);
                    matmul(M.out_w, W.xq + (size_t) (t0 + i) * C, W.x16 + (size_t) i * C, n, W.logits, c.n_rows);
                    if (n == 1) k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d + i, g_s);
                    else k::argmax_rows(W.logits, c.n_rows, n, c.n_vocab, W.argmax_d + i, g_s);
                }
                if (g_tg_capturing) return;   // (a token graph: the argmax read after its launch)
                CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, (size_t) nt * 4, cudaMemcpyDeviceToHost, g_s));
                CU(cudaStreamSynchronize(g_s));
                for (int i = 0; i < nt; ++i) (*argmax_out)[i] = W.argmax_h[i];
            } else {
                for (int i = 0; i < nt; i += 16) {   // (W.logits holds 16 rows)
                    const int n = std::min(16, nt - i);
                    matmul(M.out_w, W.xq + (size_t) (t0 + i) * C, W.x16 + (size_t) i * C, n, W.logits, c.n_rows);
                    k::softcap(W.logits, (size_t) n * c.n_rows, c.logit_cap, g_s);
                    const size_t at = logits_out.size();
                    logits_out.resize(at + (size_t) n * c.n_vocab);
                    CU(cudaMemcpy2DAsync(logits_out.data() + at, (size_t) c.n_vocab * 4, W.logits,
                                         (size_t) c.n_rows * 4, (size_t) c.n_vocab * 4, n, cudaMemcpyDeviceToHost,
                                         g_s));
                }
                CU(cudaStreamSynchronize(g_s));
            }
        }
    }
    g_pf.lap(g_pf.rest);
}

// ---- the resident token path
// When the cache holds every expert of the model (Inkling-2-Mini on a 24 GB card) none is ever missing or evicted, so
// an expert's blocks are found on the device (Layer::tab_blob) and a token needs nothing from the host between its
// embedding and its logits: no routing round trip a layer, no table upload.  The position is read on the device too
// (W.pos), so a token's launches do not change from one token to the next and are replayed as one CUDA graph.  The
// dense weights are the GGUF's blocks (IK2_DENSE=q8, the default when the experts fit), one product a launch.
bool g_fast = false;        // set at load: every expert resident, one GPU, the dense weights as blocks (IK2_FAST=0: no)
bool g_fast_want = false;
bool g_graph = true;        // IK2_GRAPH=0: the resident path's launches one by one
struct FastGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool disabled = false;
};
FastGraph g_fgraph[4];      // by OutMode

// one token's launches: its embedding row in W.emb_h and its position in W.pos_h[0] (pinned host; copied first)
void fast_token_launches(Model& M, Work& W, OutMode out_mode) {
    const Config& c = M.c;
    const int C = c.C, HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel;
    const size_t Cb = (size_t) C * 4;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f, v_scale = 32.0f;
    // the attention's key splits: fixed, so that the launches are the same at every position
    const int P_sl = std::min(128, std::max(1, (c.window + 63) / 64)), P_full = 128;
    CU(cudaMemcpyAsync(W.tmp, W.emb_h, Cb, cudaMemcpyHostToDevice, g_s));
    k::rms(W.tmp, M.embd_norm, 1, C, c.eps, W.h, g_s);
    CU(cudaMemcpyAsync(W.pos, W.pos_h, 4, cudaMemcpyHostToDevice, g_s));
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    int nb = 0, nb_pre = 0, n_bank = 0;
    for (int vis = 0; vis < c.n_visit; ++vis) {
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        // a looped core's passes (as forward()'s)
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
            } else {
                n_bank = 0;
                for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * C, blk(i), Cb, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * C, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
                ++n_bank;
                CU(cudaMemcpyAsync(W.h, W.lp_pre, Cb, cudaMemcpyDeviceToDevice, g_s));
                nb = nb_pre;
                // the pass's messages, every core layer's at once: the bank stays as it is until the pass ends, and a
                // layer's launch of its own was 7 us on one SM (a job a block: the same sums, 24 launches less a token)
                k::ResStates st{};
                for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * C;
                k::attn_res_jobs(st, n_bank, M.lp_query_all, 2, 2 * (c.core1 - c.core0), C, 1e-6f, W.lp_msg_all, g_s);
            }
        }
        if (M.V[vis].pass > 0) msg = W.lp_msg_all + (size_t) (l - c.core0) * 4 * C;   // (q, k | v, MoE)
        // the attention's inputs, normed, in F32: W.xq, W.xk, W.xv (sep: q, k and v each have their own)
        float* x3[3] = {W.xq, W.xk, W.xv};
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        bool sep = !Ly.sliding;
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.h;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, 1, C, 1e-6f, W.lp_in, g_s);
                k::LoopGate gt[3];
                for (int j = 0; j < 3; ++j)
                    gt[j] = {W.lp_in + (size_t) (Ly.n_res_q == 3 ? j : 0) * C, msg + (size_t) j * C, Ly.lp_down[j],
                             Ly.lp_forget[j], Ly.lp_gain[j], x3[j]};
                k::loop_gates(gt, 3, 1, C, c.rank, 1e-6f, W.lp_g, g_s);
                k::rms_multi(k::RmsJobs{{x3[0], x3[1], x3[2]}, {nw[0], nw[1], nw[2]}, {x3[0], x3[1], x3[2]}}, 3, C, c.eps,
                             g_s);
                sep = true;
            } else if (l > 0) {
                k::RmsOuts outs{};
                uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
                for (int j = 0; j < Ly.n_res_q; ++j) outs.w[j] = nw[j], outs.y16[j] = y16[j], outs.y32[j] = x3[j];
                k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, 1, C, 1e-6f, c.eps, outs, g_s);
            } else {
                for (int j = 0; j < (sep ? 3 : 1); ++j) k::rms(W.h, nw[j], 1, C, c.eps, x3[j], g_s);
            }
        }
        // the projections, straight over the blocks
        float *q_p, *k_p, *v_p, *r_p, *gate_p, *kd_p = nullptr, *vd_p = nullptr;
        // (k's and v's short convs ride their products' launches - mv1c - and the sinks the clips' launch below)
        if (Ly.sliding) {
            const Mat& m = Ly.wqkvrg;   // q | k | v | r | gate
            q_p = W.qkv, k_p = q_p + HD, v_p = k_p + KD, r_p = v_p + KD, gate_p = r_p + HR;
            if (!sep) {
                mv1c(m, W.xq, W.qkv, k::MvConv{HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                     k::MvConv{HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv});
            } else if (m.q8) {   // (q, k and v each with its input: k's and v's rows take theirs, one launch)
                k::mv_q8p_conv(m.q8, m.q8d, m.N, m.K, W.xq, W.qkv,
                               k::MvConv{HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                               k::MvConv{HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv}, g_s,
                               k::MvAlt{W.xk, HD, KD}, k::MvAlt{W.xv, HD + KD, KD});
            } else {
                mv1(mat_rows(m, 0, HD), W.xq, W.qkv);
                mv1c(mat_rows(m, HD, KD), W.xk, k_p, k::MvConv{0, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv});
                mv1c(mat_rows(m, HD + KD, KD), W.xv, v_p,
                     k::MvConv{0, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv});
                mv1(mat_rows(m, HD + 2 * KD, m.N - HD - 2 * KD), W.xq, W.qkv + HD + 2 * KD);
            }
        } else {
            const Mat& m = Ly.wqrg;     // q | r | gate
            mv1(m, W.xq, W.qkv);
            q_p = W.qkv, r_p = q_p + HD, gate_p = r_p + HR;
            k_p = W.kk, v_p = W.vv, kd_p = W.dyn, vd_p = W.dyn + (size_t) c.KV * 4;
            k::mv_f16(Ly.sc_k_dyn16, c.KV * 4, C, W.xk, kd_p, g_s);
            k::mv_f16(Ly.sc_v_dyn16, c.KV * 4, C, W.xv, vd_p, g_s);
            mv1c(Ly.wk, W.xk, k_p, k::MvConv{0, KD, Ly.h_k, Ly.sc_k, kd_p, c.KV, nullptr, W.kconv});
            mv1c(Ly.wv, W.xv, v_p, k::MvConv{0, KD, Ly.h_v, Ly.sc_v, vd_p, c.KV, nullptr, W.vconv});
        }
        // the clips, the token's K and V into the cache at its position, the attention
        const int extent = Ly.sliding ? c.extent_swa : c.extent;
        const int window = Ly.sliding ? c.window : (1 << 30);
        const int P = Ly.sliding ? P_sl : P_full;
        // the attention on the tensor cores, from an FP16 copy of q (IK2_ATTN_MMA=0: the scalar kernels - a layer
        // without a window, its keys the whole context, is bound by their arithmetic)
        const bool mma = g_attn_mma == 1 || (g_attn_mma == 2 && !Ly.sliding);
        const int ring = Ly.sliding ? Ly.kv_cap : 0;
        const int ns = Ly.sliding ? std::min(g_mma_split, std::max(1, (c.window + 63) / 64)) : g_mma_split;
        uint16_t* q16 = mma ? W.x16 : nullptr;
        if (g_kv_i8) {
            k::prep_qkv_r_i8_at(q_p, Ly.q_clip, W.kconv, (int8_t*) Ly.kc8, W.vconv, (int8_t*) Ly.vc8, r_p, Ly.r_clip, c.H,
                                c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha, c.n_floor,
                                Ly.kv_cap, q16, g_s, Ly.sink16, W.xq, C, W.sink);
            if (mma)
                k::attn_decode_mma_i8(q16, r_p, W.sink, (const int8_t*) Ly.kc8, (const int8_t*) Ly.vc8, ring, c.H, c.KV,
                                      W.pos, 0, ns, qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p,
                                      W.x16, g_s);
            else
                k::attn_decode_i8_at(q_p, r_p, W.sink, (const int8_t*) Ly.kc8, (const int8_t*) Ly.vc8,
                                     Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P, qk_scale, Ly.rel_proj, c.d_rel,
                                     extent, window, W.lg, W.o, g_s);
        } else if (g_kv_fp8) {
            k::prep_qkv_r_fp8_at(q_p, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, Ly.r_clip, c.H, c.KV, c.D, c.d_rel,
                                 c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap,
                                 k_scale, v_scale, q16, g_s, Ly.sink16, W.xq, C, W.sink);
            if (mma)
                k::attn_decode_mma_fp8(q16, r_p, W.sink, Ly.kc8, Ly.vc8, ring, c.H, c.KV, W.pos, 0, ns,
                                       qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg,
                                       W.o, gate_p, W.x16, g_s);
            else
                k::attn_decode_fp8_at(q_p, r_p, W.sink, Ly.kc8, Ly.vc8, Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P,
                                      qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg,
                                      W.o, g_s);
        } else {
            k::prep_qkv_r_f16_at(q_p, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, Ly.r_clip, c.H, c.KV, c.D, c.d_rel,
                                 c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap, q16,
                                 g_s, Ly.sink16, W.xq, C, W.sink);
            if (mma)
                k::attn_decode_mma_f16(q16, r_p, W.sink, Ly.kc, Ly.vc, ring, c.H, c.KV, W.pos, 0, ns, qk_scale,
                                       Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p, W.x16, g_s);
            else
                k::attn_decode_f16_at(q_p, r_p, W.sink, Ly.kc, Ly.vc, Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P,
                                      qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
        if (!mma) k::gate_mul_f16(W.o, gate_p, (size_t) HD, W.x16, true, g_s);   // (the tensor cores' merge has it)
        // the output projection with its short conv (and the stream added, but where a block starts: there the
        // stream joins the blocks and starts again)
        const bool first = l % c.block == 0;
        if (first) CU(cudaMemcpyAsync(blk(nb), W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
        mv1c(Ly.wo, W.o, W.tmp, k::MvConv{0, C, Ly.h_attn, Ly.sc_attn, nullptr, 0, first ? nullptr : W.h, W.r});
        if (first) ++nb;
        // the MoE: its input, the router and the picks' tables on the device, the experts
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, 1, C, 1e-6f, W.xq, g_s);
                k::loop_gate(W.xq, msg + (size_t) 3 * C, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], 1, C, c.rank,
                             1e-6f, W.lp_g, W.xq, g_s);
                k::rms(W.xq, Ly.ffn_norm, 1, C, c.eps, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        k::mv_f16(Ly.router16, c.E, C, W.xq, W.rlog, g_s);
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + c.K);
        // (NVFP4 experts go through the tensor cores, their input as two FP16 limbs the router's launch writes)
        static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 1;
        const bool exp_tc = exp_mma && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F);
        if (!g_ep_fast) {
            k::route_token(W.rlog, Ly.probs_b, c.E, c.K, c.route_scale, Ly.tab_blob, Ly.tab_sc[0], Ly.tab_sc[1],
                           Ly.tab_sc[2], rtop, rw, W.blob, W.tok, W.inv, W.sg, W.su, W.wt, g_s, W.xq, C,
                           exp_tc ? (uint16_t*) W.gu32 : nullptr);
            if (exp_tc) {
                k::experts_gate_up_swiglu_mma(W.blob, c.K, Ly.eb[0], (const uint16_t*) W.gu32, C, c.F, W.sg, W.su,
                                              c.gate_cap, c.up_cap, W.h16, g_s);
                k::experts_down_mma(W.blob, c.K, Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            } else {
                k::experts_gate_up_swiglu(W.blob, W.tok, c.K, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, c.F, W.sg, W.su,
                                          c.gate_cap, c.up_cap, W.h16, g_s);
                k::experts_down(W.blob, c.K, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            }
            k::moe_combine_conv(W.y32, W.inv, W.wt, c.K, C, Ly.h_mlp, Ly.sc_mlp, W.r, W.h, g_s);
        } else {
            const int S = c.K;
            const int F = c.F;
            k::route(W.rlog, Ly.probs_b, 1, c.E, c.K, c.route_scale, rtop, rw, g_s);

            // 1. Send xq and route_d from GPU 0 to GPU 1
            if (g_ep_stage.d_xq[0]) {
                CU(cudaMemcpyAsync(g_ep_stage.d_xq[0], dev_work(0).xq, (size_t) C * 4, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(g_ep_stage.d_route[0], dev_work(0).route_d, (size_t) S * 8, cudaMemcpyDeviceToDevice, g_s));
            } else {
                CU(cudaMemcpyPeerAsync(dev_work(1).xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                CU(cudaMemcpyPeerAsync(dev_work(1).route_d, 1, dev_work(0).route_d, 0, (size_t) S * 8, g_s));
            }
            CU(cudaEventRecord(g_ep_x, g_s));

            // Helper to run experts on a device
            auto run_dev_exp = [&](int /*d*/, Work& Wd, cudaStream_t strm) {
                if (exp_tc) {
                    k::to_f16_limbs(Wd.xq, (size_t) C, (uint16_t*) Wd.gu32, strm);
                    k::experts_gate_up_swiglu_mma(Wd.blob, S, Ly.eb[0], (const uint16_t*) Wd.gu32, C, F, Wd.sg, Wd.su,
                                                  c.gate_cap, c.up_cap, Wd.h16, strm);
                    k::experts_down_mma(Wd.blob, S, Ly.eb[0] + Ly.eb[1], Wd.h16, C, F, Wd.y32, strm);
                } else {
                    k::experts_gate_up_swiglu(Wd.blob, Wd.tok, S, Ly.ty[0], Ly.ty[1], Ly.eb[0], Wd.xq, C, F, Wd.sg, Wd.su,
                                              c.gate_cap, c.up_cap, Wd.h16, strm);
                    k::experts_down(Wd.blob, S, Ly.ty[2], Ly.eb[0] + Ly.eb[1], Wd.h16, C, F, Wd.y32, strm);
                }
            };

            // 2. GPU 1: wait for xq & route_d, select resident experts, compute experts, copy result to GPU 0
            use_dev(1);
            CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
            if (g_ep_stage.d_xq[1]) {
                CU(cudaMemcpyAsync(dev_work(1).xq, g_ep_stage.d_xq[1], (size_t) C * 4, cudaMemcpyDeviceToDevice, g_s));
                k::ep_select_token(g_ep_stage.d_route[1], (const float*)(g_ep_stage.d_route[1] + S), Ly.ep_tab_blob[1],
                                   Ly.ep_tab_sc[1][0], Ly.ep_tab_sc[1][1], Ly.ep_tab_sc[1][2],
                                   S, dev_work(1).blob, dev_work(1).tok, dev_work(1).inv, dev_work(1).sg, dev_work(1).su, dev_work(1).wt, g_s);
            } else {
                k::ep_select_token(dev_work(1).route_d, (const float*)(dev_work(1).route_d + S), Ly.ep_tab_blob[1],
                                   Ly.ep_tab_sc[1][0], Ly.ep_tab_sc[1][1], Ly.ep_tab_sc[1][2],
                                   S, dev_work(1).blob, dev_work(1).tok, dev_work(1).inv, dev_work(1).sg, dev_work(1).su, dev_work(1).wt, g_s);
            }
            run_dev_exp(1, dev_work(1), g_s);
            if (g_ep_stage.d_y[1]) {
                CU(cudaMemcpyAsync(g_ep_stage.d_y[1], dev_work(1).y32, (size_t) S * C * 4, cudaMemcpyDeviceToDevice, g_s));
            } else {
                CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) S * C, 0, dev_work(1).y32, 1, (size_t) S * C * 4, g_s));
            }
            CU(cudaEventRecord(g_ep_done, g_s));

            // 3. GPU 0: select resident experts, compute experts, wait for GPU 1, combine & conv
            use_dev(0);
            k::ep_select_token(dev_work(0).route_d, (const float*)(dev_work(0).route_d + S), Ly.ep_tab_blob[0],
                               Ly.ep_tab_sc[0][0], Ly.ep_tab_sc[0][1], Ly.ep_tab_sc[0][2],
                               S, dev_work(0).blob, dev_work(0).tok, dev_work(0).inv, dev_work(0).sg, dev_work(0).su, dev_work(0).wt, g_s);
            run_dev_exp(0, dev_work(0), g_s);
            CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            const float* y_peer = g_ep_stage.d_y[0] ? g_ep_stage.d_y[0] : (dev_work(0).y32 + (size_t) S * C);
            k::ep_combine_conv(dev_work(0).y32, y_peer, dev_work(0).blob, dev_work(0).wt,
                               S, C, Ly.h_mlp, Ly.sc_mlp, dev_work(0).r, dev_work(0).h, g_s);
        }
    }
    if (out_mode == OutMode::kNone) return;
    {   // the head
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = W.h;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
        mv1(M.out_w, W.xq, W.logits);
        if (out_mode == OutMode::kArgmax) {
            k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d, g_s);
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, 4, cudaMemcpyDeviceToHost, g_s));
        } else if (out_mode == OutMode::kCand) {   // (the softcap keeps the logits within +- logit_cap)
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            k::cand_topk(W.logits, c.n_vocab, -c.logit_cap, c.logit_cap, kCandK, kCandCap, W.cand_part, W.cand_d, g_s);
            CU(cudaMemcpyAsync(W.cand_h, W.cand_d, (size_t) (2 + 2 * kCandCap) * 4, cudaMemcpyDeviceToHost, g_s));
        } else {
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            CU(cudaMemcpyAsync(W.logits_h, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
        }
    }
}

// one token through the resident path: its logits appended to logits_out (kLogits) or its argmax (kArgmax)
void fast_token(Model& M, int tok, int pos, OutMode out_mode, std::vector<float>& logits_out,
                std::vector<int>* argmax_out) {
    const Config& c = M.c;
    use_dev(0);
    Work& W = dev_work(0);
    const size_t rb = (size_t) c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    k::decode_host(M.embd_type, M.embd + (size_t) tok * rb, c.C, W.emb_h);
    W.pos_h[0] = pos;
    if (!g_graph) {
        fast_token_launches(M, W, out_mode);
    } else {
        FastGraph& G = g_fgraph[(int) out_mode];
        if (!G.exec && !G.disabled) {
            cudaError_t ce = cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal);
            if (ce == cudaSuccess) {
                fast_token_launches(M, W, out_mode);
                ce = cudaStreamEndCapture(g_s, &G.graph);
                if (ce == cudaSuccess) {
                    ce = cudaGraphInstantiate(&G.exec, G.graph, 0);
                    if (ce == cudaSuccess) {
                        std::printf("resident%s: CUDA graph captured and instantiated!\n",
                                    g_ep_fast ? " EP" : "");
                        std::fflush(stdout);
                    } else {
                        std::printf("resident: graph instantiate failed (%s), fallback to direct launches\n",
                                    cudaGetErrorString(ce));
                        std::fflush(stdout);
                        cudaGraphDestroy(G.graph);
                        G.graph = nullptr;
                        G.disabled = true;
                        (void) cudaGetLastError();
                    }
                } else {
                    std::printf("resident: graph end capture failed (%s), fallback to direct launches\n",
                                cudaGetErrorString(ce));
                    std::fflush(stdout);
                    G.disabled = true;
                    (void) cudaGetLastError();
                }
            } else {
                G.disabled = true;
                (void) cudaGetLastError();
            }
        }
        if (G.exec) {
            CU(cudaGraphLaunch(G.exec, g_s));
        } else {
            fast_token_launches(M, W, out_mode);
        }
    }
    CU(cudaStreamSynchronize(g_s));
    if (out_mode == OutMode::kArgmax) {
        argmax_out->assign(1, W.argmax_h[0]);
    } else if (out_mode == OutMode::kLogits) {
        logits_out.insert(logits_out.end(), W.logits_h, W.logits_h + c.n_vocab);
    } else if (out_mode == OutMode::kCand) {
        if (W.cand_h[0] <= kCandCap) {
            g_cand.n = W.cand_h[0], g_cand.idx = W.cand_h + 2, g_cand.val = (const float*) (W.cand_h + 2 + kCandCap);
        } else {   // (more than the buffer holds - logits flat past belief: the logits themselves, this once)
            CU(cudaMemcpy(W.logits_h, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost));
            logits_out.insert(logits_out.end(), W.logits_h, W.logits_h + c.n_vocab);
            g_cand.n = -1;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// IK2_DEVPLAN (default on with the CPU lane on one GPU, the experts streamed): a decode token's layers are enqueued
// without the host waiting for each one's routing.  After the router, k_dp_plan decides on the GPU which picks are in
// the cache (g_dp.tab: an expert's slot or null, from the cache's state), fills the expert kernels' tables (a miss: a
// null blob, skipped) and writes the picks and the MoE input to a mapped host record; a host thread (dp_service)
// computes the misses on the CPU lane into mapped rows and raises a counter; k_dp_wait waits for it only when the CPU
// owes rows, k_dp_merge puts them in, the combine follows.  The cache does not change during such tokens.
struct DevPlan {
    bool ok = false, dirty = true;
    // the plans' 32-bit counters set back to 0 (a quiesce with reset asked) before they could wrap: planned counts the
    // visits enqueued since
    std::atomic<bool> reset{false};
    long long planned = 0;
    std::thread th;
    std::atomic<bool> run{false};
    int R = 64, K = 0, C = 0, F = 0;
    size_t rec_bytes = 0;
    uint8_t *rec_h = nullptr, *rec_d = nullptr;   // mapped [R] records (see k_dp_plan)
    float *ey_h = nullptr, *ey_d = nullptr;       // mapped [R][kDpS][C]: the CPU's rows
    volatile int* ctl = nullptr;                  // mapped: [0] planned (GPU), [1] done (GPU), [2] served (host)
    int* ctl_d = nullptr;
    int* seqd = nullptr;                          // device: [0] the counter, [1] the current
    const uint8_t** tab = nullptr;                // device [n_layer * E]
    const uint8_t** blob = nullptr;
    int *tok = nullptr, *inv = nullptr, *mflag = nullptr;
    float *sg = nullptr, *su = nullptr, *wt = nullptr;
    const uint8_t** gblob = nullptr;   // (a verify window's hit groups: see k_dp_plan)
    int *gn = nullptr, *gtok = nullptr, *grow = nullptr;
    float gcap = 0, ucap = 0;
    uint64_t cpu_experts = 0, copies = 0, skipped = 0;
    const Model* M = nullptr;
    // the cache during such tokens: a mirror of its residency (slot -> key, -2 while a copy is on its way; key -> slot),
    // recency, and the copies of misses (staged: see dp_advance); written back into the ExpertCache before another
    // forward (quiesce: 1 asked by the host, 2 done by the service thread)
    int pool = -1, n_active = 0, E = 0;
    std::vector<int> owner, loc;
    std::vector<uint64_t> last;
    std::vector<char> busy;
    // (IK2_DP_LFU, default on: a key's picks counted with a decay of 0.95 a token - lscore its log, shifted by its tick
    //  so that keys compare without the decay applied; the copy of a miss takes the slot of the least counted key, and
    //  only when the miss counts more.  A trace of 255 decode tokens: misses 21.9 -> 19.0 a token against LRU)
    std::vector<double> lscore;
    bool lfu = !getenv("IK2_DP_LFU") || std::atoi(getenv("IK2_DP_LFU"));
    double lnd = 0.0;   // ln of the decay a visit
    std::vector<int> changed;
    uint64_t tick = 0;
    struct Copy { bool on = false; int s = -1, key = -1, l = 0, e = 0, stage = 0, need = 0; cudaEvent_t ev = nullptr; };
    static constexpr int kQ = 48;
    Copy q[kQ];
    const uint8_t** tsrc = nullptr;   // pinned [kQ][2]: a table entry to copy (the slot, null)
    std::atomic<int> quiesce{0};
    // IK2_TOKGRAPH (default on with the plan): a decode token as one CUDA graph, captured while the token before runs
    // (its inputs: the embedding row from emb, the position set by a kernel; the argmax read after it)
    bool graph = true;
    float* emb = nullptr;
    cudaGraphExec_t exec[2] = {nullptr, nullptr};
    int exec_pos[2] = {-1, -1};
    int exec_ep = 0;   // (k::scratch_epoch() the graphs were captured at: a growth since frees what they hold)
} g_dp;
__global__ void k_set_int(int* p, int v) { *p = v; }
__device__ __forceinline__ int dp_ld_sys(const volatile int* p) {
    int v;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
// (the routing too, as k::route's warp kernel picks: an expert's rank the count of the larger sigmoid + bias, ties to the
// lower expert; the picks ascending, their weights the sigmoids over their sum - the same operations in the same order,
// the same bits - a launch less a visit)
// T tokens (1, or a verify window's <= kDpT): slot s = t K + k; record header ints [0] seq, [1] layer, [2] T, [8 + s]
// the slot's expert, [72 + s] its miss flag, then x [T][C] at kDpHdr bytes.  T > 1: the hits grouped by expert on the
// device (gblob / gn / gtok / grow, kGroupMax tokens a group, gn 0 past the last) for the grouped kernels
constexpr int kDpT = 8, kDpS = 64, kDpHdr = 1024;
__global__ void k_dp_plan(const float* __restrict__ logits, const float* __restrict__ bias, int E, float scale,
                          const uint8_t* const* __restrict__ tab_l, int K, int C, const float* __restrict__ x, int T,
                          uint8_t* ring, size_t rec_bytes, int R, int l, int* seqd, volatile int* ctl,
                          const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt, int* mflag,
                          const uint8_t** gblob, int* gn, int* gtok, int* grow) {
    __shared__ int s_seq;
    __shared__ float s_sig[256], s_val[256];
    __shared__ int s_flag[256], top[kDpS];
    __shared__ float w[kDpS];
    for (int t = 0; t < T; ++t) {   // (each token's routing as k::route's warp kernel picks)
        const int e = threadIdx.x;
        const float* lg = logits + (size_t) t * E;
        if (e < E) {
            const float sgm = 1.0f / (1.0f + expf(-__ldg(lg + e)));
            s_sig[e] = sgm, s_val[e] = sgm + __ldg(bias + e);
        }
        __syncthreads();
        if (e < E) {
            const float v = s_val[e];
            int rank = 0;
            for (int j = 0; j < E; ++j) {
                const float vj = s_val[j];
                rank += vj > v || (vj == v && j < e);
            }
            s_flag[e] = rank < K;
        }
        __syncthreads();
        if (e < E && s_flag[e]) {
            int pos = 0;
            for (int j = 0; j < e; ++j) pos += s_flag[j];
            top[t * K + pos] = e;
        }
        __syncthreads();
        if (e == 0) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += s_sig[top[t * K + k]];
            for (int k = 0; k < K; ++k) w[t * K + k] = s_sig[top[t * K + k]] / (sum + 1e-20f) * scale;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int q = seqd[0] + 1;
        seqd[0] = q, seqd[1] = q, s_seq = q;
        // (the ring's record q % R free: the host done with plan q - R - a plan without misses does not wait for the
        // host, so a run of them could otherwise lap a record still being read)
        while (dp_ld_sys(ctl + 2) < q - R) strata_spin_pause();
    }
    __syncthreads();
    const int seq = s_seq, S = T * K;
    uint8_t* rec = ring + (size_t) (seq % R) * rec_bytes;
    int* hd = (int*) rec;
    float* xr = (float*) (rec + kDpHdr);
    for (int i = threadIdx.x; i < T * C; i += blockDim.x) xr[i] = x[i];
    if (threadIdx.x < S) {
        const int s = threadIdx.x, e = top[s];
        const uint8_t* p = tab_l[e];
        blob[s] = p, tok[s] = s / K, inv[s] = s, sg[s] = 1.0f, su[s] = 1.0f, wt[s] = w[s];
        mflag[s] = p == nullptr;
        hd[8 + s] = e, hd[72 + s] = p == nullptr;
    }
    if (T > 1) {   // the hits' groups: a group an expert (its first slot leads), its slots in their order
        __shared__ int s_gi[kDpS];
        const int sl = threadIdx.x;
        if (sl < kDpS) gn[sl] = 0;
        int lead = -1, pos = 0;
        if (sl < S && tab_l[top[sl]]) {
            const int e = top[sl];
            lead = sl;
            for (int j = 0; j < sl; ++j)
                if (top[j] == e) { if (lead == sl) lead = j; ++pos; }
            if (lead == sl) {   // (its index: the leaders before it)
                int g = 0;
                for (int j = 0; j < sl; ++j) {
                    bool lj = tab_l[top[j]] != nullptr;
                    for (int i = 0; lj && i < j; ++i) lj = top[i] != top[j];
                    g += lj;
                }
                s_gi[sl] = g;
                gblob[g] = tab_l[e];
            }
        }
        __syncthreads();
        if (lead >= 0 && pos < k::kGroupMax) {
            const int gi = s_gi[lead];
            gtok[gi * k::kGroupMax + pos] = sl / K, grow[gi * k::kGroupMax + pos] = sl;
            atomicAdd(&gn[gi], 1);
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        hd[1] = l, hd[2] = T;
        __threadfence_system();
        *(volatile int*) &hd[0] = seq;
        __threadfence_system();
        ctl[0] = seq;
    }
}
__global__ void k_dp_wait(const int* __restrict__ mflag, int K, const int* __restrict__ seqd, volatile int* ctl) {
    if (threadIdx.x) return;
    const int seq = seqd[1];
    int any = 0;
    for (int k = 0; k < K; ++k) any |= mflag[k];
    if (any && dp_ld_sys(ctl + 2) < seq) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        while (dp_ld_sys(ctl + 2) < seq) strata_spin_pause();
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        ctl[3] = (int) min(2147483647ull, (unsigned long long) ctl[3] + (t1 - t0) / 1000);   // the GPU's wait, us (saturating)
        ctl[4] = (int) min(2147483647ull, (unsigned long long) ctl[4] + 1);
    }
    __threadfence_system();
    ctl[1] = seq;
}
__global__ void k_dp_merge(float* __restrict__ y, const float* ey, const int* __restrict__ mflag,
                           const int* __restrict__ seqd, int C, int R) {
    const int k = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!mflag[k] || i >= C) return;
    y[(size_t) k * C + i] = __ldcv(ey + ((size_t) (seqd[1] % R) * kDpS + k) * C + i);
}
// IK2_DP_MULTI=1: a verify window (2 .. kDpT tokens) through the plan too - no host sync a visit; measured no faster
// than the host-planned path on 8 GB (the window's CPU misses bound it) and drafting slower beside its threads
bool g_dp_multi = getenv("IK2_DP_MULTI") && std::atoi(getenv("IK2_DP_MULTI"));
bool g_dp_multi_types = false;   // (every layer's experts batch_multi_ok: set at load)
bool g_dp_time = getenv("IK2_DP_TIME") && std::atoi(getenv("IK2_DP_TIME"));
double g_dp_t0 = 0, g_dp_t_enq = 0, g_dp_t_tok = 0;
long g_dp_n = 0;
bool dp_ready(cudaEvent_t e) {
    const cudaError_t r = cudaEventQuery(e);
    if (r == cudaSuccess) return true;
    if (r != cudaErrorNotReady) CU(r);
    return false;
}
// a copy's stages: 1 the old expert's table entry nulled (on the copy stream), 2 waiting for the GPU to pass every plan
// that may have read it (its done counter at the planned one + 1, or the GPU idle), 3 the blocks and the new entry
// queued, done once landed
void dp_advance(bool idle) {
    if (g_dp.pool < 0) return;   // (the service runs before the first dp_sync_tab names the pool)
    const Model& M = *g_dp.M;
    ExpertCache& ECr = dev_cache(0);
    auto& P = ECr.pools[g_dp.pool];
    for (int qi = 0; qi < DevPlan::kQ; ++qi) {
        auto& q = g_dp.q[qi];
        if (!q.on) continue;
        if (q.stage == 1 && dp_ready(q.ev)) { q.need = g_dp.ctl[0] + 1; q.stage = 2; }
        if (q.stage == 2 && (idle || g_dp.ctl[1] >= q.need)) {
            uint8_t* dst = P.data + (size_t) q.s * P.slot_bytes;
            ECr.copy_in(M.L[q.l], q.e, dst, g_cs);
            g_dp.tsrc[2 * qi] = dst;
            CU(cudaMemcpyAsync(g_dp.tab + q.key, g_dp.tsrc + 2 * qi, sizeof(void*), cudaMemcpyHostToDevice, g_cs));
            CU(cudaEventRecord(q.ev, g_cs));
            q.stage = 3;
        } else if (q.stage == 3 && dp_ready(q.ev)) {
            g_dp.owner[q.s] = q.key;
            g_dp.loc[q.key] = q.s;
            g_dp.last[q.s] = g_dp.tick;
            g_dp.busy[q.key] = 0;
            g_dp.changed.push_back(q.s);
            q.on = false;
            ++g_dp.copies;
        }
    }
}
// a pick of key at the current tick counted (see DevPlan::lscore)
void dp_count(int key) {
    if (!g_dp.lfu) return;
    double& L = g_dp.lscore[key];
    const double tl = (double) g_dp.tick * g_dp.lnd;
    const double eff = std::isfinite(L) ? std::exp(L + tl) : 0.0;
    L = std::log(eff + 1.0) - tl;
}
void dp_request(int l, int e) {   // a miss: its copy into the least recently used (least counted: lfu) slot not in use
    const int key = l * g_dp.E + e;
    if (g_dp.loc[key] >= 0 || g_dp.busy[key]) return;
    int qi = 0;
    while (qi < DevPlan::kQ && g_dp.q[qi].on) ++qi;
    if (qi == DevPlan::kQ) { ++g_dp.skipped; return; }
    int v = -1;
    for (int s2 = 0; s2 < g_dp.n_active; ++s2) {
        if (g_dp.owner[s2] == -2 || g_dp.last[s2] == g_dp.tick) continue;
        if (g_dp.owner[s2] < 0) { v = s2; break; }
        if (v < 0) { v = s2; continue; }
        if (g_dp.lfu ? g_dp.lscore[g_dp.owner[s2]] < g_dp.lscore[g_dp.owner[v]] : g_dp.last[s2] < g_dp.last[v]) v = s2;
    }
    if (v < 0) return;
    if (g_dp.lfu && g_dp.owner[v] >= 0 && !(g_dp.lscore[key] > g_dp.lscore[g_dp.owner[v]])) return;   // (not worth it)
    auto& q = g_dp.q[qi];
    const cudaEvent_t ev = q.ev;
    q = DevPlan::Copy{true, v, key, l, e, 1, 0, ev};
    const int old = g_dp.owner[v];
    g_dp.owner[v] = -2;
    g_dp.busy[key] = 1;
    if (old >= 0) {
        g_dp.loc[old] = -1;
        g_dp.tsrc[2 * qi + 1] = nullptr;
        CU(cudaMemcpyAsync(g_dp.tab + old, g_dp.tsrc + 2 * qi + 1, sizeof(void*), cudaMemcpyHostToDevice, g_cs));
    }
    CU(cudaEventRecord(q.ev, g_cs));
}
// (the GPU idle) every copy landed, the residency written back into the ExpertCache
void dp_drain() {
    for (;;) {
        bool any = false;
        for (auto& q : g_dp.q) any |= q.on;
        if (!any) break;
        dp_advance(true);
        CU(cudaStreamSynchronize(g_cs));
    }
    ExpertCache& ECr = dev_cache(0);
    std::sort(g_dp.changed.begin(), g_dp.changed.end());
    g_dp.changed.erase(std::unique(g_dp.changed.begin(), g_dp.changed.end()), g_dp.changed.end());
    for (int s2 : g_dp.changed)
        if (g_dp.owner[s2] >= 0) ECr.assign(g_dp.pool, s2, g_dp.owner[s2], ++ECr.tick);
    ECr.refree(g_dp.pool);
    g_dp.changed.clear();
}
// the device plan's copies drained and its mirror written back into the cache, before the host changes the cache
// (a forward of several tokens, a growth of the K / V, slots lent); then the plan is rebuilt at its next token
void dp_quiesce() {
    if (g_dp.M && !g_dp.dirty) {
        CU(cudaDeviceSynchronize());
        g_dp.quiesce.store(1, std::memory_order_release);
        while (g_dp.quiesce.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        g_dp.quiesce.store(0, std::memory_order_release);
    }
    g_dp.dirty = true;
}
// the host's side: each record in order, its misses on the CPU lane, then the counter
void dp_service() {
    const Model& M = *g_dp.M;
    static ik2::cpu::BatchJob jobs[kDpS];
    int next = 1;
    while (g_dp.run.load(std::memory_order_relaxed)) {
        unsigned spin = 0;
        while (g_dp.ctl[0] < next) {
            if (!g_dp.run.load(std::memory_order_relaxed)) return;
            if (g_dp.quiesce.load(std::memory_order_acquire) == 1) {   // (the GPU idle, every record served)
                dp_drain();
                const bool rs = g_dp.reset.load(std::memory_order_acquire);
                g_dp.quiesce.store(2, std::memory_order_release);
                if (rs) {   // (the counters set back by the host: its quiesce 0 says they are)
                    while (g_dp.quiesce.load(std::memory_order_acquire) != 0) std::this_thread::yield();
                    next = 1;
                }
            }
            if ((spin & 63) == 0) dp_advance(false);   // (the copies' stages: driver calls, not on every spin)
            if (++spin < 200000) __builtin_ia32_pause();
            else std::this_thread::yield();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const uint8_t* rec = g_dp.rec_h + (size_t) (next % g_dp.R) * g_dp.rec_bytes;
        const int* hd = (const int*) rec;
        const float* xr = (const float*) (rec + kDpHdr);
        const int lay = hd[1];   // (read before the record is released: ctl[2] lets the GPU reuse it)
        const Layer& Ly = M.L[lay];
        const int T = std::max(1, std::min(kDpT, hd[2])), S = T * g_dp.K;
        float* ey = g_dp.ey_h + (size_t) (next % g_dp.R) * kDpS * g_dp.C;
        const uint64_t t = ++g_dp.tick;
        for (int k = 0; k < S; ++k) {   // the hits' recency, every pick's count
            dp_count(lay * g_dp.E + hd[8 + k]);
            if (!hd[72 + k]) {
                const int s2 = g_dp.loc[lay * g_dp.E + hd[8 + k]];
                if (s2 >= 0) g_dp.last[s2] = t;
            }
        }
        int n = 0;   // the misses, an expert's tokens a job (its weights read once for them)
        int ej[kDpS];
        for (int k = 0; k < S; ++k) {
            if (!hd[72 + k]) continue;
            const int e = hd[8 + k];
            const float* x = xr + (size_t) (k / g_dp.K) * g_dp.C;
            float* y = ey + (size_t) k * g_dp.C;
            int j = 0;
            while (j < n && !(ej[j] == e && jobs[j].nt < 8)) ++j;
            if (j < n) {
                auto& J = jobs[j];
                J.xt[J.nt] = x, J.yt[J.nt] = y;
                ++J.nt;
                continue;
            }
            ej[n] = e;
            auto& J = jobs[n++];
            J = ik2::cpu::BatchJob{};
            J.gate = Ly.host[0] + (size_t) e * Ly.stride[0];
            J.up = Ly.host[1] + (size_t) e * Ly.stride[1];
            J.down = Ly.host[2] + (size_t) e * Ly.stride[2];
            J.tg = Ly.ty[0], J.tu = Ly.ty[1], J.td = Ly.ty[2];
            J.x = x, J.sg = 1.0f, J.su = 1.0f;
            J.y = y;
        }
        // the misses on the CPU and published first (the GPU waits for them), then their copies for the next tokens
        ik2::cpu::experts_batch(jobs, n, g_dp.C, g_dp.F, g_dp.gcap, g_dp.ucap);
        g_dp.cpu_experts += n;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        g_dp.ctl[2] = next;
        ++next;
        for (int j = 0; j < n; ++j) dp_request(lay, ej[j]);
        if (n) dp_advance(false);
    }
}
void dp_init(const Model& M) {
    const Config& c = M.c;
    g_dp.M = &M, g_dp.K = c.K, g_dp.C = c.C, g_dp.F = c.F, g_dp.gcap = c.gate_cap, g_dp.ucap = c.up_cap;
    g_dp.lscore.assign((size_t) c.n_layer * c.E, -INFINITY);   // (kept across the plan's rebuilds)
    g_dp.lnd = std::log(0.95) / std::max(1, c.n_visit);
    g_dp.rec_bytes = kDpHdr + (size_t) kDpT * c.C * 4;
    CU(cudaHostAlloc((void**) &g_dp.rec_h, g_dp.rec_bytes * g_dp.R, cudaHostAllocMapped));
    std::memset(g_dp.rec_h, 0, g_dp.rec_bytes * g_dp.R);
    CU(cudaHostGetDevicePointer((void**) &g_dp.rec_d, g_dp.rec_h, 0));
    CU(cudaHostAlloc((void**) &g_dp.ey_h, (size_t) g_dp.R * kDpS * c.C * 4, cudaHostAllocMapped));
    CU(cudaHostGetDevicePointer((void**) &g_dp.ey_d, g_dp.ey_h, 0));
    int* ct = nullptr;
    CU(cudaHostAlloc((void**) &ct, 64, cudaHostAllocMapped));
    std::memset(ct, 0, 64);
    g_dp.ctl = ct;
    CU(cudaHostGetDevicePointer((void**) &g_dp.ctl_d, ct, 0));
    CU(cudaHostAlloc((void**) &g_dp.tsrc, DevPlan::kQ * 2 * sizeof(void*), cudaHostAllocDefault));
    for (auto& q : g_dp.q) CU(cudaEventCreateWithFlags(&q.ev, cudaEventDisableTiming));
    ik2::cpu::spin_start(g_pool->size());   // (the service thread is the batches' caller)
    g_dp.run = true;
    g_dp.th = std::thread(dp_service);
    g_dp.th.detach();
}
// the device side's buffers, at load (before the expert cache takes what is left)
void dp_alloc(const Config& c) {
    g_dp.seqd = dalloc<int>(2);
    CU(cudaMemset(g_dp.seqd, 0, 8));
    g_dp.tab = dalloc<const uint8_t*>((size_t) c.n_layer * c.E);
    g_dp.blob = dalloc<const uint8_t*>(kDpS);
    g_dp.tok = dalloc<int>(kDpS), g_dp.inv = dalloc<int>(kDpS), g_dp.mflag = dalloc<int>(kDpS);
    g_dp.sg = dalloc<float>(kDpS), g_dp.su = dalloc<float>(kDpS), g_dp.wt = dalloc<float>(kDpS);
    g_dp.gblob = dalloc<const uint8_t*>(kDpS), g_dp.gn = dalloc<int>(kDpS);
    g_dp.gtok = dalloc<int>((size_t) kDpS * k::kGroupMax), g_dp.grow = dalloc<int>((size_t) kDpS * k::kGroupMax);
    g_dp.emb = dalloc<float>(c.C);
}
// the cache's residency into g_dp.tab (the GPU idle: the copies landed)
void dp_sync_tab(const Model& M, ExpertCache& ECr) {
    const Config& c = M.c;
    std::vector<const uint8_t*> t((size_t) c.n_layer * c.E, nullptr);
    for (int l = 0; l < c.n_layer; ++l) {
        const int p = ECr.layer_pool[l];
        if (p < 0) continue;
        const auto& P = ECr.pools[p];
        for (int e = 0; e < c.E; ++e) {
            const int s2 = ECr.loc[(size_t) l * c.E + e];
            if (s2 >= 0 && s2 < P.n_active) t[(size_t) l * c.E + e] = P.data + (size_t) s2 * P.slot_bytes;
        }
    }
    CU(cudaMemcpy(g_dp.tab, t.data(), t.size() * sizeof(void*), cudaMemcpyHostToDevice));
    // the mirror (one pool: the decode's layers)
    g_dp.pool = ECr.layer_pool[0];
    const auto& P = ECr.pools[g_dp.pool];
    g_dp.E = c.E, g_dp.n_active = P.n_active;
    g_dp.owner.assign(P.n_slots, -1), g_dp.loc.assign((size_t) c.n_layer * c.E, -1), g_dp.last.assign(P.n_slots, 0);
    g_dp.busy.assign((size_t) c.n_layer * c.E, 0);
    for (int s2 = 0; s2 < P.n_active; ++s2)
        if (P.owner[s2] >= 0) g_dp.owner[s2] = P.owner[s2], g_dp.loc[P.owner[s2]] = s2, g_dp.last[s2] = P.used[s2] ? 1 : 0;
}
// --kv-grow (default on a card under 16 GB): the full-attention layers' K / V are not allocated for the whole --ctx at
// load but kept in the top slots of the expert cache's pool, as many as the positions so far need, doubling; a growth
// drops the experts in the slots it takes and moves the K / V there (through host memory: the regions overlap)
bool g_kv_grow = false;
int g_kv_cap_full = 0;
int g_kv_ctx = 0;        // --ctx   // the full layers' K / V capacity now (tokens)
void kv_ensure(Model& M, int need) {
    if (!g_kv_grow || need <= g_kv_cap_full) return;
    const Config& c = M.c;
    const size_t KD = (size_t) c.KV * c.D, tb = kv_tok_b(KD);   // (bytes a token's K, or V, row)
    int ncap = std::max(need, std::max(2 * g_kv_cap_full, kv_b8() ? 4096 : 2048));
    ncap = std::min((ncap + 255) / 256 * 256, M.V.empty() ? ncap : g_kv_ctx);
    if (ncap < need) die("--kv-grow: past --ctx");
    std::vector<int> full;
    for (int v = 0; v < (int) M.V.size(); ++v) if (!M.L[M.V[v].l].sliding) full.push_back(v);
    if (full.empty()) return;
    ExpertCache& EC = dev_cache(0);
    const int p = EC.layer_pool[M.V[full[0]].l];
    if (p < 0) die("--kv-grow: no pool");
    auto& P = EC.pools[p];
    auto slots_for = [&](int cap) { return (int) ((2 * full.size() * (size_t) cap * tb + P.slot_bytes - 1) / P.slot_bytes); };
    if (P.n_slots - slots_for(ncap) < 2 * c.E + 80)   // (the doubling does not fit: as few positions as asked for)
        ncap = std::max(need, std::min((need + 255) / 256 * 256, g_kv_ctx));
    const size_t ob = (size_t) g_kv_cap_full * tb, nb = (size_t) ncap * tb;
    const size_t total = 2 * full.size() * nb;
    const int k_new = slots_for(ncap);
    if (P.n_slots - k_new < 2 * c.E + 80)
        die("--kv-grow: the K / V for " + std::to_string(ncap) + " tokens leaves too few expert slots (this card holds the "
            "full layers' K / V for about " + std::to_string((int) ((size_t) std::max(0, P.n_slots - 2 * c.E - 80) * P.slot_bytes /
                                                                   (2 * full.size() * tb))) + " tokens; a smaller --ctx)");
    if (P.n_active < P.n_slots - P.kv_res) die("--kv-grow: a growth while slots are lent (internal)");
    CU(cudaDeviceSynchronize());
    if (g_dp.M && !g_dp.dirty) {   // (the device plan's table and mirror name slots this growth takes: drained, rebuilt)
        g_dp.quiesce.store(1, std::memory_order_release);
        while (g_dp.quiesce.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        g_dp.quiesce.store(0, std::memory_order_release);
    }
    g_dp.dirty = true;
    g_seq_done = g_seq_next - 1;
    std::vector<uint8_t> old;   // the K / V so far, through the host (the old and the new regions overlap)
    if (ob) {
        old.resize(2 * full.size() * ob);
        for (size_t i = 0; i < full.size(); ++i) {
            const Visit& Vi = M.V[full[i]];
            const void* kp = kv_b8() ? (const void*) Vi.kc8 : (const void*) Vi.kc;
            const void* vp = kv_b8() ? (const void*) Vi.vc8 : (const void*) Vi.vc;
            CU(cudaMemcpy(old.data() + (2 * i) * ob, kp, ob, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(old.data() + (2 * i + 1) * ob, vp, ob, cudaMemcpyDeviceToHost));
        }
    }
    if (P.n_active > P.n_slots - k_new) EC.lend(p, P.n_active - (P.n_slots - k_new));   // the experts in them dropped
    P.kv_res = k_new;
    uint8_t* base = P.data + (size_t) (P.n_slots - k_new) * P.slot_bytes;
    for (size_t i = 0; i < full.size(); ++i) {
        Visit& Vi = M.V[full[i]];
        uint8_t* kp = base + (2 * i) * nb;
        uint8_t* vp = base + (2 * i + 1) * nb;
        if (ob) {
            CU(cudaMemcpy(kp, old.data() + (2 * i) * ob, ob, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(vp, old.data() + (2 * i + 1) * ob, ob, cudaMemcpyHostToDevice));
        }
        if (kv_b8()) Vi.kc8 = kp, Vi.vc8 = vp;
        else Vi.kc = (uint16_t*) kp, Vi.vc = (uint16_t*) vp;
        Vi.kv_cap = ncap;
    }
    std::printf("kv-grow: the full layers' K / V for %d tokens (%.2f GiB, %d expert slots; %d left)\n", ncap,
                total / 1073741824.0, k_new, P.n_slots - k_new);
    g_kv_cap_full = ncap;
}
void forward(Model& M, const int* tok, int T, int pos0, bool all_logits, OutMode out_mode, const std::string& dump,
             std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp) {
    kv_ensure(M, pos0 + T);
    const Config& c = M.c;
    // (a decode token; or a verify window of <= kDpT tokens - its taps and its convs' records as forward() keeps them)
    const bool dp_tok = g_dp.ok && dump.empty() && !g_prof && !g_trace && g_pop.empty() &&
                        ((T == 1 && !g_rec && !g_taps) ||
                         (T > 1 && T <= kDpT && T * c.K <= kDpS && g_dp_multi && g_dp_multi_types));
    if (dp_tok && g_dp_time && !g_tg_capturing) g_dp_t0 = now();
    if (dp_tok && g_dp.planned > g_dp_reset_at && !g_tg_capturing) {   // (the plans' counters back to 0, every record served, before a wrap)
        CU(cudaDeviceSynchronize());
        g_dp.reset.store(true, std::memory_order_release);
        g_dp.quiesce.store(1, std::memory_order_release);
        while (g_dp.quiesce.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        for (int i = 0; i < 3; ++i) g_dp.ctl[i] = 0;
        CU(cudaMemset(g_dp.seqd, 0, 2 * sizeof(int)));
        CU(cudaDeviceSynchronize());
        g_dp.reset.store(false, std::memory_order_release);
        g_dp.quiesce.store(0, std::memory_order_release);
        g_dp.planned = 0;
        g_dp.dirty = true;
    }
    if (dp_tok) g_dp.planned += c.n_visit;
    if (dp_tok && g_dp.dirty) {
        CU(cudaDeviceSynchronize());
        g_seq_done = g_seq_next - 1;
        if (!g_dp.M) dp_init(M);
        dp_sync_tab(M, dev_cache(0));
        g_dp.dirty = false;
    }
    if (!dp_tok && g_dp.M && !g_dp.dirty) {   // (another forward may change the cache: the copies drained, written back)
        CU(cudaDeviceSynchronize());
        g_dp.quiesce.store(1, std::memory_order_release);
        while (g_dp.quiesce.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        g_dp.quiesce.store(0, std::memory_order_release);
    }
    if (!dp_tok) g_dp.dirty = true;
    const int C = c.C;
    if ((g_fast || g_ep_fast) && T == 1 && dump.empty() && !g_prof && !g_rec && !g_taps && !g_trace && g_pop.empty() &&
        g_img.at(pos0) < 0) {
        fast_token(M, tok[0], pos0, out_mode, logits_out, argmax_out);
        return;
    }
    if (out_mode == OutMode::kCand) out_mode = OutMode::kLogits, g_cand.n = -1;   // (this path: the logits)
    use_dev(M.L[0].dev);
    Work* Wp = &dev_work(g_cur);
    ExpertCache* ECp = &dev_cache(g_cur);
#define W (*Wp)
#define EC (*ECp)
    // the embedding rows, normed (pinned host staging, no stream sync needed)
    if (g_tg_capturing) {   // (a token graph: the row from g_dp.emb, the position a constant of the graph)
        CU(cudaMemcpyAsync(W.tmp, g_dp.emb, (size_t) C * 4, cudaMemcpyDeviceToDevice, g_s));
        k::rms(W.tmp, M.embd_norm, T, C, c.eps, W.h, g_s);
        k_set_int<<<1, 1, 0, g_s>>>(W.pos, pos0);
    } else {
        {
            const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
            for (int t = 0; t < T; ++t)
                k::decode_host(M.embd_type, M.embd + (size_t) tok[t] * rb, C, W.emb_h + (size_t) t * C);
            for (int t = 0; t < T; ++t) W.pos_h[t] = pos0 + t;
        }
        CU(cudaMemcpyAsync(W.tmp, W.emb_h, (size_t) T * C * 4, cudaMemcpyHostToDevice, g_s));
        k::rms(W.tmp, M.embd_norm, T, C, c.eps, W.h, g_s);
        place_pictures(W.h, T, pos0, C, g_s);
        CU(cudaMemcpyAsync(W.pos, W.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
    }
    int nb = 0;
    const size_t TC = (size_t) T * C;
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    auto dump_f = [&](const std::string& name, const float* d, size_t n) {
        if (dump.empty()) return;
        std::vector<float> h(n);
        CU(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen((dump + "/" + name + ".f32").c_str(), "wb");
        if (!f) die("cannot write in " + dump);
        std::fwrite(h.data(), 4, n, f);
        std::fclose(f);
    };

    int nb_pre = 0, n_bank = 0;   // (a looped core: the blocks before it, the states of the pass before's bank)
    for (int vis = 0; vis < c.n_visit; ++vis) {
        const int l = M.V[vis].l;   // (the layer: its weights, its experts in the cache; the state is the visit's)
        Layer& Ly = bind_visit(M, vis);
        if (Ly.dev != g_cur) {   // the next GPU's layers: the stream and the block states go along
            const int from = g_cur;
            Work& F = dev_work(from);
            Work& Tn = dev_work(Ly.dev);
            CU(cudaStreamSynchronize(g_dev[from].s));
            CU(cudaMemcpyPeer(Tn.h, Ly.dev, F.h, from, TC * 4));
            for (int i = 0; i < nb; ++i)
                CU(cudaMemcpyPeer(Tn.blocks + (size_t) i * Tn.blk_ld * C, Ly.dev, F.blocks + (size_t) i * F.blk_ld * C,
                                  from, TC * 4));
            CU(cudaMemcpyPeer(Tn.pos, Ly.dev, F.pos, from, (size_t) T * 4));
            use_dev(Ly.dev);
            Wp = &dev_work(g_cur);
            ECp = &dev_cache(g_cur);
        }
        g_pf.start();
        // A looped core.  Its first pass: the stream it is given is kept.  A later pass starts again from that stream
        // and the blocks before the core, and reads the pass before through messages: that pass's bank is the blocks
        // it added after its first (the first is the kept stream) and its final stream.
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            } else {
                n_bank = 0;
                for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, blk(i), TC * 4, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                ++n_bank;
                CU(cudaMemcpyAsync(W.h, W.lp_pre, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                nb = nb_pre;
            }
        }
        if (M.V[vis].pass > 0) {   // the layer's messages: q's, k's, v's, the MoE's
            k::ResStates st{};
            for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * TC;
            // (two queries a launch: the kernel takes at most three)
            k::attn_res(st, n_bank, Ly.lp_query, 2, T, C, 1e-6f, W.lp_msg, g_s);
            k::attn_res(st, n_bank, Ly.lp_query + (size_t) 2 * C, 2, T, C, 1e-6f, W.lp_msg + 2 * TC, g_s);
            msg = W.lp_msg;
        }
        // the drafter's taps
        if (g_taps) {
            int tap_idx = -1;
            if (c.n_pass > 1) {
                if (vis == 5) tap_idx = 0;
                else if (vis == 23) tap_idx = 1;
                else if (vis == 41) tap_idx = 2;
            } else {
                if (vis % 6 == 5 && vis < c.n_visit - 1) tap_idx = vis / 6;
            }
            if (tap_idx >= 0 && tap_idx < g_taps_n) {
                CU(cudaMemcpyAsync(g_taps + (size_t) tap_idx * g_taps_ld * C, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            }
        }
        run_layer_attn(M, Ly, l, T, pos0, W.h, W.pos, nb, 0, W.r, W, msg);
        dump_f("blk." + std::to_string(vis) + ".attn_out", W.r, TC);
        if (l % c.block == 0) ++nb;
        // the MoE
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {   // (its message gates the input before the norm)
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, W.xq, g_s);
                const k::LoopGate gt{W.xq, msg + (size_t) 3 * TC, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], W.xq};
                loop_gates_chunk(W, &gt, 1, T, C, c.rank, 1e-6f);
                k::rms_f16(W.xq, Ly.ffn_norm, T, C, c.eps, W.x16, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm;
                outs.y16[0] = W.x16;
                outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        g_pf.lap(g_pf.attn);
        gemm16(Ly.router16, c.E, C, W.x16, T, W.rlog, c.E);
        const int S = T * c.K;
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
        if (dp_tok) {   // IK2_DEVPLAN: no wait for the routing (see k_dp_plan, which routes too)
            const int S1 = T * c.K;
            k_dp_plan<<<1, 256, 0, g_s>>>(W.rlog, Ly.probs_b, c.E, c.route_scale, g_dp.tab + (size_t) l * c.E, c.K, C,
                                          W.xq, T, g_dp.rec_d, g_dp.rec_bytes,
                                          g_dp.R, l, g_dp.seqd, g_dp.ctl_d, g_dp.blob, g_dp.tok, g_dp.inv, g_dp.sg,
                                          g_dp.su, g_dp.wt, g_dp.mflag, g_dp.gblob, g_dp.gn, g_dp.gtok, g_dp.grow);
            if (T == 1) {
                k::experts_gate_up_swiglu(g_dp.blob, g_dp.tok, c.K, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, c.F, g_dp.sg,
                                          g_dp.su, c.gate_cap, c.up_cap, W.h16, g_s, 1);
                k::experts_down(g_dp.blob, c.K, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            } else {   // (a verify window: the hits' groups, an expert's weights read once for its tokens)
                k::experts_gate_up_swiglu_g(g_dp.gblob, g_dp.gn, g_dp.gtok, g_dp.grow, S1, Ly.ty[0], Ly.ty[1], Ly.eb[0],
                                            W.xq, C, c.F, g_dp.sg, g_dp.su, c.gate_cap, c.up_cap, W.h16, g_s, T);
                k::experts_down_g(g_dp.gblob, g_dp.gn, g_dp.grow, S1, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F,
                                  W.y32, g_s, S1);
            }
            k_dp_wait<<<1, 32, 0, g_s>>>(g_dp.mflag, S1, g_dp.seqd, g_dp.ctl_d);
            k_dp_merge<<<dim3((unsigned) ((C + 255) / 256), (unsigned) S1), 256, 0, g_s>>>(W.y32, g_dp.ey_d, g_dp.mflag,
                                                                                         g_dp.seqd, C, g_dp.R);
            k::moe_combine_f32(W.y32, g_dp.inv, g_dp.wt, T, c.K, C, W.tmp, false, g_s);
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            continue;
        }
        k::route(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
        const bool cpu_lane = g_pool && g_overlap && !g_prof && T <= 8 && ik2::cpu::available_for(Ly.ty[0]) &&
                              ik2::cpu::available_for(Ly.ty[1]) && ik2::cpu::available_for(Ly.ty[2]);
        const bool fast_ep_token = g_ep_fast && g_ep && T <= 8 && g_overlap && !g_prof &&
                                   dump.empty() && !g_trace && g_pop.empty();
        const int* top = nullptr;
        const float* w = nullptr;
        if (!fast_ep_token) {
            CU(cudaMemcpyAsync(W.route_h, W.route_d, (size_t) S * 8, cudaMemcpyDeviceToHost, g_s));
            if (cpu_lane) CU(cudaMemcpyAsync(W.x_h, W.xq, TC * 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
            top = W.route_h;
            w = (const float*) (W.route_h + S);
            pop_count(l, top, S);
            g_pf.lap(g_pf.route);
            if (!dump.empty()) {   // the experts picked, [T][K] int32 (compare with the reference's blk.L.picks)
                FILE* f = std::fopen((dump + "/blk." + std::to_string(vis) + ".picks.i32").c_str(), "wb");
                if (f) std::fwrite(top, 4, (size_t) S, f), std::fclose(f);
            }
            if (g_trace) {
                const int hdr[3] = {l, T, c.K};
                std::fwrite(hdr, 4, 3, g_trace);
                std::fwrite(top, 4, S, g_trace);
            }
        }
        if (g_tp && T == 1 && g_overlap && !g_prof) {   // tensor-parallel experts, one token
            const double te0 = now();
            const int F = c.F, Fd = F / 2, Cd = C / 2;
            CU(cudaEventRecord(g_ep_x, g_dev[0].s));
            const uint8_t* canon[2][16];
            // (1) each GPU: its halves in (hits run first), its half of h for every pick into tp_h (row = the pick)
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                ExpertCache& Ed = dev_cache(d);
                const uint64_t tick = ++Ed.tick;
                int ord[16], nhit = 0, nmis = 0, mis[16];
                for (int i = 0; i < S; ++i) {
                    bool missed = false;
                    canon[d][i] = Ed.get(Ly, l, top[i], tick, g_cs, &missed, true, false);
                    if (missed) mis[nmis++] = i;
                    else ord[nhit++] = i;
                }
                for (int j = 0; j < nmis; ++j) ord[nhit + j] = mis[j];
                CU(cudaEventRecord(g_ep_cp[d], g_cs));
                const int ps = Wd.pack_slot;
                Wd.pack_slot = (Wd.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = Wd.pack_h[ps];
                uint8_t* pk_d = Wd.pack_d[ps];
                const uint8_t* blob[16];
                int zeros[16] = {}, orow[16];
                float sgv[16], suv[16];
                for (int j = 0; j < S; ++j) {
                    const int i = ord[j], e = top[i];
                    blob[j] = canon[d][i], orow[j] = i;
                    sgv[j] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                    suv[j] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                }
                std::memcpy(pk, blob, S * sizeof(void*));
                std::memcpy(pk + 512, zeros, S * 4);
                std::memcpy(pk + 768, orow, S * 4);
                std::memcpy(pk + 1024, sgv, S * 4);
                std::memcpy(pk + 1280, suv, S * 4);
                std::memcpy(pk + 1536, canon[d], S * sizeof(void*));
                CU(cudaMemcpyAsync(pk_d, pk, 1600, cudaMemcpyHostToDevice, g_s));
                const auto* pb = (const uint8_t* const*) pk_d;
                const int* ptk = (const int*) (pk_d + 512);
                const int* por = (const int*) (pk_d + 768);
                const float* psg = (const float*) (pk_d + 1024);
                const float* psu = (const float*) (pk_d + 1280);
                if (d == 1) {
                    CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
                    CU(cudaMemcpyPeerAsync(Wd.xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                }
                size_t o, b0g;
                tp_part(Ly, d, 0, o, b0g);
                for (int part = 0; part < 2; ++part) {
                    const int p0 = part ? nhit : 0, m = part ? nmis : nhit;
                    if (part) CU(cudaStreamWaitEvent(g_s, g_ep_cp[d], 0));
                    if (!m) continue;
                    k::experts_gate_up_swiglu_rows(pb + p0, ptk + p0, m, Ly.ty[0], Ly.ty[1], b0g, Wd.xq, C, Fd,
                                                   psg + p0, psu + p0, c.gate_cap, c.up_cap, Wd.tp_h + (size_t) d * Fd,
                                                   por + p0, F, g_s);
                }
                CU(cudaEventRecord(g_tp_h[d], g_s));
                Wd.tp_pk = pk_d;
            }
            // (2) each GPU: the other's half of h in, its rows of down; GPU 1's rows to GPU 0, combined there
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                const Work& O = dev_work(1 - d);
                CU(cudaStreamWaitEvent(g_s, g_tp_h[1 - d], 0));
                CU(cudaMemcpy2DAsync(Wd.tp_h + (size_t) (1 - d) * Fd, (size_t) F * 2, O.tp_h + (size_t) (1 - d) * Fd,
                                     (size_t) F * 2, (size_t) Fd * 2, S, cudaMemcpyDefault, g_s));
                size_t o, b0g, b1u;
                tp_part(Ly, d, 0, o, b0g);
                tp_part(Ly, d, 1, o, b1u);
                k::experts_down((const uint8_t* const*) (Wd.tp_pk + 1536), S, Ly.ty[2], b0g + b1u, Wd.tp_h, Cd, F,
                                Wd.y32, g_s);
                if (d == 1) {
                    CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) 8 * C, 0, Wd.y32, 1, (size_t) S * Cd * 4, g_s));
                    CU(cudaEventRecord(g_ep_done, g_s));
                }
            }
            CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            {
                const int ps = W.pack_slot;
                W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = W.pack_h[ps];
                uint8_t* pk_d = W.pack_d[ps];
                int id[16];
                float wt[16];
                for (int i = 0; i < S; ++i) {
                    id[i] = i;
                    wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][top[i]]);
                }
                std::memcpy(pk + 768, id, S * 4);
                std::memcpy(pk + 1536, wt, S * 4);
                CU(cudaMemcpyAsync(pk_d + 768, pk + 768, 1024, cudaMemcpyHostToDevice, g_s));
                k::moe_combine_f32(W.y32, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), 1, c.K, Cd, W.tmp,
                                   false, g_s);
                k::moe_combine_f32(W.y32 + (size_t) 8 * C, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), 1,
                                   c.K, Cd, W.tmp + Cd, false, g_s);
            }
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (fast_ep_token) {
            const int F = c.F;
            static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 1;
            const bool exp_tc = exp_mma && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, F);

            // 1. Send xq (TC*4 B) and route_d (S*8 B) from GPU 0 to GPU 1
            if (g_ep_stage.d_xq[0]) {
                CU(cudaMemcpyAsync(g_ep_stage.d_xq[0], dev_work(0).xq, (size_t) T * C * 4, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(g_ep_stage.d_route[0], dev_work(0).route_d, (size_t) S * 8, cudaMemcpyDeviceToDevice, g_s));
            } else {
                CU(cudaMemcpyPeerAsync(dev_work(1).xq, 1, dev_work(0).xq, 0, (size_t) T * C * 4, g_s));
                CU(cudaMemcpyPeerAsync(dev_work(1).route_d, 1, dev_work(0).route_d, 0, (size_t) S * 8, g_s));
            }
            CU(cudaEventRecord(g_ep_x, g_s));

            // Helper to run experts on a device
            auto run_dev_exp = [&](int /*d*/, Work& Wd, cudaStream_t strm) {
                if (exp_tc) {
                    k::to_f16_limbs_tokens(Wd.xq, T, C, (uint16_t*) Wd.gu32, strm);
                    k::experts_gate_up_swiglu_mma(Wd.blob, S, Ly.eb[0], (const uint16_t*) Wd.gu32, C, F, Wd.sg, Wd.su,
                                                  c.gate_cap, c.up_cap, Wd.h16, strm, Wd.tok);
                    k::experts_down_mma(Wd.blob, S, Ly.eb[0] + Ly.eb[1], Wd.h16, C, F, Wd.y32, strm);
                } else {
                    k::experts_gate_up_swiglu(Wd.blob, Wd.tok, S, Ly.ty[0], Ly.ty[1], Ly.eb[0], Wd.xq, C, F, Wd.sg, Wd.su,
                                              c.gate_cap, c.up_cap, Wd.h16, strm);
                    k::experts_down(Wd.blob, S, Ly.ty[2], Ly.eb[0] + Ly.eb[1], Wd.h16, C, F, Wd.y32, strm);
                }
            };

            // 2. GPU 1: wait for xq & route_d, select resident experts, compute experts, copy result to GPU 0
            use_dev(1);
            CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
            const int* r_top1 = g_ep_stage.d_route[1] ? g_ep_stage.d_route[1] : dev_work(1).route_d;
            const float* r_w1 = g_ep_stage.d_route[1] ? (const float*)(g_ep_stage.d_route[1] + S) : (const float*)(dev_work(1).route_d + S);
            if (g_ep_stage.d_xq[1]) {
                CU(cudaMemcpyAsync(dev_work(1).xq, g_ep_stage.d_xq[1], (size_t) T * C * 4, cudaMemcpyDeviceToDevice, g_s));
            }
            k::ep_select_token(r_top1, r_w1, Ly.ep_tab_blob[1],
                               Ly.ep_tab_sc[1][0], Ly.ep_tab_sc[1][1], Ly.ep_tab_sc[1][2],
                               S, dev_work(1).blob, dev_work(1).tok, dev_work(1).inv, dev_work(1).sg, dev_work(1).su, dev_work(1).wt, g_s, c.K);
            run_dev_exp(1, dev_work(1), g_s);
            if (g_ep_stage.d_y[1]) {
                CU(cudaMemcpyAsync(g_ep_stage.d_y[1], dev_work(1).y32, (size_t) S * C * 4, cudaMemcpyDeviceToDevice, g_s));
            } else {
                CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) S * C, 0, dev_work(1).y32, 1, (size_t) S * C * 4, g_s));
            }
            CU(cudaEventRecord(g_ep_done, g_s));

            // 3. GPU 0: select resident experts, compute experts, wait for GPU 1, combine & conv
            use_dev(0);
            k::ep_select_token(dev_work(0).route_d, (const float*)(dev_work(0).route_d + S), Ly.ep_tab_blob[0],
                               Ly.ep_tab_sc[0][0], Ly.ep_tab_sc[0][1], Ly.ep_tab_sc[0][2],
                               S, dev_work(0).blob, dev_work(0).tok, dev_work(0).inv, dev_work(0).sg, dev_work(0).su, dev_work(0).wt, g_s, c.K);
            run_dev_exp(0, dev_work(0), g_s);
            CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            const float* y_peer = g_ep_stage.d_y[0] ? g_ep_stage.d_y[0] : (dev_work(0).y32 + (size_t) S * C);
            if (T == 1 && !g_rec) {
                k::ep_combine_conv(dev_work(0).y32, y_peer, dev_work(0).blob, dev_work(0).wt,
                                   S, C, Ly.h_mlp, Ly.sc_mlp, dev_work(0).r, dev_work(0).h, g_s);
            } else {
                if (T == 1) {
                    k::ep_combine_token(dev_work(0).y32, y_peer, dev_work(0).blob, dev_work(0).wt,
                                        S, C, dev_work(0).tmp, g_s);
                } else {
                    k::ep_combine_block(dev_work(0).y32, y_peer, dev_work(0).blob, dev_work(0).wt,
                                        T, c.K, C, dev_work(0).tmp, g_s);
                }
                conv_rec(Ly, 3, dev_work(0).tmp, Ly.h_mlp, T, C);
                k::conv4_res(dev_work(0).tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, dev_work(0).r, dev_work(0).h, g_s);
            }
            dump_f("blk." + std::to_string(vis) + ".out", dev_work(0).h, TC);
            continue;
        }
        if (g_ep && T == 1 && g_overlap && !g_prof) {
            const double te0 = now();
            const int F = c.F;
            int row_of[16], n_d[2] = {0, 0};
            const uint8_t* blob_d[2][16];
            float sg_d[2][16], su_d[2][16];
            int dev_of[16];
            for (int i = 0; i < S; ++i) dev_of[i] = ep_owner(l, top[i]);
            // GPU 1 first (it has the farther to go): x over, its picks (hits first, then the copied ones), y back
            CU(cudaEventRecord(g_ep_x, g_dev[0].s));
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                ExpertCache& Ed = dev_cache(d);
                const uint64_t tick = ++Ed.tick;
                int hit[16], mis[16], nhit = 0, nmis = 0;
                const uint8_t* where[16];
                for (int i = 0; i < S; ++i) {
                    if (dev_of[i] != d) continue;
                    bool missed = false;
                    where[i] = Ed.get(Ly, l, top[i], tick, g_cs, &missed, true, false);
                    (missed ? mis[nmis++] : hit[nhit++]) = i;
                }
                CU(cudaEventRecord(g_ep_cp[d], g_cs));
                int n = 0;
                for (int pass = 0; pass < 2; ++pass)
                    for (int j = 0; j < (pass ? nmis : nhit); ++j) {
                        const int i = pass ? mis[j] : hit[j], e = top[i];
                        blob_d[d][n] = where[i];
                        sg_d[d][n] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                        su_d[d][n] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                        row_of[i] = n++;
                    }
                n_d[d] = n;
                if (d == 1) CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
                if (!n) continue;
                const int ps = Wd.pack_slot;
                Wd.pack_slot = (Wd.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = Wd.pack_h[ps];
                uint8_t* pk_d = Wd.pack_d[ps];
                const int zeros[16] = {};
                std::memcpy(pk, blob_d[d], n * sizeof(void*));
                std::memcpy(pk + 512, zeros, n * 4);
                std::memcpy(pk + 1024, sg_d[d], n * 4);
                std::memcpy(pk + 1280, su_d[d], n * 4);
                CU(cudaMemcpyAsync(pk_d, pk, 1536, cudaMemcpyHostToDevice, g_s));
                const auto* pb = (const uint8_t* const*) pk_d;
                const int* ptk = (const int*) (pk_d + 512);
                const float* psg = (const float*) (pk_d + 1024);
                const float* psu = (const float*) (pk_d + 1280);
                if (d == 1) CU(cudaMemcpyPeerAsync(Wd.xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                for (int part = 0; part < 2; ++part) {
                    const int p0 = part ? nhit : 0, m = part ? nmis : nhit;
                    if (part) CU(cudaStreamWaitEvent(g_s, g_ep_cp[d], 0));
                    if (!m) continue;
                    k::experts_gate_up_swiglu(pb + p0, ptk + p0, m, Ly.ty[0], Ly.ty[1], Ly.eb[0], Wd.xq, C, F, psg + p0,
                                              psu + p0, c.gate_cap, c.up_cap, Wd.h16 + (size_t) p0 * F, g_s);
                    k::experts_down(pb + p0, m, Ly.ty[2], Ly.eb[0] + Ly.eb[1], Wd.h16 + (size_t) p0 * F, C, F,
                                    Wd.y32 + (size_t) p0 * C, g_s);
                }
                if (d == 1) {
                    CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) 8 * C, 0, Wd.y32, 1, (size_t) n * C * 4, g_s));
                    CU(cudaEventRecord(g_ep_done, g_s));
                }
            }
            // on GPU 0 (current): the combine over both GPUs' rows (GPU 1's from row 8)
            if (n_d[1]) CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            {
                const int ps = W.pack_slot;
                W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = W.pack_h[ps];
                uint8_t* pk_d = W.pack_d[ps];
                int id[16];
                float wt[16];
                for (int i = 0; i < S; ++i) {
                    id[i] = (dev_of[i] ? 8 : 0) + row_of[i];
                    wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][top[i]]);
                }
                std::memcpy(pk + 768, id, S * 4);
                std::memcpy(pk + 1536, wt, S * 4);
                CU(cudaMemcpyAsync(pk_d + 768, pk + 768, 1024, cudaMemcpyHostToDevice, g_s));
                k::moe_combine_f32(W.y32, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), T, c.K, C, W.tmp,
                                   false, g_s);
            }
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (g_fast && T > 1 && S <= W.rg_slots && !Ly.tab_h.empty()) {
            // Every expert in VRAM and a prompt of a few tokens (a chat turn): the experts straight over their
            // blocks, an expert's blocks read once for up to kGroupMax of the tokens that picked it.  (The GEMM path
            // below decodes a whole expert to FP16 for the two or three tokens it has: a 47-token prompt took 0.34 s.)
            const double te0 = now();
            const int GM = W.rg_groups;
            const size_t o_n = (size_t) GM * 8, o_tok = o_n + (size_t) GM * 4, o_row = o_tok + (size_t) GM * k::kGroupMax * 4,
                         o_sg = o_row + (size_t) GM * k::kGroupMax * 4, o_su = o_sg + (size_t) W.rg_slots * 4,
                         o_wt = o_su + (size_t) W.rg_slots * 4, o_id = o_wt + (size_t) W.rg_slots * 4;
            auto** gb = (const uint8_t**) W.rg_h;
            int *gn = (int*) (W.rg_h + o_n), *gtok = (int*) (W.rg_h + o_tok), *grow = (int*) (W.rg_h + o_row);
            float *sg = (float*) (W.rg_h + o_sg), *su = (float*) (W.rg_h + o_su), *wt = (float*) (W.rg_h + o_wt);
            int* id = (int*) (W.rg_h + o_id);
            if (++W.stamp == 0) {
                std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                W.stamp = 1;
            }
            const uint32_t stp = W.stamp;
            int G = 0;
            for (int i = 0; i < S; ++i) {   // slot i (token i / K, one of its picks): row i of h16 and y32
                const int e = top[i];
                int g = W.gi_stamp[e] == stp ? W.gi_idx[e] : -1;
                if (g < 0 || gn[g] == k::kGroupMax) {   // the expert's first group, or its next
                    g = G++;
                    W.gi_stamp[e] = stp, W.gi_idx[e] = g;
                    gb[g] = Ly.tab_h[e], gn[g] = 0;
                }
                gtok[(size_t) g * k::kGroupMax + gn[g]] = i / c.K;
                grow[(size_t) g * k::kGroupMax + gn[g]] = i;
                ++gn[g];
                sg[i] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                su[i] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
                id[i] = i;
            }
            // (the stream was synchronised after the routing: the copy of the last layer's tables is done)
            CU(cudaMemcpyAsync(W.rg_d, W.rg_h, W.rg_bytes, cudaMemcpyHostToDevice, g_s));
            k::experts_gate_up_swiglu_g((const uint8_t* const*) W.rg_d, (const int*) (W.rg_d + o_n),
                                        (const int*) (W.rg_d + o_tok), (const int*) (W.rg_d + o_row), G, Ly.ty[0],
                                        Ly.ty[1], Ly.eb[0], W.xq, C, c.F, (const float*) (W.rg_d + o_sg),
                                        (const float*) (W.rg_d + o_su), c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_down_g((const uint8_t* const*) W.rg_d, (const int*) (W.rg_d + o_n), (const int*) (W.rg_d + o_row),
                              G, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            k::moe_combine_f32(W.y32, (const int*) (W.rg_d + o_id), (const float*) (W.rg_d + o_wt), T, c.K, C, W.tmp,
                               false, g_s);
            g_pf.lap(g_pf.experts);
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            g_pf.lap(g_pf.rest);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (g_mv && S <= 64) {
            const double te0 = now();
            // the token path: each slot (token t's pick i) its expert's blob straight from the cache
            // The slots already in VRAM first, then the ones copied in now: the first group's experts run while the
            // second's blocks are on their way (the copies on a stream of their own).
            const uint8_t* blob[64];
            int tk[64], id[64];
            float sg[64], su[64], wt[64];
            const uint64_t tick = ++EC.tick;
            const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
            const uint8_t* where[64];
            char late[64] = {};
            int slot_of[64];
            if (++W.stamp == 0) {
                std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                W.stamp = 1;
            }
            const uint32_t stp = W.stamp;
            lazy_poll();
            // (a few tokens - a verify window: every miss on the CPU, grouped by expert, and copied in afterwards)
            const bool multi = cpu_lane && T > 1 && g_verify_cpu &&
                               ik2::cpu::batch_multi_ok(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F);
            const bool lazy_ok = (g_cpu_lazy && g_seq_next - 1 - g_seq_done < (uint64_t) g_lazy_max) ||
                                 (multi && g_seq_next - 1 - g_seq_done < (uint64_t) kRing - 1);
            double acc_up = 0, acc_cpu = 0;
            int urgent[64], lazy[64];
            int n_urgent = 0, n_lazy = 0;
            for (int i = 0; i < S; ++i) {
                const int e = top[i];
                if (W.seen_stamp[e] == stp) {   // a second token of this call on an expert missing in it
                    late[i] = W.seen_late[e];
                    where[i] = EC.has(l, e) ? EC.get(Ly, l, e, tick, cst, nullptr, false, false) : nullptr;
                    slot_of[i] = where[i] ? EC.last_handle : -1;
                    continue;
                }
                if (EC.has(l, e)) {
                    where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                    slot_of[i] = EC.last_handle;
                    if (EC.slot_sseq(EC.last_handle) > g_seq_done) late[i] = 1;   // still arriving from a lazy copy
                    continue;
                }
                // a miss: the CPU's or the link's, whichever is free first
                if (cpu_lane && (multi || acc_cpu + g_t_cpu <= acc_up + g_t_up)) {
                    late[i] = 2, acc_cpu += g_t_cpu;
                    // (a verify window copies in at most g_verify_copies of its misses a visit: the link moves ~1.5
                    //  experts in the window's time, more stalls the host on the copy ring)
                    if (lazy_ok && (!multi || n_lazy < g_verify_copies)) {   // a slot for its blocks, copied in afterwards
                        where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                        slot_of[i] = EC.last_handle;
                        lazy[n_lazy++] = i;
                    } else {            // no slot taken: nothing in VRAM is evicted for it
                        ++EC.misses;
                        where[i] = nullptr, slot_of[i] = -1;
                    }
                } else {
                    late[i] = 1, acc_up += g_t_up;
                    where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                    slot_of[i] = EC.last_handle;
                    urgent[n_urgent++] = i;
                }
                W.seen_stamp[e] = stp;
                W.seen_late[e] = late[i];
            }
            int nh = 0, nw = 0;
            for (int i = 0; i < S; ++i) nh += late[i] == 0, nw += late[i] == 1;
            for (int ui = 0; ui < n_urgent; ++ui) {
                const int i = urgent[ui];
                EC.copy_in(Ly, top[i], (uint8_t*) where[i], cst);
            }
            if (nw && cst != g_s) CU(cudaEventRecord(g_ev, g_cs));
            if (n_lazy > 0) {
                const uint64_t seq = g_seq_next++;
                if (seq > g_seq_done + kRing - 1) {   // the ring's entry is still an unfinished batch's
                    CU(cudaStreamSynchronize(g_cs));
                    g_seq_done = seq - 1;
                }
                for (int li = 0; li < n_lazy; ++li) {
                    const int i = lazy[li];
                    EC.copy_in(Ly, top[i], (uint8_t*) where[i], g_cs);
                    EC.slot_sseq(slot_of[i]) = seq;
                }
                CU(cudaEventRecord(g_ring[seq % kRing], g_cs));
            }
            const int ps = W.pack_slot;
            W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
            uint8_t* pk = W.pack_h[ps];
            uint8_t* pk_d = W.pack_d[ps];
            float* yh = W.y_h[ps];
            int ph = 0, pm = nh, pc = nh + nw;
            ik2::cpu::ExpertJob jobs[64];
            const int n_jobs = S - nh - nw;
            const double tc0 = now();
            for (int i = 0; i < S; ++i) {
                const int e = top[i], p = late[i] == 2 ? pc++ : late[i] ? pm++ : ph++;
                blob[p] = where[i];
                tk[p] = i / c.K;
                id[i] = p;
                sg[p] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                su[p] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
                if (late[i] == 2 && !multi) {
                    auto& J = jobs[p - nh - nw];
                    J.gate = Ly.host[0] + (size_t) e * Ly.stride[0];
                    J.up = Ly.host[1] + (size_t) e * Ly.stride[1];
                    J.down = Ly.host[2] + (size_t) e * Ly.stride[2];
                    J.x = W.x_h + (size_t) (i / c.K) * C;
                    J.tg = Ly.ty[0], J.tu = Ly.ty[1], J.td = Ly.ty[2];
                    J.sg = sg[p], J.su = su[p];
                    J.y = yh + (size_t) (p - nh - nw) * C;
                    ik2::cpu::post_expert(*g_pool, &J, C, c.F, c.gate_cap, c.up_cap);
                }
            }
            // the window's CPU misses as one batch, an expert's tokens a job (its weights read once for all of them)
            ik2::cpu::BatchJob bj[64];
            int n_bj = 0;
            if (multi && n_jobs > 0) {
                int ej[64];
                for (int i = 0; i < S; ++i) {
                    if (late[i] != 2) continue;
                    const int e = top[i], p = id[i];
                    int j = 0;
                    while (j < n_bj && ej[j] != e) ++j;
                    float* y = yh + (size_t) (p - nh - nw) * C;
                    const float* x = W.x_h + (size_t) (i / c.K) * C;
                    if (j == n_bj) {
                        ej[n_bj++] = e;
                        auto& J = bj[j];
                        J = ik2::cpu::BatchJob{};
                        J.gate = Ly.host[0] + (size_t) e * Ly.stride[0];
                        J.up = Ly.host[1] + (size_t) e * Ly.stride[1];
                        J.down = Ly.host[2] + (size_t) e * Ly.stride[2];
                        J.tg = Ly.ty[0], J.tu = Ly.ty[1], J.td = Ly.ty[2];
                        J.sg = sg[p], J.su = su[p];
                        J.x = x, J.y = y;
                    } else if (bj[j].nt < 8) {
                        auto& J = bj[j];
                        J.xt[J.nt] = x, J.yt[J.nt] = y;
                        ++J.nt;
                    } else {   // (more than 8 tokens on one expert: a job more)
                        auto& J = bj[n_bj];
                        J = bj[j];
                        J.nt = 1, J.x = x, J.y = y;
                        ej[n_bj++] = -1;
                    }
                }
            }
            g_n_cpu += n_jobs, g_n_wait += nw;
            // the groups: a part's (0: in VRAM, 1: waited for) slots by expert, each expert's blocks read once
            int ng[2] = {0, 0};
            const uint8_t* gb[64];
            int gn[64];
            int gtok[64 * k::kGroupMax];
            int grow[64 * k::kGroupMax];
            int G = 0;
            for (int part = 0; part < 2; ++part) {
                if (++W.stamp == 0) {
                    std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                    std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                    W.stamp = 1;
                }
                const uint32_t gstp = W.stamp;
                for (int i = 0; i < S; ++i) {
                    if (late[i] != part) continue;
                    const int e = top[i];
                    int g;
                    if (W.gi_stamp[e] != gstp) {
                        W.gi_stamp[e] = gstp;
                        g = G++;
                        W.gi_idx[e] = g;
                        gb[g] = where[i];
                        gn[g] = 0;
                        ++ng[part];
                    } else {
                        g = W.gi_idx[e];
                    }
                    gtok[(size_t) g * k::kGroupMax + gn[g]] = i / c.K;
                    grow[(size_t) g * k::kGroupMax + gn[g]] = id[i];
                    ++gn[g];
                }
            }
            g_pf.lap(g_pf.upload);
            // one copy of the tables from pinned memory: [64] pointers, then tok, inv, sg, su, wt [64] each
            std::memcpy(pk, blob, S * sizeof(void*));
            std::memcpy(pk + 512, tk, S * 4);
            std::memcpy(pk + 768, id, S * 4);
            std::memcpy(pk + 1024, sg, S * 4);
            std::memcpy(pk + 1280, su, S * 4);
            std::memcpy(pk + 1536, wt, S * 4);
            // ... and the groups: [64] pointers, n [64], tok [64][8], row [64][8]
            std::memcpy(pk + 1792, gb, G * sizeof(void*));
            std::memcpy(pk + 2304, gn, G * 4);
            std::memcpy(pk + 2560, gtok, (size_t) G * k::kGroupMax * 4);
            std::memcpy(pk + 4608, grow, (size_t) G * k::kGroupMax * 4);
            CU(cudaMemcpyAsync(pk_d, pk, 6656, cudaMemcpyHostToDevice, g_s));
            const auto* pgb = (const uint8_t* const*) (pk_d + 1792);
            const int* pgn = (const int*) (pk_d + 2304);
            const int* pgt = (const int*) (pk_d + 2560);
            const int* pgr = (const int*) (pk_d + 4608);
            const auto* pb = (const uint8_t* const*) pk_d;
            const int* ptk = (const int*) (pk_d + 512);
            const int* pid = (const int*) (pk_d + 768);
            const float* psg = (const float*) (pk_d + 1024);
            const float* psu = (const float*) (pk_d + 1280);
            const float* pwt = (const float*) (pk_d + 1536);
            const int F = c.F;
            for (int part = 0; part < 2; ++part) {
                const int p0 = part ? nh : 0, n = part ? nw : nh;
                if (part && cst != g_s) CU(cudaStreamWaitEvent(g_s, g_ev, 0));
                if (!n) continue;
                if (g_group && g_group_exp && T > 1) {
                    const int g0 = part ? ng[0] : 0, gc = ng[part];
                    k::experts_gate_up_swiglu_g(pgb + g0, pgn + g0, pgt + (size_t) g0 * k::kGroupMax,
                                                pgr + (size_t) g0 * k::kGroupMax, gc, Ly.ty[0], Ly.ty[1], Ly.eb[0],
                                                W.xq, C, F, psg, psu, c.gate_cap, c.up_cap, W.h16, g_s, T);
                    k::experts_down_g(pgb + g0, pgn + g0, pgr + (size_t) g0 * k::kGroupMax, gc, Ly.ty[2],
                                      Ly.eb[0] + Ly.eb[1], W.h16, C, F, W.y32, g_s, S);
                    continue;
                }
                k::experts_gate_up_swiglu(pb + p0, ptk + p0, n, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, F, psg + p0,
                                          psu + p0, c.gate_cap, c.up_cap, W.h16 + (size_t) p0 * F, g_s);
                k::experts_down(pb + p0, n, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16 + (size_t) p0 * F, C, F,
                                W.y32 + (size_t) p0 * C, g_s);
            }
            if (n_jobs > 0) {
                if (multi) ik2::cpu::experts_batch(bj, n_bj, C, F, c.gate_cap, c.up_cap);   // (the GPU's groups run)
                else g_pool->wait();
                g_t_cpu = 0.9 * g_t_cpu + 0.1 * (now() - tc0) / n_jobs;
                CU(cudaMemcpyAsync(W.y32 + (size_t) (nh + nw) * C, yh, (size_t) n_jobs * C * 4, cudaMemcpyHostToDevice,
                                   g_s));
            }
            k::moe_combine_f32(W.y32, pid, pwt, T, c.K, C, W.tmp, false, g_s);
            g_pf.lap(g_pf.experts);
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            g_pf.lap(g_pf.rest);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        run_prefill_moe(M, Ly, l, T, W.x16, top, w, W.r, W.h, W, EC, t_exp);
        g_pf.lap(g_pf.rest);
        dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
    }
    const double t_enq = dp_tok && g_dp_time ? now() : 0;
    run_head(M, T, all_logits, out_mode, nb, W.h, W, logits_out, argmax_out);
    if (dp_tok && g_dp_time && !g_tg_capturing) {   // IK2_DP_TIME=1: the host's enqueue time of a token against the token's
        g_dp_t_enq += t_enq - g_dp_t0, g_dp_t_tok += now() - g_dp_t0, ++g_dp_n;
        if (g_dp_n % 64 == 0)
            std::fprintf(stderr, "devplan time a token: %.2f ms enqueued, %.2f ms in all\n", g_dp_t_enq / g_dp_n * 1e3,
                         g_dp_t_tok / g_dp_n * 1e3);
    }
#undef W
#undef EC
}

// Layer-major super-chunk prefill (single GPU, no dump, no drafter):
// A long prompt in fewer passes: the prefill scratch for a pass of T tokens (~256 KB a token) is lent by the expert
// cache (ExpertCache::lend), which a long prefill streams through anyway, and given back afterwards - so the passes
// can be as long as the cache allows without the decode's cache being smaller (IK2_BORROW=0: off).
struct BigScratch {
    int pool = -1, T_sup = 0;
    size_t blk_ld = 0;
    float *h = nullptr, *r = nullptr, *tmp = nullptr, *blocks = nullptr, *gu32 = nullptr, *qkv = nullptr,
          *y32 = nullptr, *sg = nullptr, *su = nullptr, *wt = nullptr, *wtr = nullptr, *rlog = nullptr;
    uint16_t *xq16_sup = nullptr, *ex16 = nullptr, *h16 = nullptr;
    int *pos = nullptr, *route_d = nullptr, *gidx = nullptr, *inv = nullptr;
    float* lg = nullptr;
    uint16_t* lg16 = nullptr;
    size_t lg_cap = 0;
};
// the scratch of a T-token pass, carved from base (or only its size, base null)
int g_sub_max = 512;   // a layer-major pass's attention sub-chunk (IK2_SUB)
size_t big_carve(const Config& c, int T, size_t gu32_have, size_t lg_want, size_t lg_have, Work* W, uint8_t* base) {
    const size_t C = c.C, F = c.F, S = (size_t) T * c.K, nblk = c.n_layer / c.block + 1;
    const size_t Tr = std::max<size_t>(64, T);
    size_t off = 0;
    auto take = [&](size_t bytes) {
        uint8_t* p = base ? base + off : nullptr;
        off += (bytes + 255) / 256 * 256;
        return p;
    };
    float* h = (float*) take(T * C * 4);
    float* r = (float*) take(T * C * 4);
    float* tmp = (float*) take(T * C * 4);
    float* blocks = (float*) take(nblk * T * C * 4);
    uint16_t* xq = (uint16_t*) take(T * C * 2);
    int* pos = (int*) take((size_t) T * 4);
    float* rlog = (float*) take((size_t) T * c.E * 4);
    int* route = (int*) take(S * 8);
    uint16_t* ex16 = (uint16_t*) take(T * C * 2);
    uint16_t* h16 = (uint16_t*) take(Tr * F * 2);
    const bool new_gu = (size_t) T * 2 * F * 4 > gu32_have;
    float* gu32 = new_gu ? (float*) take((size_t) T * 2 * F * 4) : nullptr;
    float* y32 = (float*) take(Tr * C * 4);
    float* sg = (float*) take(S * 4);
    float* su = (float*) take(S * 4);
    float* wt = (float*) take(S * 4);
    float* wtr = (float*) take(S * 4);
    int* gidx = (int*) take(S * 4);
    int* inv = (int*) take(S * 4);
    float *lpre = nullptr, *lbank = nullptr, *lmsg = nullptr, *lsub = nullptr;
    if (c.n_pass > 1) {   // a looped core's pass state (see Work::lp_pre_s)
        lpre = (float*) take(T * C * 4);
        lbank = (float*) take((size_t) ((c.core1 - c.core0) / c.block) * T * C * 4);
        lmsg = (float*) take(4 * T * C * 4);
        lsub = (float*) take((size_t) 4 * g_sub_max * C * 4);
    }
    // the attention's logits and probabilities, room for kAttnRows queries at the prompt's end (lg_want of each)
    const bool new_lg = lg_want > lg_have;
    float* lg = new_lg ? (float*) take(lg_want * 4) : nullptr;
    uint16_t* lg16 = new_lg ? (uint16_t*) take(lg_want * 2) : nullptr;
    if (W && base) {
        W->h = h, W->r = r, W->tmp = tmp, W->blocks = blocks, W->xq16_sup = xq, W->pos = pos, W->rlog = rlog;
        W->route_d = route, W->ex16 = ex16, W->h16 = h16, W->y32 = y32;
        W->sg = sg, W->su = su, W->wt = wt, W->wtr = wtr, W->gidx = gidx, W->inv = inv;
        if (new_gu) W->gu32 = gu32, W->qkv = gu32, W->lg16 = (uint16_t*) gu32;
        if (new_lg) W->lg = lg, W->lg16 = lg16, W->lg_cap = lg_want;
        W->blk_ld = (size_t) T;
        W->T_sup = T;
        W->lp_pre_s = lpre, W->lp_bank_s = lbank, W->lp_msg_s = lmsg, W->lp_sub = lsub;
    }
    return off;
}
// Tokens a pass for a prompt of n new tokens: W.T_sup, or more with a lent scratch (then big_end gives it back).
int big_begin(const Model& M, Work& W, ExpertCache& EC, int n, int ctx_end, BigScratch& bs) {
    bs.pool = -1;
    kv_ensure(const_cast<Model&>(M), ctx_end);   // (--kv-grow takes the pool's top slots: before any are lent)
    if (W.T_sup <= 0 || dual() || (getenv("IK2_BORROW") && !std::atoi(getenv("IK2_BORROW")))) return W.T_sup;
    dp_quiesce();   // (no copy of the plan's may land in a slot about to be lent)
    const Config& c = M.c;
    // the attention's scratch for 256 queries against the whole context: few, large GEMMs (the default ~400 MB left
    // room for only ~20 queries a step at 100k tokens) - also lent for a short prompt at the end of a long context
    const size_t lg_want = (g_flash && !g_kv_fp8 && c.D == 128) ? 0 : (size_t) c.H * 256 * (size_t) ctx_end;
    const bool long_ctx = lg_want > W.lg_cap;
    if (n <= W.T_sup && !long_ctx) return W.T_sup;
    int p = -1;
    size_t most = 0;
    for (size_t i = 0; i < EC.pools.size(); ++i) {
        const size_t b = (size_t) EC.pools[i].n_active * EC.pools[i].slot_bytes;
        if (b > most) most = b, p = (int) i;
    }
    if (p < 0) return W.T_sup;
    const auto& P = EC.pools[p];
    const int k_max = P.n_active - (2 * c.E + 80);   // (prefetch_layer still needs two layers' slots)
    if (k_max <= 0) return W.T_sup;
    const size_t avail = (size_t) k_max * P.slot_bytes;
    const size_t lg_bytes = long_ctx ? (lg_want * 4 + 255) / 256 * 256 + lg_want * 2 + 256 : 0;
    if (lg_bytes >= avail) return W.T_sup;
    int T = W.T_sup;
    if (n > W.T_sup) {
        const size_t per_tok = big_carve(c, 1 << 14, W.gu32_bytes, 0, 0, nullptr, nullptr) >> 14;
        const int T_cap = (int) std::min<size_t>((size_t) W.T_host, (avail - lg_bytes) / (per_tok + 1));
        if (T_cap > W.T_sup) {
            const int passes = (n + T_cap - 1) / T_cap;
            T = (n + passes - 1) / passes;
            while (T > W.T_sup && big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, nullptr, nullptr) > avail) T -= 256;
            if (T < W.T_sup) T = W.T_sup;
        }
    }
    if (T == W.T_sup && !long_ctx) return W.T_sup;
    const size_t bytes = T > W.T_sup ? big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, nullptr, nullptr) : lg_bytes;
    const int k = (int) ((bytes + P.slot_bytes - 1) / P.slot_bytes);
    CU(cudaStreamSynchronize(g_cs));
    CU(cudaStreamSynchronize(g_s));
    g_seq_done = g_seq_next - 1;
    uint8_t* base = EC.lend(p, k);
    if (!base) return W.T_sup;
    bs.h = W.h, bs.r = W.r, bs.tmp = W.tmp, bs.blocks = W.blocks, bs.gu32 = W.gu32, bs.qkv = W.qkv, bs.y32 = W.y32;
    bs.sg = W.sg, bs.su = W.su, bs.wt = W.wt, bs.wtr = W.wtr, bs.rlog = W.rlog, bs.xq16_sup = W.xq16_sup;
    bs.ex16 = W.ex16, bs.h16 = W.h16, bs.pos = W.pos, bs.route_d = W.route_d, bs.gidx = W.gidx, bs.inv = W.inv;
    bs.blk_ld = W.blk_ld, bs.T_sup = W.T_sup;
    bs.lg = W.lg, bs.lg16 = W.lg16, bs.lg_cap = W.lg_cap;
    if (T > W.T_sup) {
        big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, &W, base);
    } else {   // only the attention's scratch
        W.lg = (float*) base;
        W.lg16 = (uint16_t*) (base + (lg_want * 4 + 255) / 256 * 256);
        W.lg_cap = lg_want;
    }
    bs.pool = p;
    return T;
}
void big_end(Work& W, ExpertCache& EC, BigScratch& bs) {
    if (bs.pool < 0) return;
    CU(cudaStreamSynchronize(g_cs));
    CU(cudaStreamSynchronize(g_s));
    W.h = bs.h, W.r = bs.r, W.tmp = bs.tmp, W.blocks = bs.blocks, W.gu32 = bs.gu32, W.qkv = bs.qkv, W.y32 = bs.y32;
    W.sg = bs.sg, W.su = bs.su, W.wt = bs.wt, W.wtr = bs.wtr, W.rlog = bs.rlog, W.xq16_sup = bs.xq16_sup;
    W.ex16 = bs.ex16, W.h16 = bs.h16, W.pos = bs.pos, W.route_d = bs.route_d, W.gidx = bs.gidx, W.inv = bs.inv;
    W.blk_ld = bs.blk_ld, W.T_sup = bs.T_sup;
    W.lg = bs.lg, W.lg16 = bs.lg16, W.lg_cap = bs.lg_cap;
    W.lp_pre_s = W.lp_bank_s = W.lp_msg_s = W.lp_sub = nullptr;
    EC.give_back(bs.pool);
    bs.pool = -1;
}

// Sweeps each layer once across up to W.T_sup tokens (sub-chunked attention for KV/attention scratch, then one combined MoE pass).
// a decode token's graph for position pos into exec slot (captured, not run)
void tg_capture(Model& M, int pos, int slot) {
    kv_ensure(M, pos + 1);   // (a growth synchronizes: not inside a capture)
    if (g_dp.dirty) {   // (a growth made the plan's table stale: this token runs without a graph, which rebuilds it)
        g_dp.exec_pos[slot] = -1;
        return;
    }
    g_tg_capturing = true;
    CU(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
    const int tok = 0;
    std::vector<float> lg;
    std::vector<int> ag;
    double te = 0;
    forward(M, &tok, 1, pos, false, OutMode::kArgmax, std::string(), lg, &ag, te);
    cudaGraph_t g;
    CU(cudaStreamEndCapture(g_s, &g));
    g_tg_capturing = false;
    cudaGraphExec_t& ex = g_dp.exec[slot];
    bool updated = false;
    if (ex) {
        cudaGraphExecUpdateResultInfo info;
        updated = cudaGraphExecUpdate(ex, g, &info) == cudaSuccess;
        if (!updated) {
            cudaGetLastError();
            CU(cudaGraphExecDestroy(ex));
            ex = nullptr;
        }
    }
    if (!updated) CU(cudaGraphInstantiate(&ex, g, 0));
    CU(cudaGraphDestroy(g));
    g_dp.exec_pos[slot] = pos;
}
void forward_prefill_sup(Model& M, const int* tok, int T, int pos0, int sub_chunk, OutMode out_mode,
                         std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C;
    use_dev(M.L[0].dev);
    Work& W = dev_work(g_cur);
    ExpertCache& EC = dev_cache(g_cur);
    const bool looped = c.n_pass > 1;
    if (looped && !W.lp_msg_s) {   // (no lent pass for a looped model: forward()'s chunks)
        for (int p0 = 0; p0 < T; p0 += W.Tm) {
            const int n = std::min(W.Tm, T - p0);
            forward(M, tok + p0, n, pos0 + p0, false, p0 + n == T ? out_mode : OutMode::kNone, std::string(), logits_out,
                    argmax_out, t_exp);
        }
        return;
    }
    kv_ensure(M, pos0 + T);
    if (g_dp.M && !g_dp.dirty) {   // (the device plan's copies drained and written back first: this changes the cache)
        CU(cudaDeviceSynchronize());
        g_dp.quiesce.store(1, std::memory_order_release);
        while (g_dp.quiesce.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        g_dp.quiesce.store(0, std::memory_order_release);
    }
    g_dp.dirty = true;

    // Embed and norm all T tokens in sub-chunks of <= W.Tm
    {
        const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        for (int off = 0; off < T; off += W.Tm) {
            const int ts = std::min(W.Tm, T - off);
            for (int t = 0; t < ts; ++t)
                k::decode_host(M.embd_type, M.embd + (size_t) tok[off + t] * rb, C, W.emb_h + (size_t) t * C);
            CU(cudaMemcpyAsync(W.tmp, W.emb_h, (size_t) ts * C * 4, cudaMemcpyHostToDevice, g_s));
            k::rms(W.tmp, M.embd_norm, ts, C, c.eps, W.h + (size_t) off * C, g_s);
            place_pictures(W.h + (size_t) off * C, ts, pos0 + off, C, g_s);
            if (off + ts < T) CU(cudaStreamSynchronize(g_s));   // (W.emb_h reused on next iteration)
        }
        for (int t = 0; t < T; ++t) W.pos_h[t] = pos0 + t;
        CU(cudaMemcpyAsync(W.pos, W.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
    }

    // (enough tokens that a layer uses most of its experts; below, the misses copied after the routing cost less than
    // prefetching experts that turn out unused - a 119-token turn took 3.4 s with 2E, 1.9 s without)
    const bool pf = (size_t) T * c.K >= (size_t) 8 * c.E;
    // (not for a looped model: a pass comes back to the core's layers, and evicting its own experts first could take a
    //  slot a revisit has just been given as a hit - wrong values; KL 0.48 against forward()'s chunks)
    EC.scan = pf && !looped && !(getenv("IK2_SCAN") && !std::atoi(getenv("IK2_SCAN")));
    if (dual()) dev_cache(1).scan = EC.scan;
    // (under EP each GPU prefetches the experts it owns, by GPU 0's picks so far)
    // One layer ahead: two or three layers in flight cost more than they hide - their slots squeeze the cache (2x RTX
    // PRO 6000, a 21.8k prompt: copies 244 -> 328 GiB, 8.05 -> 9.73 s); the next layer's prefetch issued before this
    // layer's routing round trip instead of after its experts are queued was slower too
    const int depth = 1;
    auto prefetch = [&](int ll, int ahead) {
        if (!dual()) {
            prefetch_layer(M, ll, W, EC, T, nullptr, -1, ahead);
            return;
        }
        use_dev(1);
        prefetch_layer(M, ll, dev_work(1), dev_cache(1), T, &W, g_tp ? -1 : 1, ahead);
        use_dev(0);
        prefetch_layer(M, ll, W, EC, T, nullptr, g_tp ? -1 : 0, ahead);
    };
    if (pf)
        for (int ll = 0; ll < depth; ++ll) prefetch(ll, ll + 1);
    int nb = 0, nb_pre = 0, n_bank = 0;
    const size_t TC = (size_t) T * C;
    for (int vis = 0; vis < c.n_visit; ++vis) {
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        g_pf.start();
        // a looped core (as forward()): its first pass keeps the stream it is given; a later pass starts again from it
        // and reads the pass before through messages, from that pass's bank (its blocks after the first, its stream)
        const float* msg = nullptr;
        if (looped && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre_s, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            } else {
                n_bank = 0;
                for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                    CU(cudaMemcpyAsync(W.lp_bank_s + (size_t) n_bank * TC, W.blocks + (size_t) i * W.blk_ld * C, TC * 4,
                                       cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(W.lp_bank_s + (size_t) n_bank * TC, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                ++n_bank;
                CU(cudaMemcpyAsync(W.h, W.lp_pre_s, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                nb = nb_pre;
            }
        }
        if (looped && M.V[vis].pass > 0) {
            k::ResStates st{};
            for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank_s + (size_t) i * TC;
            k::attn_res(st, n_bank, Ly.lp_query, 2, T, C, 1e-6f, W.lp_msg_s, g_s);
            k::attn_res(st, n_bank, Ly.lp_query + (size_t) 2 * C, 2, T, C, 1e-6f, W.lp_msg_s + 2 * TC, g_s);
            msg = W.lp_msg_s;
        }
        for (int off = 0; off < T; off += sub_chunk) {
            const int ts = std::min(sub_chunk, T - off);
            const float* msg_s = nullptr;
            if (msg) {   // (the sub-chunk's q / k / v messages, [3][ts][C] as run_layer_attn reads them; the MoE's
                for (int j = 0; j < 3; ++j)  //  row 3 is read from lp_msg_s below)
                    CU(cudaMemcpyAsync(W.lp_sub + (size_t) j * ts * C, msg + j * TC + (size_t) off * C, (size_t) ts * C * 4,
                                       cudaMemcpyDeviceToDevice, g_s));
                msg_s = W.lp_sub;
            }
            run_layer_attn(M, Ly, l, ts, pos0 + off, W.h + (size_t) off * C, W.pos + off, nb, off,
                           W.r + (size_t) off * C, W, msg_s);
        }
        if (l % c.block == 0) ++nb;
        if (msg) {   // the MoE's input gated by its message before the norm
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + (size_t) i * W.blk_ld * C;
            st.st[nb] = W.r;
            k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, W.tmp, g_s);
            for (int off = 0; off < T; off += sub_chunk) {
                const int ts = std::min(sub_chunk, T - off);
                float* xo = W.tmp + (size_t) off * C;
                const k::LoopGate gt{xo, msg + 3 * TC + (size_t) off * C, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], xo};
                loop_gates_chunk(W, &gt, 1, ts, C, c.rank, 1e-6f);
            }
            k::rms_f16(W.tmp, Ly.ffn_norm, T, C, c.eps, W.xq16_sup, W.tmp, g_s);
        } else {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + (size_t) i * W.blk_ld * C;
            st.st[nb] = W.r;
            k::RmsOuts outs{};
            outs.w[0] = Ly.ffn_norm;
            outs.y16[0] = W.xq16_sup;
            k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
        }
        g_pf.lap(g_pf.attn);
        gemm16(Ly.router16, c.E, C, W.xq16_sup, T, W.rlog, c.E);
        const int S = T * c.K;
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
        k::route(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
        CU(cudaMemcpyAsync(W.route_h, W.route_d, (size_t) S * 8, cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        const int* top_h = W.route_h;
        const float* w_h = (const float*) (W.route_h + S);
        pop_count(l, top_h, S);
        g_pf.lap(g_pf.route);
        if (g_trace) {
            const int hdr[3] = {l, T, c.K};
            std::fwrite(hdr, 4, 3, g_trace);
            std::fwrite(top_h, 4, S, g_trace);
        }
        run_prefill_moe(M, Ly, l, T, W.xq16_sup, top_h, w_h, W.r, W.h, W, EC, t_exp);
        if (pf) prefetch(l + depth, depth);
        g_pf.lap(g_pf.rest);
    }
    EC.scan = false;
    if (dual()) dev_cache(1).scan = false;

    if (out_mode != OutMode::kNone) {
        g_pf.start();
        const int last = T - 1;
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + ((size_t) i * W.blk_ld + last) * C;
        st.st[nb] = W.h + (size_t) last * C;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm;
        outs.y16[0] = W.x16;
        outs.y32[0] = g_dense_q8 ? W.xq : nullptr;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
        matmul(M.out_w, W.xq, W.x16, 1, W.logits, c.n_rows);
        if (out_mode == OutMode::kArgmax) {
            argmax_out->resize(1);
            k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d, g_s);
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
            (*argmax_out)[0] = W.argmax_h[0];
        } else {
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            const size_t at = logits_out.size();
            logits_out.resize(at + (size_t) c.n_vocab);
            CU(cudaMemcpyAsync(logits_out.data() + at, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
        }
        g_pf.lap(g_pf.rest);
    } else {
        CU(cudaStreamSynchronize(g_s));
    }
}

// every short conv's history as after the first n of the verify window's tokens
void rollback(Model& M, int n) {
    const int C = M.c.C, KD = M.c.KV * M.c.D;
    for (int vis = 0; vis < (int) M.V.size(); ++vis) {
        Layer& Ly = bind_visit(M, vis);
        float* hist[4] = {Ly.h_k, Ly.h_v, Ly.h_attn, Ly.h_mlp};
        const int cc[4] = {KD, KD, C, C};
        for (int i = 0; i < 4; ++i) k::conv_hist_after(Ly.rec_old[i], Ly.rec_x[i], n, cc[i], hist[i], g_s);
    }
}

// the embedding row of token t, normed as the model's input, into out (device [C])
void embed_row(Model& M, int t, float* raw, float* out) {
    const int C = M.c.C;
    float* e = dev_work(M.L[0].dev).emb_h;
    const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    k::decode_host(M.embd_type, M.embd + (size_t) t * rb, C, e);
    CU(cudaMemcpyAsync(raw, e, (size_t) C * 4, cudaMemcpyHostToDevice, g_s));
    k::rms(raw, M.embd_norm, 1, C, M.c.eps, out, g_s);
}

}  // namespace

namespace ik2 {

int run(int argc, char** argv) {
    std::string model, toks, dump, draft_path;
    int gen = 16, chunk = 2048, ctx = 262144, threads = 8, n_gpu = 1;
    bool threads_given = false;
    bool chunk_given = false;   // (--chunk N; auto, or none: the default, smaller on a small card)
    int kv_grow_arg = -1;   // --kv-grow 0 / 1 (default: on a card under 16 GB)
    bool serve = false, serve_strata = true;   // (--serve: the engine protocol; --serve-ik2: the @-line one)
    std::vector<int> eos_ids;                 // --eos a,b: the tokens a request ends at (the engine protocol)
    double cache_gb = 60;
    double vram_reserve_mib = 0;
    bool vram_reserve_given = false;   // (an explicit 0 leaves 0: see the expert cache's room)
    int image_token = 200054;   // --image-token: Inkling 2's <|unused_200054|>, the template's placeholder of a picture
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value of " + a);
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--tokens") {   // (--tokens @FILE: the ids from a file - a long prompt passes a shell's limit)
            toks = next();
            if (!toks.empty() && toks[0] == '@') {
                std::ifstream in(toks.substr(1));
                toks.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
                while (!toks.empty() && std::isspace((unsigned char) toks.back())) toks.pop_back();
            }
        }
        else if (a == "--gen" || a == "--max-new") gen = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") {
            const std::string v = next();
            if (v == "auto") chunk = 2048;
            else chunk = std::max(1, std::atoi(v.c_str())), chunk_given = true;
        }
        else if (a == "--vram-reserve-mib") vram_reserve_mib = std::atof(next().c_str()), vram_reserve_given = true;
        else if (a == "--kv-grow") kv_grow_arg = std::atoi(next().c_str());
        else if (a == "--ctx" || a == "--max-context") ctx = std::atoi(next().c_str());   // (--max-context: serve/server.py's name)
        else if (a == "--cache-gb" || a == "--expert-vram") cache_gb = std::atof(next().c_str());
        else if (a == "--threads" || a == "--cpu-threads" || a == "--pool-workers") threads = std::atoi(next().c_str()), threads_given = true;
        else if (a == "--gpus") n_gpu = std::atoi(next().c_str());
        else if (a == "--draft") draft_path = next();
        else if (a == "--serve") serve = true;
        else if (a == "--serve-ik2") serve = true, serve_strata = false;
        else if (a == "--eos") {
            const std::string v = next();
            for (const char* p = v.c_str(); *p;) {
                eos_ids.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
        else if (a == "--image-token") image_token = std::atoi(next().c_str());
        else if (a == "--dump") dump = next();
        else if (a.rfind("--", 0) == 0) {
            // Strata server passes generic flags (e.g. --layer-split, --vram-reserve-mib, etc.): skip unrecognized flag and its value if any
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) ++i;
        }
        else die("unknown argument " + a);
    }
    auto getenv_str = [](const char* name) -> std::string {
        const char* val = getenv(name);
        if (!val) return "";
        std::string s(val);
        while (!s.empty() && std::isspace((unsigned char) s.back())) s.pop_back();
        while (!s.empty() && std::isspace((unsigned char) s.front())) s.erase(0, 1);
        return s;
    };
    g_prof = getenv("IK2_PROF") && std::atoi(getenv("IK2_PROF"));
    g_flash = !(getenv("IK2_FLASH") && !std::atoi(getenv("IK2_FLASH")));
    g_mv = !getenv("IK2_MV") || std::atoi(getenv("IK2_MV"));
    g_group = !getenv("IK2_GROUP") || std::atoi(getenv("IK2_GROUP"));
    if (getenv("IK2_GROUP")) {
        const int v = std::atoi(getenv("IK2_GROUP"));
        g_group_exp = v == 1 || v >= 3, g_group_attn = v >= 2;
    }
    g_dense_q8 = getenv_str("IK2_DENSE") == "q8";
    // the K / V cache: int8 by default (a scale per 64 values; half FP16's bytes, KL ~0.008 median of FP16's on
    // Inkling-2-Mini: about the cost of rounding to 11 bits), IK2_KV=f16 / fp8 otherwise
    g_kv_i8 = !getenv("IK2_KV");
    if (getenv("IK2_KV")) {
        const std::string kv = getenv_str("IK2_KV");
        if (kv == "f16" || kv == "fp16") g_kv_fp8 = false;
        else if (kv == "fp8" || kv == "e4m3") g_kv_fp8 = true;
        else if (kv == "i8" || kv == "int8") g_kv_i8 = true;
        else die("unknown IK2_KV=" + kv + " (expected fp8, i8 or f16)");
        if (g_kv_fp8) {   // (the prompt attention's FP8 tiles need more shared memory than a block gets before sm_80)
            int mj = 99;
            int n_dev = 0;
            CU(cudaGetDeviceCount(&n_dev));
            for (int d = 0; d < std::max(1, std::min(n_gpu, n_dev)); ++d) {   // (every GPU that may hold layers)
                int m = 0;
                CU(cudaDeviceGetAttribute(&m, cudaDevAttrComputeCapabilityMajor, d));
                mj = std::min(mj, m);
            }
            if (mj < 8) {
                std::printf("IK2_KV=fp8: not on a compute capability %d GPU (the prefill attention needs sm_80); FP16 K / V\n", mj);
                g_kv_fp8 = false;
            }
        }
    }
    if (getenv("IK2_TRACE") && *getenv("IK2_TRACE")) {
        g_trace = std::fopen(getenv("IK2_TRACE"), "wb");
        if (!g_trace) die(std::string("cannot write ") + getenv("IK2_TRACE"));
    }
    if (model.empty() || (toks.empty() && !serve))
        die("usage: strata-ik2 --model FIRST.gguf (--tokens 1,2,3 [--gen N] | --serve) [--chunk 2048] [--ctx N] ...");
    std::vector<int> prompt;
    for (const char* p = toks.c_str(); *p;) {
        prompt.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    const double t_start = now();
    Shards S(model);
    Model M;
    Config& c = M.c;
    // the architecture: "inkling2" (PR #32187 and unsloth's files), or "inkling2_loop" (llama.cpp's merged Mini
    // layout, #32694: block_count counts layers, loop.count / loop.start / loop.end, a layer's message queries and
    // gates under blk.N)
    const auto* arch = S.get("general.architecture");
    const bool loop_arch = arch && arch->s == "inkling2_loop";
    const std::string a = loop_arch ? "inkling2_loop." : "inkling2.";
    // a key under PR #32187's name, else under llama.cpp master's; a value master writes a visit (an array of equal
    // values) read as one
    auto key = [&](const char* k1, const char* k2 = nullptr) -> double {
        const auto* v = S.get(a + k1);
        if (!v && k2) v = S.get(a + k2);
        if (!v) die(std::string("no key ") + a + k1);
        return v->items.empty() ? v->num() : v->items[0].num();
    };
    c.n_visit = (int) key("block_count");
    c.C = (int) key("embedding_length");
    c.H = (int) key("attention.head_count");
    c.KV = (int) key("attention.head_count_kv");
    c.D = (int) key("attention.key_length");
    c.E = (int) key("expert_count");
    c.K = (int) key("expert_used_count");
    c.F = (int) key("expert_feed_forward_length");
    c.window = (int) key("attention.sliding_window");
    c.d_rel = (int) key("d_rel", "attention.relative_length");
    c.extent = (int) key("rel_extent", "attention.relative_extent");
    c.extent_swa = S.get(a + "rel_extent_swa") ? (int) key("rel_extent_swa") : c.window;   // (master: the window)
    c.block = (int) key("attn_res.block_size");
    c.n_vocab = (int) key("unpadded_vocab_size", "vocab_size_unpadded");
    c.eps = (float) key("attention.layer_norm_rms_epsilon");
    c.route_scale = (float) key("expert_weights_scale");
    c.qk_clip = (float) key("qk_clip_threshold", "attention.qk_clip");
    c.rel_clip = (float) key("rel_clip_threshold", "attention.relative_clip");
    c.alpha = (float) key("log_scaling_alpha", "attention.log_scale_alpha");
    c.n_floor = (float) key("log_scaling_n_floor", "attention.log_scale_floor");
    c.gate_cap = (float) key("mlp_gate_softcap", "activation.situ_beta");
    c.up_cap = (float) key("mlp_up_softcap", "activation.situ_linear_beta");
    c.logit_cap = (float) key("final_logit_softcapping");
    // the loop (Inkling-2-Mini): block_count counts the layer visits, the core's layers once a pass
    if (S.get(a + "num_loops")) c.n_pass = (int) key("num_loops");
    if (loop_arch && S.get(a + "loop.count")) c.n_pass = (int) key("loop.count");
    if (c.n_pass > 1) {
        if (loop_arch) {
            c.core0 = (int) key("loop.start"), c.core1 = (int) key("loop.end"), c.rank = (int) key("loop.gate_rank");
            if (c.core1 > c.n_visit) die("loop.end past block_count");   // (block_count: the layers here)
            c.n_visit += (c.n_pass - 1) * (c.core1 - c.core0);
        } else {
            c.core0 = (int) key("loop.core_start"), c.core1 = (int) key("loop.core_end"), c.rank = (int) key("loop.gate_rank");
            if (c.core1 > c.n_visit) die("loop.core_end past block_count");
        }
        if (c.core0 <= 0 || c.core1 <= c.core0 || c.core0 % c.block || (c.core1 - c.core0) % c.block || c.rank <= 0)
            die("the looped core must start after the first block and span whole blocks");
    }
    const int n_core = c.core1 - c.core0;
    c.n_layer = c.n_visit - (c.n_pass - 1) * n_core;
    {   // the pattern is a visit's; a layer's is its first visit's (the same in every pass)
        const auto* sp = S.get(a + "attention.sliding_window_pattern");
        const bool per_layer = loop_arch && sp && (int) sp->items.size() == c.n_layer && c.n_layer != c.n_visit;
        if (!sp || ((int) sp->items.size() != c.n_visit && !per_layer)) die("sliding_window_pattern");
        for (int l = 0; l < c.n_layer; ++l)
            c.sliding.push_back(sp->items[per_layer || l < c.core1 ? l : l + (c.n_pass - 1) * n_core].u != 0);
    }
    // (a row of an expert is whole blocks of its type: checked per tensor below; 64 covers NVFP4 and Q8_0)
    if (c.D != 128 || c.H % c.KV || c.C % 64 || c.F % 64) die("unexpected shapes");
    const int n_sliding = (int) std::count(c.sliding.begin(), c.sliding.end(), true);
    std::printf("strata-ik2: %d layers (%d sliding), %d experts (%d a token), width %d, vocab %d; %zu shards\n",
                c.n_layer, n_sliding, c.E, c.K, c.C, c.n_vocab, S.files.size());
    if (c.n_pass > 1) {
        std::printf("looped: layers %d .. %d run %d times - %d layer visits a token; message gates of rank %d\n", c.core0,
                    c.core1 - 1, c.n_pass, c.n_visit, c.rank);
    }
    if (getenv("IK2_POP_OUT")) g_pop.assign((size_t) c.n_layer * c.E, 0), g_pop_E = c.E;
    g_tp_F = c.F, g_tp_C = c.C;
    if (n_gpu == 2 && getenv("IK2_EP") && std::atoi(getenv("IK2_EP"))) {
        const double share = getenv("IK2_EP_SPLIT") ? std::atof(getenv("IK2_EP_SPLIT")) : 0.5;
        g_pop_E_ep = c.E;
        g_ep_own.assign((size_t) c.n_layer * c.E, 0);
        std::vector<uint64_t> cnt;
        if (const char* wf = getenv("IK2_WARM"); wf && *wf) {
            FILE* in = std::fopen(wf, "rb");
            int32_t hdr[2] = {0, 0};
            if (in && std::fread(hdr, 4, 2, in) == 2 && hdr[0] == c.n_layer && hdr[1] == c.E) {
                cnt.resize((size_t) c.n_layer * c.E);
                if (std::fread(cnt.data(), 8, cnt.size(), in) != cnt.size()) cnt.clear();
            }
            if (in) std::fclose(in);
        }
        for (int l = 0; l < c.n_layer; ++l) {
            if (cnt.empty()) {
                for (int e = 0; e < c.E; ++e)
                    g_ep_own[(size_t) l * c.E + e] = ((uint32_t) (e * 2654435761u + l * 40503u) >> 8) % 10000 < share * 10000;
                continue;
            }
            std::vector<int> ord(c.E);
            for (int e = 0; e < c.E; ++e) ord[e] = e;
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return cnt[(size_t) l * c.E + a] > cnt[(size_t) l * c.E + b]; });
            double got[2] = {0, 0};
            const double want[2] = {1.0 - share, share};
            for (int e : ord) {   // (the GPU furthest below its share takes the next most picked)
                const double v = (double) cnt[(size_t) l * c.E + e] + 1.0;
                const int d = got[0] / want[0] <= got[1] / want[1] ? 0 : 1;
                g_ep_own[(size_t) l * c.E + e] = (uint8_t) d;
                got[d] += v;
            }
        }
    }

    {
        int nd = 0;
        CU(cudaGetDeviceCount(&nd));
        if (n_gpu < 1 || n_gpu > nd) die("--gpus " + std::to_string(n_gpu) + ": " + std::to_string(nd) + " visible");
    }
    g_dev.resize(n_gpu);
    for (int d = 0; d < n_gpu; ++d) {
        CU(cudaSetDevice(d));
        CU(cudaStreamCreate(&g_dev[d].s));
        CU(cudaStreamCreate(&g_dev[d].cs));
        CU(cudaEventCreateWithFlags(&g_dev[d].ev, cudaEventDisableTiming));
        CB(cublasCreate(&g_dev[d].blas));
        CB(cublasSetStream(g_dev[d].blas, g_dev[d].s));
        CB(cublasSetMathMode(g_dev[d].blas, CUBLAS_DEFAULT_MATH));
    }
    g_ep = n_gpu == 2 && getenv("IK2_EP") && std::atoi(getenv("IK2_EP"));
    g_tp = n_gpu == 2 && !g_ep && getenv("IK2_TP") && std::atoi(getenv("IK2_TP"));
    if (dual()) {
        int p01 = 0, p10 = 0;
        CU(cudaDeviceCanAccessPeer(&p01, 0, 1));
        CU(cudaDeviceCanAccessPeer(&p10, 1, 0));
        if (p01 && p10) {
            CU(cudaSetDevice(0));
            CU(cudaDeviceEnablePeerAccess(1, 0));
            CU(cudaSetDevice(1));
            CU(cudaDeviceEnablePeerAccess(0, 0));
        }
        CU(cudaSetDevice(0));
        CU(cudaEventCreateWithFlags(&g_ep_x, cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&g_ep_cp[0], cudaEventDisableTiming));
        CU(cudaSetDevice(1));
        CU(cudaEventCreateWithFlags(&g_ep_cp[1], cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&g_ep_done, cudaEventDisableTiming));
        CU(cudaSetDevice(0));
        CU(cudaEventCreateWithFlags(&g_tp_h[0], cudaEventDisableTiming));
        CU(cudaSetDevice(1));
        CU(cudaEventCreateWithFlags(&g_tp_h[1], cudaEventDisableTiming));
        std::printf("%s over 2 GPUs (peer access %s)\n", g_tp ? "experts tensor-parallel" : "expert parallel",
                    p01 && p10 ? "on" : "off: copies through the host");
        // Pinned mapped staging for inter-GPU zero-copy activation transfer
        auto alloc_mapped = [&](size_t bytes, void** h_ptr, void** d0_ptr, void** d1_ptr) {
            cudaError_t e = cudaHostAlloc(h_ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
            if (e != cudaSuccess) { (void) cudaGetLastError(); *h_ptr = nullptr; return false; }
            CU(cudaSetDevice(0));
            e = cudaHostGetDevicePointer(d0_ptr, *h_ptr, 0);
            if (e != cudaSuccess) { (void) cudaGetLastError(); return false; }
            CU(cudaSetDevice(1));
            e = cudaHostGetDevicePointer(d1_ptr, *h_ptr, 0);
            if (e != cudaSuccess) { (void) cudaGetLastError(); return false; }
            return true;
        };
        const int max_S = 64;
        const int max_T = 8;
        bool stage_ok = alloc_mapped((size_t) max_T * c.C * sizeof(float), &g_ep_stage.h_xq, (void**) &g_ep_stage.d_xq[0], (void**) &g_ep_stage.d_xq[1]) &&
                        alloc_mapped((size_t) max_S * 8, &g_ep_stage.h_route, (void**) &g_ep_stage.d_route[0], (void**) &g_ep_stage.d_route[1]) &&
                        alloc_mapped((size_t) max_S * c.C * sizeof(float), &g_ep_stage.h_y, (void**) &g_ep_stage.d_y[0], (void**) &g_ep_stage.d_y[1]);
        if (stage_ok) {
            std::printf("zero-copy mapped EP staging enabled (%zu KB)\n",
                        ((size_t) max_T * c.C * sizeof(float) + (size_t) max_S * 8 + (size_t) max_S * c.C * sizeof(float)) / 1024);
        } else {
            std::printf("zero-copy mapped EP staging disabled (fallback to standard transfer)\n");
        }
    }
    use_dev(0);
    g_overlap = !getenv("IK2_OVERLAP") || std::atoi(getenv("IK2_OVERLAP"));
    for (auto& e : g_ring) CU(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    // (off by default: on an RTX PRO 6000 node's Xeon 6737P an NVFP4 expert takes 0.73 ms on 7 threads against 0.33 ms
    //  over PCIe, and the lane measured 16.4 against 16.8 tok/s - IK2_CPU=1 to try it)
    // A card under 16 GB streams most of its experts over a slower link: the lane on by default (IK2_CPU=0: off), on
    // half the hardware threads, the link's expert taken as 1 ms (IK2_UP_MS) and no lazy copies (IK2_LAZY_MAX) - a
    // CPU-computed expert's copy queues ahead of the next layers' urgent ones on the one host-to-device engine
    // (RTX 2060 SUPER, PCIe 3.0 x8, Ryzen 9 3900X, Inkling-2-Mini IQ3_S: TG 18.9 -> 28.8 tok/s)
    bool small_card = false;
    size_t vram_total = 0;
    {
        size_t fr2 = 0;
        CU(cudaMemGetInfo(&fr2, &vram_total));
        small_card = n_gpu == 1 && vram_total < (16ull << 30);
    }
    // (the host copy mapped from the files cannot feed the lane: then the lane is not turned on by default)
#ifdef _WIN32
    const bool host_mmap_planned = getenv_str("IK2_HOST").empty() || getenv_str("IK2_HOST") != "pinned";
#else
    const bool host_mmap_planned = getenv_str("IK2_HOST") == "mmap";
#endif
    const bool cpu_want = getenv("IK2_CPU") ? std::atoi(getenv("IK2_CPU")) != 0 : small_card && !host_mmap_planned;
    if (small_card && cpu_want && !threads_given)   // (half the hardware threads, at least 2)
        threads = std::max(2, (int) std::thread::hardware_concurrency() / 2);
    if (cpu_want && ik2::cpu::available_any() && threads > 1 && n_gpu == 1)
        g_pool = new ik2::cpu::Pool(threads - 1);
    if (g_pool) ik2::cpu::spin_start(g_pool->size());   // (the batches' workers: the device plan's and a verify's)
    if (small_card && g_pool) g_lazy_max = 0;
    // a small card with a looped model: a long prompt takes lent layer-major passes (forward_prefill_sup), so forward()'s
    // chunk only sizes the short prompts' and the decode's buffers and the sliding layers' rings - 512 instead of 2048
    // gives the expert cache their VRAM (RTX 2060 SUPER, Inkling-2-Mini IQ3_S: 1192 -> 1365 slots, TG 36.2 -> 38.4)
    if (small_card && !chunk_given && c.n_pass > 1 &&
        !(getenv("IK2_LOOP_SUP") && !std::atoi(getenv("IK2_LOOP_SUP"))))
        chunk = 512;
    g_kv_grow = kv_grow_arg >= 0 ? kv_grow_arg != 0 : small_card;
    if (g_kv_grow && (n_gpu != 1 || dual())) {   // (the K / V live in GPU 0's pool: one GPU only)
        if (kv_grow_arg > 0) std::printf("--kv-grow: one GPU only; off\n");
        g_kv_grow = false;
    }
    g_kv_ctx = ctx;   // (and the link's expert time measured once the cache is up)
    if (getenv("IK2_UP_MS")) g_t_up = std::atof(getenv("IK2_UP_MS")) * 1e-3;
    g_cpu_lazy = !getenv("IK2_CPU_LAZY") || std::atoi(getenv("IK2_CPU_LAZY"));
    if (getenv("IK2_LAZY_MAX")) g_lazy_max = std::max(0, std::atoi(getenv("IK2_LAZY_MAX")));
    if (getenv("IK2_TOKGRAPH")) g_dp.graph = std::atoi(getenv("IK2_TOKGRAPH")) != 0;
    if (getenv("IK2_SUB")) g_sub_max = std::max(64, std::atoi(getenv("IK2_SUB")));
    if (n_gpu > 2 && getenv("IK2_DENSE")) die("IK2_DENSE with --gpus > 2");

    // Do the experts all fit in VRAM beside the dense weights and the K / V cache (Inkling-2-Mini's 14.7 GiB on a 24 GB
    // card)?  Then they all stay there: the dense weights are kept as the GGUF's blocks unless IK2_DENSE says otherwise
    // (half an FP16 copy's bytes, a token's products straight over them) and tokens take the resident path
    // (IK2_FAST=0: off; IK2_GRAPH=0: its launches one by one instead of a CUDA graph a token).
    g_graph = !(getenv("IK2_GRAPH") && !std::atoi(getenv("IK2_GRAPH")));
    g_attn_mma = c.D == 128 && c.H / c.KV <= 8 ? (getenv("IK2_ATTN_MMA") ? std::atoi(getenv("IK2_ATTN_MMA")) : 1) : 0;
    g_mma_split = k::attn_decode_mma_blocks(c.KV);
    // a small card streaming its experts: the dense weights as blocks too (an FP16 output head alone is ~0.9 GiB of a
    // cache that holds ~3 GiB: RTX 2060 SUPER 8 GB, Inkling-2-Mini IQ3_S, 654 -> 908 slots, TG 14.0 -> 20.7) - one GPU
    // (IK2_DENSE=q8 is refused past one: its FP16 scratch is one device's), and not with --draft (it reads the output
    // head in FP16)
    if (!getenv("IK2_DENSE") && small_card) g_dense_q8 = true;   // (with --draft too: its verify's products by mv_q8p_t)
    const bool can_resident = (n_gpu == 1 || (dual() && g_ep)) && !g_pool;
    if (can_resident) {
        size_t exp_b = 0, dense_b = 0, kv_b = 0;
        for (const auto& kv : S.by_name) {
            const auto& t = *kv.second.second;
            if (!k::block_values((int) t.type)) continue;
            if (t.name.find("_exps.weight") != std::string::npos) exp_b += tensor_bytes(t);
            else if (t.name != "token_embd.weight") dense_b += tensor_bytes(t);
        }
        for (int l = 0; l < c.n_layer; ++l) {
            const size_t cap = c.sliding[l] ? (size_t) std::min(ctx, c.window + std::max(chunk, ik2::draft::kBlock)) : (size_t) ctx;
            kv_b += (l >= c.core0 && l < c.core1 ? c.n_pass : 1) * cap * kv_tok_b((size_t) c.KV * c.D) * 2;
        }
        size_t fr0 = 0, tot0 = 0;
        CU(cudaMemGetInfo(&fr0, &tot0));
        const size_t reserve = (size_t) 3 << 29;   // (the work buffers and a quarter of a GiB of room: 1.5 GiB)
        const size_t exp_b_dev0 = dual() ? (exp_b / 2) : exp_b;
        const bool fits = exp_b_dev0 + dense_b + kv_b + reserve <= fr0 && exp_b_dev0 <= (size_t) (cache_gb * 1073741824.0);
        if (fits && !getenv("IK2_DENSE")) g_dense_q8 = true;
        g_fast_want = fits && g_dense_q8 && !(getenv("IK2_FAST") && !std::atoi(getenv("IK2_FAST")));
        if (g_fast_want && g_kv_grow) {   // (the resident tables point into every slot: none may become K / V)
            if (kv_grow_arg > 0) std::printf("--kv-grow: not with every expert resident; off\n");
            g_kv_grow = false;
        }
        std::printf("experts %.1f GiB (%s), dense weights %.1f GiB as blocks, K / V cache %.1f GiB, %.1f GiB free: %s\n",
                    exp_b / 1073741824.0, dual() ? "split across 2 GPUs" : "1 GPU",
                    dense_b / 1073741824.0, kv_b / 1073741824.0, fr0 / 1073741824.0,
                    fits ? "every expert stays in VRAM" : "the experts in host memory, a cache of them in VRAM");
    }

    // the dense weights
    const double t_dense = now();
    const size_t C = c.C, HD = (size_t) c.H * c.D, KD = (size_t) c.KV * c.D;
    M.L.resize(c.n_layer);
    // two layouts: llama.cpp PR #32187's (ref/inkling2.py's: the short convs folded, gate and up apart) and unsloth's
    // "GGUF_shard_rewrite" one that llama.cpp's master reads (names of its own, the convs' residual and scale outside
    // their weights, gate and up fused - gate's rows first - with one scale): the second's folded here as it loads
    const bool rw = S.info("blk.0.ffn_gate_up_exps.weight") != nullptr;
    auto nm = [&](const std::string& b, const char* old_name) -> std::string {
        static const std::map<std::string, std::string> m = {
            {"attn_r.weight", "attn_rel.weight"}, {"attn_r_clip.weight", "attn_rel_clip.weight"},
            {"attn_rel_proj.weight", "attn_rel_bias.weight"}, {"attn_res_score.weight", "attn_res_query.weight"},
            {"ffn_res_score.weight", "ffn_res_query.weight"}, {"output_res_score.weight", "output_res_query.weight"},
            {"attn_q_in_norm.weight", "attn_q_inp_norm.weight"}, {"attn_k_in_norm.weight", "attn_k_inp_norm.weight"},
            {"attn_v_in_norm.weight", "attn_v_inp_norm.weight"}, {"shortconv_k.weight", "attn_k_conv.weight"},
            {"shortconv_v.weight", "attn_v_conv.weight"}, {"shortconv_attn.weight", "attn_post_conv.weight"},
            {"shortconv_mlp.weight", "ffn_post_conv.weight"}, {"shortconv_k_dyn.weight", "attn_k_conv_proj.weight"},
            {"shortconv_v_dyn.weight", "attn_v_conv_proj.weight"}};
        auto it = m.find(old_name);
        return b + (rw && it != m.end() ? it->second : std::string(old_name));
    };
    // a conv's weights [C'][4] with the residual (tap 3: the current token) and the output scale folded in
    auto conv_w = [&](const std::string& b, const char* n, size_t Cc, bool dynamic) -> float* {
        auto w = host_f32(S, nm(b, n), Cc * 4);
        if (rw) {
            const float sc = 1.0f / std::sqrt((float) ((dynamic ? 2 : 1) * 4 + 1));
            for (size_t i = 0; i < Cc; ++i) {
                for (int j = 0; j < 4; ++j) w[i * 4 + j] *= sc;
                w[i * 4 + 3] += sc;
            }
        }
        return dev_f32(w);
    };
    auto dyn_host = [&](const std::string& b, const char* n) -> std::vector<float> {
        auto w = host_f32(S, nm(b, n), (size_t) c.KV * 4 * C);
        if (rw)
            for (auto& x : w) x *= 1.0f / std::sqrt(9.0f);
        return w;
    };
    std::printf("layout: %s\n", rw ? "unsloth's GGUF_shard_rewrite (llama.cpp master)" : "llama.cpp PR #32187");
    struct ExpJob {
        int l;
        std::string name;   // a tensor read whole into pinned memory
        std::vector<int> roles;
    };
    std::vector<ExpJob> exp_jobs;
    size_t host_total = 0;
    M.embd_norm = load_f32(S, "token_embd_norm.weight", C);   // (on GPU 0, where the embedding rows go)
    std::vector<float> lp_q, lp_kv;   // a looped core's message queries, [2 n_core][C] each
    std::vector<float> lp_all;
    if (c.n_pass > 1 && !loop_arch) {   // (inkling2_loop: blk.N.loop_msg_query, a layer's 4 in this order already)
        lp_q = host_f32(S, "loop_res_query.weight", (size_t) 2 * n_core * C);
        lp_kv = host_f32(S, "loop_kv_res_query.weight", (size_t) 2 * n_core * C);
    }
    for (int l = 0; l < c.n_layer; ++l) {
        Layer& Ly = M.L[l];
        const std::string b = "blk." + std::to_string(l) + ".";
        Ly.dev = dual() ? 0 : l * n_gpu / c.n_layer;
        use_dev(Ly.dev);
        Ly.sliding = c.sliding[l];
        auto sink_v = host_f32(S, b + "attn_sink_proj.weight", (size_t) c.H * C);
        if (Ly.sliding) {
            std::vector<MatPart> qkv_parts = {
                {b + "attn_q.weight", (int) HD, {}},
                {b + "attn_k.weight", (int) KD, {}},
                {b + "attn_v.weight", (int) KD, {}},
                {nm(b, "attn_r.weight"), c.H * c.d_rel, {}},
                {b + "attn_gate.weight", (int) HD, {}},
            };
            if (!g_dense_q8) {
                qkv_parts.push_back({"", c.H, std::move(sink_v)});
                Ly.wqkvrg = load_fused_f16(S, qkv_parts, (int) C);
            } else {
                if (!try_load_fused_raw(S, qkv_parts, (int) C, Ly.wqkvrg)) {
                    Ly.wq = load_mat(S, b + "attn_q.weight", (int) HD, (int) C);
                    Ly.wk = load_mat(S, b + "attn_k.weight", (int) KD, (int) C);
                    Ly.wv = load_mat(S, b + "attn_v.weight", (int) KD, (int) C);
                    Ly.wr = load_mat(S, nm(b, "attn_r.weight"), c.H * c.d_rel, (int) C);
                    Ly.wgate = load_mat(S, b + "attn_gate.weight", (int) HD, (int) C);
                }
                Ly.sink16 = dev_f16(sink_v);
            }
        } else {
            std::vector<MatPart> qrg_parts = {
                {b + "attn_q.weight", (int) HD, {}},
                {nm(b, "attn_r.weight"), c.H * c.d_rel, {}},
                {b + "attn_gate.weight", (int) HD, {}},
            };
            auto kd_v = dyn_host(b, "shortconv_k_dyn.weight");
            auto vd_v = dyn_host(b, "shortconv_v_dyn.weight");
            if (!g_dense_q8) {
                qrg_parts.push_back({"", c.H, std::move(sink_v)});
                Ly.wqrg = load_fused_f16(S, qrg_parts, (int) C);
                Ly.wk_dyn = load_fused_f16(S, {{b + "attn_k.weight", (int) KD, {}}, {"", c.KV * 4, std::move(kd_v)}},
                                           (int) C);
                Ly.wv_dyn = load_fused_f16(S, {{b + "attn_v.weight", (int) KD, {}}, {"", c.KV * 4, std::move(vd_v)}},
                                           (int) C);
            } else {
                if (!try_load_fused_raw(S, qrg_parts, (int) C, Ly.wqrg)) {
                    Ly.wq = load_mat(S, b + "attn_q.weight", (int) HD, (int) C);
                    Ly.wr = load_mat(S, nm(b, "attn_r.weight"), c.H * c.d_rel, (int) C);
                    Ly.wgate = load_mat(S, b + "attn_gate.weight", (int) HD, (int) C);
                }
                Ly.wk = load_mat(S, b + "attn_k.weight", (int) KD, (int) C);
                Ly.wv = load_mat(S, b + "attn_v.weight", (int) KD, (int) C);
                Ly.sink16 = dev_f16(sink_v);
                Ly.sc_k_dyn16 = dev_f16(kd_v);
                Ly.sc_v_dyn16 = dev_f16(vd_v);
            }
        }
        Ly.wo = load_mat(S, b + "attn_output.weight", (int) C, (int) HD);
        Ly.router16 = load_f16(S, b + "ffn_gate_inp.weight", (size_t) c.E * C);
        Ly.probs_b = load_f32(S, b + "exp_probs_b.bias", c.E);
        Ly.ffn_norm = load_f32(S, b + "ffn_norm.weight", C);
        if (Ly.sliding) {
            Ly.attn_norm = load_f32(S, b + "attn_norm.weight", C);
        } else {
            Ly.q_in_norm = load_f32(S, nm(b, "attn_q_in_norm.weight"), C);
            Ly.k_in_norm = load_f32(S, nm(b, "attn_k_in_norm.weight"), C);
            Ly.v_in_norm = load_f32(S, nm(b, "attn_v_in_norm.weight"), C);
        }
        Ly.q_clip = load_f32(S, b + "attn_q_clip.weight", HD);
        Ly.r_clip = load_f32(S, nm(b, "attn_r_clip.weight"), (size_t) c.H * c.d_rel);
        {   // [extent][d_rel]: PR #32187's attn_rel_proj is [d_rel][extent], the rewrite's attn_rel_bias already this
            const int ext = Ly.sliding ? c.extent_swa : c.extent;
            const auto v = host_f32(S, nm(b, "attn_rel_proj.weight"), (size_t) ext * c.d_rel);
            std::vector<float> t(v.size());
            if (rw) {
                t = v;
            } else {
                for (int r = 0; r < c.d_rel; ++r)
                    for (int e = 0; e < ext; ++e) t[(size_t) e * c.d_rel + r] = v[(size_t) r * ext + e];
            }
            Ly.rel_proj = dev_f32(t);
        }
        if (l > 0) {
            const std::string rn = nm(b, "attn_res_score.weight");
            const auto* ti = S.info(rn);
            if (!ti) die(rn);
            Ly.n_res_q = (int) (ti->elements() / C);
            Ly.res_score = load_f32(S, rn, (size_t) Ly.n_res_q * C);
        }
        Ly.ffn_res_score = load_f32(S, nm(b, "ffn_res_score.weight"), C);
        Ly.sc_k = conv_w(b, "shortconv_k.weight", KD, !Ly.sliding);
        Ly.sc_v = conv_w(b, "shortconv_v.weight", KD, !Ly.sliding);
        Ly.sc_attn = conv_w(b, "shortconv_attn.weight", C, false);
        Ly.sc_mlp = conv_w(b, "shortconv_mlp.weight", C, false);
        // the experts: (role tensor, its roles); a fused gate_up is gate's F rows then up's, an expert's 2 F rows
        std::vector<std::pair<std::string, std::vector<int>>> parts;
        if (rw) parts = {{"gate_up", {0, 1}}, {"down", {2}}};
        else parts = {{"gate", {0}}, {"up", {1}}, {"down", {2}}};
        for (const auto& pr : parts) {
            const std::string n = b + "ffn_" + pr.first + "_exps.weight";
            const auto* ti = S.info(n);
            if (!ti) die("no " + n);
            const int nr = (int) pr.second.size();
            if (ti->elements() != (uint64_t) c.E * c.F * C * nr) die(n + ": shape");
            if (!k::block_values((int) ti->type)) die(n + ": type " + ti->type_name() + " not decoded here");
            const int row_k = pr.first == "down" ? c.F : (int) C;   // gate / up rows are C long, down's F
            if (row_k % k::block_values((int) ti->type)) die(n + ": a row is not whole " + ti->type_name() + " blocks");
            const size_t per = tensor_bytes(*ti) / c.E;
            std::vector<float> sc;
            const std::string sn = b + "ffn_" + pr.first + "_exps.scale";
            if (S.info(sn)) sc = host_f32(S, sn, c.E);
            else if (ti->type == k::kNVFP4) die(n + ": NVFP4 without its .scale");
            for (int i = 0; i < nr; ++i) {
                const int r = pr.second[i];
                Ly.ty[r] = (int) ti->type;
                Ly.eb[r] = per / nr;
                Ly.stride[r] = per;
                Ly.scale[r] = sc;
            }
            host_total += tensor_bytes(*ti);
            exp_jobs.push_back({l, n, pr.second});
        }
        if (c.n_pass > 1 && l >= c.core0 && l < c.core1) {   // the core: its later passes' message gates and queries
            static const char* mod[4] = {"attn", "k", "v", "mlp"};
            static const char* mod_loop[4] = {"q", "k", "v", "ffn"};   // (inkling2_loop: loop_<m>_ctrl_*)
            for (int m = 0; m < 4; ++m) {
                const std::string p = loop_arch ? b + "loop_" + mod_loop[m] + "_ctrl_" : b + "loop_" + mod[m] + "_";
                Ly.lp_down[m] = dev_f16_of(host_f32(S, p + "down.weight", (size_t) c.rank * 2 * C));
                for (int fg = 0; fg < 2; ++fg) {   // (the file's [C][rank] by rank, [rank][C]: k::loop_gate's layout)
                    const auto w = host_f32(S, p + (fg ? "gain.weight" : "forget.weight"), C * c.rank);
                    std::vector<float> t(w.size());
                    for (size_t ch = 0; ch < C; ++ch)
                        for (int r = 0; r < c.rank; ++r) t[(size_t) r * C + ch] = w[ch * c.rank + r];
                    (fg ? Ly.lp_gain[m] : Ly.lp_forget[m]) = dev_f16_of(t);
                }
            }
            // loop_res_query: rows 2 i (q's) and 2 i + 1 (the MoE's) of core layer i; loop_kv_res_query: k's and v's
            const size_t i = (size_t) (l - c.core0);
            std::vector<float> q;
            if (loop_arch) {
                q = host_f32(S, b + "loop_msg_query.weight", 4 * C);   // rows q, k, v, the MoE's
            } else {
                q.resize(4 * C);
                std::memcpy(q.data(), lp_q.data() + 2 * i * C, C * 4);
                std::memcpy(q.data() + C, lp_kv.data() + 2 * i * C, C * 4);
                std::memcpy(q.data() + 2 * C, lp_kv.data() + (2 * i + 1) * C, C * 4);
                std::memcpy(q.data() + 3 * C, lp_q.data() + (2 * i + 1) * C, C * 4);
            }
            Ly.lp_query = dev_f32(q);
            lp_all.insert(lp_all.end(), q.begin(), q.end());
            if (l == c.core1 - 1) M.lp_query_all = dev_f32(lp_all);
        }
    }
    // the visits of a token and their state (FP8 E4M3 or FP16 KV; sliding layers ring-buffered in both CLI and --serve)
    M.V.resize(c.n_visit);
    for (int v = 0; v < c.n_visit; ++v) {
        Visit& Vi = M.V[v];
        if (c.n_pass == 1 || v < c.core0) Vi.l = v;
        else if (v < c.core0 + c.n_pass * n_core) Vi.l = c.core0 + (v - c.core0) % n_core, Vi.pass = (v - c.core0) / n_core;
        else Vi.l = v - (c.n_pass - 1) * n_core;
        const Layer& Ly = M.L[Vi.l];
        use_dev(Ly.dev);
        Vi.kv_cap = Ly.sliding ? std::min(ctx, c.window + std::max(chunk, ik2::draft::kBlock)) : ctx;
        if (g_kv_grow && !Ly.sliding) Vi.kv_cap = 0;   // (kv_ensure: in the expert pool, as the positions need)
        else if (kv_b8()) {
            Vi.kc8 = dalloc<uint8_t>((size_t) Vi.kv_cap * kv_tok_b(KD));
            Vi.vc8 = dalloc<uint8_t>((size_t) Vi.kv_cap * kv_tok_b(KD));
        } else {
            Vi.kc = dalloc<uint16_t>((size_t) Vi.kv_cap * KD);
            Vi.vc = dalloc<uint16_t>((size_t) Vi.kv_cap * KD);
        }
        Vi.h_k = dalloc<float>(3 * KD);
        Vi.h_v = dalloc<float>(3 * KD);
        Vi.h_attn = dalloc<float>(3 * C);
        Vi.h_mlp = dalloc<float>(3 * C);
    }
    c.n_rows = (int) (S.info("output.weight")->elements() / C);
    M.out_w = load_mat(S, "output.weight", c.n_rows, (int) C);
    if (g_dense_q8) {   // (a layer's largest; the head's logits go a token at a time over its blocks)
        g_w16_n = (2 * HD + 2 * KD + (size_t) c.H * c.d_rel) * C;
        g_w16 = dalloc<uint16_t>(g_w16_n);
    }
    M.out_norm = load_f32(S, "output_norm.weight", C);
    M.out_res_score = load_f32(S, rw ? "output_res_query.weight" : "output_res_score.weight", C);
    M.embd = S.data("token_embd.weight");
    M.embd_type = (int) S.info("token_embd.weight")->type;
    if (!k::block_values(M.embd_type)) die("token_embd's type");
    if (g_stage) cudaFree(g_stage), g_stage = nullptr, g_stage_bytes = 0;
    {
        size_t kv_bytes = 0;
        for (const auto& Vi : M.V) kv_bytes += (size_t) Vi.kv_cap * kv_tok_b(KD) * 2;
        std::printf("kv cache: %s (%.2f GiB across %d layer%s)\n",
                    g_kv_i8 ? "int8, an FP16 scale per 64" : g_kv_fp8 ? "FP8 E4M3" : "FP16",
                    kv_bytes / 1073741824.0, c.n_visit, c.n_pass > 1 ? " visits" : "s");
        std::fflush(stdout);
    }
    size_t fr = 0, tot = 0;
    for (int d = 0; d < n_gpu; ++d) {
        use_dev(d);
        CU(cudaMemGetInfo(&fr, &tot));
        std::printf("dense weights in VRAM: %.1f s; GPU %d: %.1f GiB of %.1f free\n", now() - t_dense, d,
                    fr / 1073741824.0, tot / 1073741824.0);
        std::fflush(stdout);
    }

    // the experts into pinned host memory: the allocations, then the reads, several files at once.  IK2_HOST=mmap:
    // read where the files are mapped instead (pageable: slower copies, no 323 GiB of host memory - for several
    // GPUs whose caches hold most of the experts)
    const double t_host = now();
    Pinned pin;
    double t_alloc = 0;
    // (the resident path keeps every expert in VRAM, so no pinned host copy is needed: the experts go from where the
    //  files are mapped straight to their slots, and stay in the page cache for the next load - IK2_HOST=pinned: as
    //  for the other models)
    if (g_pool) {   // (no layer's experts the CPU can multiply - say NVFP4 on an AVX2-only CPU: no idle lane threads)
        bool any = false;
        for (int l = 0; !any && l < c.n_layer; ++l)
            any = ik2::cpu::available_for(M.L[l].ty[0]) && ik2::cpu::available_for(M.L[l].ty[1]) &&
                  ik2::cpu::available_for(M.L[l].ty[2]);
        if (!any) {
            std::printf("CPU lane: no layer's expert types on this CPU; off\n");
            delete g_pool;
            g_pool = nullptr;
            if (!getenv("IK2_LAZY_MAX")) g_lazy_max = 1 << 30;
        }
    }
#ifdef _WIN32
    // Windows restricts physical memory pinning (cudaHostAlloc fails with OOM for multi-gigabyte models).
    // Default to direct mmap reading unless IK2_HOST=pinned is explicitly requested.
    const bool host_mmap = !getenv_str("IK2_HOST").empty() ? getenv_str("IK2_HOST") != "pinned" : true;
#else
    const bool host_mmap = !getenv_str("IK2_HOST").empty() ? getenv_str("IK2_HOST") == "mmap" : g_fast_want;
#endif
    if (host_mmap) {
        if (g_pool) die("IK2_HOST=mmap with the CPU lane");
        for (const auto& J : exp_jobs) {
            Layer& Ly = M.L[J.l];
            const uint8_t* p = S.data(J.name);
            for (size_t i = 0; i < J.roles.size(); ++i) Ly.host[J.roles[i]] = p + i * Ly.eb[J.roles[i]];
            {   // (read ahead: a cold file otherwise comes in a page fault at a time)
                const size_t pg = 4096, a0 = (size_t) p & ~(pg - 1);
                ::madvise((void*) a0, (size_t) p - a0 + tensor_bytes(*S.info(J.name)), MADV_WILLNEED);
            }
        }
        std::printf("experts: read where the files are mapped (IK2_HOST=mmap)\n");
        std::fflush(stdout);
    } else {
        // Pinned host memory: anonymous mappings read into and then registered with CUDA, a layer as soon as its
        // reads are done (323 GiB in ~60 s from VAST); IK2_ALLOC=hostalloc: cudaHostAlloc instead, which pins at
        // ~3 GiB/s (the driver serializes it; 110-190 s)
#ifndef _WIN32
        bool use_mmap_reg = !(getenv("IK2_ALLOC") && std::string(getenv("IK2_ALLOC")) == "hostalloc");
        if (use_mmap_reg) {
            const size_t probe_bytes = 2ULL << 20;
            void* mp = ::mmap(nullptr, probe_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mp == MAP_FAILED) {
                use_mmap_reg = false;
            } else {
                *((volatile uint8_t*) mp) = 0;
                if (cudaHostRegister(mp, probe_bytes, cudaHostRegisterPortable) != cudaSuccess) {
                    cudaGetLastError();
                    use_mmap_reg = false;
                } else {
                    CU(cudaHostUnregister(mp));
                }
                ::munmap(mp, probe_bytes);
            }
        }
        pin.blocks.resize(c.n_layer);
        if (use_mmap_reg) {
            for (int l = 0; l < c.n_layer; ++l) {
                const Layer& Ly = M.L[l];
                const size_t slot_bytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2];
                const size_t layer_bytes = (size_t) c.E * slot_bytes;
                void* p = ::mmap(nullptr, layer_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) {
                    for (int k = 0; k < l; ++k) ::munmap(pin.blocks[k].ptr, pin.blocks[k].bytes);
                    pin.blocks.assign(c.n_layer, {});
                    use_mmap_reg = false;
                    break;
                }
                pin.blocks[l] = {p, layer_bytes, true, false};
            }
        }
#else
        bool use_mmap_reg = false;
        pin.blocks.resize(c.n_layer);
#endif
        // (pinned: the allocations and the reads overlap - a layer's reads start as soon as its buffer exists;
        // IK2_ALLOC_THREADS threads allocate, default 1)
        std::vector<std::atomic<int>> ready(c.n_layer);
        for (auto& r : ready) r.store(use_mmap_reg ? 1 : 0);
        for (int l = 0; l < c.n_layer; ++l) {
            Layer& Ly = M.L[l];
            const size_t slot_bytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2];
            Ly.host_contig = true;
            Ly.slot_bytes = slot_bytes;
            Ly.stride[0] = Ly.stride[1] = Ly.stride[2] = slot_bytes;
        }
        std::vector<std::thread> alloc_th;
        std::atomic<int> next_alloc{0};
        double t_alloc_end = 0;
        std::atomic<int> alloc_left{use_mmap_reg ? 0 : c.n_layer};
        if (!use_mmap_reg) {
            const int na = std::max(1, getenv("IK2_ALLOC_THREADS") ? std::atoi(getenv("IK2_ALLOC_THREADS")) : 1);
            for (int t = 0; t < na; ++t)
                alloc_th.emplace_back([&]() {
                    for (;;) {
                        const int l = next_alloc++;
                        if (l >= c.n_layer) return;
                        const size_t layer_bytes = (size_t) c.E * M.L[l].slot_bytes;
                        void* p = nullptr;
                        CU(cudaHostAlloc(&p, layer_bytes, cudaHostAllocPortable));
                        pin.blocks[l] = {p, layer_bytes, false, false};
                        ready[l].store(1, std::memory_order_release);
                        if (--alloc_left == 0) t_alloc_end = now();
                    }
                });
        }
        auto wait_layer = [&](int l) {
            while (!ready[l].load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            Layer& Ly = M.L[l];
            uint8_t* buf = (uint8_t*) pin.blocks[l].ptr;
            Ly.host[0] = buf;
            Ly.host[1] = buf + Ly.eb[0];
            Ly.host[2] = buf + Ly.eb[0] + Ly.eb[1];
        };
        std::vector<std::atomic<int>> jobs_left(c.n_layer);
        for (auto& v : jobs_left) v.store(0);
        for (const auto& J : exp_jobs) ++jobs_left[J.l];
        std::mutex reg_m;
        std::condition_variable reg_cv;
        std::deque<int> reg_q;
        int reg_fail = 0;
        std::thread reg_th;
        if (use_mmap_reg)
            reg_th = std::thread([&]() {   // registers each layer once its reads are done, while the others are read
                for (int done = 0; done < c.n_layer; ++done) {
                    int l;
                    {
                        std::unique_lock<std::mutex> lk(reg_m);
                        reg_cv.wait(lk, [&] { return !reg_q.empty(); });
                        l = reg_q.front();
                        reg_q.pop_front();
                    }
                    if (cudaHostRegister(pin.blocks[l].ptr, pin.blocks[l].bytes, cudaHostRegisterPortable) == cudaSuccess)
                        pin.blocks[l].registered = true;
                    else
                        cudaGetLastError(), ++reg_fail;
                }
            });
        auto layer_read = [&](int l) {
            if (!use_mmap_reg) return;
            {
                std::lock_guard<std::mutex> lk(reg_m);
                reg_q.push_back(l);
            }
            reg_cv.notify_one();
        };
        for (int l = 0; l < c.n_layer; ++l)
            if (jobs_left[l].load() == 0) layer_read(l);
        std::vector<std::thread> th;
        std::atomic<int> next{0};
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&]() {
                for (;;) {
                    const int j = next++;
                    if (j >= (int) exp_jobs.size()) return;
                    const auto& J = exp_jobs[j];
                    wait_layer(J.l);
                    const Layer& Ly = M.L[J.l];
                    const size_t role_off = J.roles[0] == 0 ? 0 : (J.roles[0] == 1 ? Ly.eb[0] : Ly.eb[0] + Ly.eb[1]);
                    const size_t exp_bytes = tensor_bytes(*S.info(J.name)) / c.E;
                    uint8_t* dst = (uint8_t*) pin.blocks[J.l].ptr + role_off;
                    read_into_strided(S.paths[S.file_of(J.name)], S.file_offset(J.name), c.E, exp_bytes,
                                      Ly.slot_bytes, dst);
                    if (--jobs_left[J.l] == 0) layer_read(J.l);
                }
            });
        for (auto& t : alloc_th) t.join();
        for (auto& t : th) t.join();
        for (int l = 0; l < c.n_layer; ++l) wait_layer(l);   // (host[] of layers no job read)
        t_alloc = (use_mmap_reg ? 0.0 : t_alloc_end - t_host);
        const double t_read = now() - t_host - t_alloc;   // (the reads left after the last allocation)
        double t_pin = 0;
        if (use_mmap_reg) {
            const double tp0 = now();
            reg_th.join();   // (the layers read last are still registering)
            t_pin = now() - tp0;
            if (reg_fail > 0)
                std::fprintf(stderr, "strata-ik2: warning: cudaHostRegister failed on %d layers (running pageable)\n",
                             reg_fail);
        }
        if (use_mmap_reg) {
            std::printf("experts in pinned (mmap+register) memory: %.1f GiB; allocating %.1f s, reading %.1f s "
                        "(%.2f GiB/s), pinning %.1f s\n",
                        host_total / 1073741824.0, t_alloc, t_read, host_total / 1073741824.0 / t_read, t_pin);
        } else {
            std::printf("experts in pinned memory: %.1f GiB; allocating %.1f s, reading %.1f s (%.2f GiB/s)\n",
                        host_total / 1073741824.0, t_alloc, t_read, host_total / 1073741824.0 / t_read);
        }
        std::fflush(stdout);
    }

    // the drafter (before the expert cache takes what VRAM is left)
    ik2::draft::Drafter* drafter = nullptr;
    if (!draft_path.empty()) {
        if (n_gpu > 2 || (n_gpu == 2 && !g_ep)) die("--draft: single GPU or dual-GPU EP only so far");
        use_dev(0);
        bool stage_here = false;   // (the head's staging and FP16 copy only for the fold: freed after it on dense Q8)
        if (!M.out_w.f16) {
            const auto* t = S.info("output.weight");
            const size_t nb = tensor_bytes(*t);
            if (nb > g_stage_bytes) {
                if (g_stage) cudaFree(g_stage);
                g_stage = dalloc<uint8_t>(g_stage_bytes = nb);
                stage_here = true;
            }
            CU(cudaMemcpy(g_stage, S.data("output.weight"), nb, cudaMemcpyHostToDevice));
            M.out_w.f16 = dalloc<uint16_t>((size_t) c.n_rows * c.C);
            k::decode_f16(M.out_w.type, g_stage, (size_t) c.n_rows * c.C, M.out_w.f16, g_s);
            CU(cudaStreamSynchronize(g_s));
        }
        const double t0 = now();
        try {
            drafter = ik2::draft::load(draft_path, c.C, ctx, g_blas, g_s);
            ik2::draft::fold(drafter, M.out_w.f16, c.n_rows);
        } catch (const std::exception& e) {
            die(e.what());
        }
        if (M.out_w.q8) {   // (the head stays planar Q8: the drafter's base rows by mv_q8p_t; 0.9 GiB to the cache)
            CU(cudaDeviceSynchronize());
            cudaFree(M.out_w.f16);
            M.out_w.f16 = nullptr;
            if (stage_here) cudaFree(g_stage), g_stage = nullptr, g_stage_bytes = 0;
        }
        g_taps_n = ik2::draft::taps(drafter);
        g_taps_ld = std::max(chunk, ik2::draft::kBlock);
        g_taps = dalloc<float>((size_t) g_taps_n * g_taps_ld * c.C);
        for (auto& Vi : M.V) {
            const size_t cc[4] = {KD, KD, C, C};
            for (int i = 0; i < 4; ++i) {
                Vi.rec_old[i] = dalloc<float>(3 * cc[i]);
                Vi.rec_x[i] = dalloc<float>((size_t) ik2::draft::kBlock * cc[i]);
            }
        }
        std::printf("drafter: %s, %.2f GiB of VRAM, %.1f s\n", draft_path.c_str(),
                    ik2::draft::bytes(drafter) / 1073741824.0, now() - t0);
    }

    // IK2_DEVPLAN (see DevPlan): one GPU, the CPU lane, the experts' types on it, no per-expert scales
    {
        bool ok = !(getenv("IK2_DEVPLAN") && !std::atoi(getenv("IK2_DEVPLAN"))) && g_pool && n_gpu == 1 && !dual() &&
                  !g_tp && !g_fast_want && c.K <= 16 && c.E <= 256;
        // (its mirror is one pool's: every layer's experts the same slot size, as ExpertCache::init groups them)
        for (int l = 1; ok && l < c.n_layer; ++l)
            ok = (M.L[l].eb[0] + M.L[l].eb[1] + M.L[l].eb[2] + 255) / 256 == (M.L[0].eb[0] + M.L[0].eb[1] + M.L[0].eb[2] + 255) / 256;
        for (int l = 0; ok && l < c.n_layer; ++l) {
            const Layer& Ly = M.L[l];
            for (int r = 0; r < 3; ++r) ok = ok && ik2::cpu::available_for(Ly.ty[r]) && Ly.scale[r].empty();
        }
        g_dp_multi_types = ok;   // (a verify window's misses as jobs of several tokens: the int8 dots' types only)
        for (int l = 0; g_dp_multi_types && l < c.n_layer; ++l)
            g_dp_multi_types = ik2::cpu::batch_multi_ok(M.L[l].ty[0], M.L[l].ty[1], M.L[l].ty[2], c.C, c.F);
        if (ok) {
            use_dev(0);
            dp_alloc(c);
            g_dp.ok = true;
            std::printf("devplan: decode tokens enqueued without waiting for the routing (IK2_DEVPLAN=0: off)\n");
        }
    }
    // the work buffers and the expert cache, a set a GPU
    for (int d = 0; d < n_gpu; ++d) {
    use_dev(d);
    CU(cudaMemGetInfo(&fr, &tot));
    Work& W = g_dev[d].W;
    W.Tm = std::max({chunk, 1, ik2::draft::kBlock});
    W.ctx = ctx;
    W.nk_max = std::min(ctx, ctx);
    // Tokens a layer-major prefill pass (every expert not in VRAM streams in once a pass): 16384 when there is room
    // (a pass's scratch is ~280 KB a token: 4.6 GB, taken from the expert cache), else 4096.
    // (a looped model's prompts: below - layer-major passes on one GPU, while the cache lends their scratch)
    // ... not the resident path's models: a pass borrows cache slots, which would evict experts
    W.T_sup = ((n_gpu == 1 || dual()) && draft_path.empty() && c.n_pass == 1 && !g_fast_want && fr >= (20ULL << 30)) ? (fr >= (60ULL << 30) ? 16384 : 4096) : 0;
    // a looped model on one GPU: layer-major passes too, as long as the expert cache lends them (IK2_LOOP_SUP=0: off)
    if (!W.T_sup && c.n_pass > 1 && n_gpu == 1 && draft_path.empty() && !g_fast_want &&
        !(getenv("IK2_LOOP_SUP") && !std::atoi(getenv("IK2_LOOP_SUP"))))
        W.T_sup = std::max(chunk, 1);
    // IK2_SUP=N: tokens a layer-major prefill pass (each pass streams every expert not in the cache once)
    if (W.T_sup > 0 && getenv("IK2_SUP")) W.T_sup = std::max(W.Tm, std::atoi(getenv("IK2_SUP")));
    const size_t Tm = W.Tm, S8 = Tm * c.K;
    const size_t T_buf = std::max(Tm, (size_t) W.T_sup), S_buf = T_buf * c.K;
    W.blk_ld = T_buf;
    for (float** p : {&W.h, &W.r, &W.tmp}) *p = dalloc<float>(T_buf * C);
    for (float** p : {&W.res, &W.a, &W.xq, &W.xk, &W.xv}) *p = dalloc<float>(Tm * C);
    W.q = dalloc<float>(Tm * HD), W.gate = dalloc<float>(Tm * HD), W.o = dalloc<float>(Tm * HD);
    W.kk = dalloc<float>(Tm * KD), W.vv = dalloc<float>(Tm * KD), W.kconv = dalloc<float>(Tm * KD);
    W.vconv = dalloc<float>(Tm * KD);
    W.rr = dalloc<float>(Tm * c.H * c.d_rel), W.sink = dalloc<float>(Tm * c.H);
    W.dyn = dalloc<float>(2 * Tm * c.KV * 4);   // (a full layer's dynamic taps: k's, and on the resident path v's beside)
    // (at least 256 MiB, which the prompt attention's non-flash path batches its queries into; with the flash path's
    //  prompts a small card keeps only what a decode token's attention needs, plus 16 MiB - IK2_LG_MIN_MB)
    const size_t lg_floor = (size_t) std::max(1, getenv("IK2_LG_MIN_MB") ? std::atoi(getenv("IK2_LG_MIN_MB"))
                                                                        : (small_card && g_flash && !g_kv_fp8 ? 16 : 256)) << 18;
    W.lg_cap = std::max((size_t) 8 * c.H * (size_t) ctx + (size_t) 8 * c.H * 128 * 130, lg_floor);
    W.lg = dalloc<float>(W.lg_cap);
    W.blocks = dalloc<float>((size_t) (c.n_layer / c.block + 1) * T_buf * C);
    W.x16 = dalloc<uint16_t>(Tm * std::max(C, HD)), W.xk16 = dalloc<uint16_t>(Tm * C), W.xv16 = dalloc<uint16_t>(Tm * C);
    if (W.T_sup > 0 || dual()) W.xq16_sup = dalloc<uint16_t>(T_buf * C);
    if (!getenv("IK2_RELB") || std::atoi(getenv("IK2_RELB"))) {
        W.relb_ext = std::max(c.extent, c.extent_swa);
        W.relb = dalloc<float>((size_t) Work::kRelbRows * c.H * W.relb_ext);
    }
    if (kv_b8()) {
        W.kv_tmp_cap = std::min(ctx, 32768);
        W.kv_k16 = dalloc<uint16_t>((size_t) W.kv_tmp_cap * KD);
        W.kv_v16 = dalloc<uint16_t>((size_t) W.kv_tmp_cap * KD);
    }
    W.pos = dalloc<int>(T_buf);
    W.rlog = dalloc<float>(T_buf * c.E), W.route_d = dalloc<int>(S_buf * 2);
    W.ex16 = dalloc<uint16_t>(T_buf * C), W.w16 = dalloc<uint16_t>((size_t) 2 * c.F * C);   // (one expert's tokens)
    W.h16 = dalloc<uint16_t>(std::max<size_t>(64, T_buf) * c.F);
    W.gu32_bytes = 4 * std::max({S8 * 2 * (size_t) c.F,
                  Tm * std::max<size_t>(2 * HD + 2 * KD + (size_t) c.H * c.d_rel + c.H,
                                        (2 * HD + (size_t) c.H * c.d_rel + c.H) + 2 * (KD + (size_t) c.KV * 4)),
                  T_buf * 2 * (size_t) c.F, (W.lg_cap + 1) / 2});
    W.gu32 = dalloc<float>(W.gu32_bytes / 4);
    W.qkv = W.gu32;
    W.lg16 = (uint16_t*) W.gu32;
    W.y32 = dalloc<float>(std::max<size_t>(64, T_buf) * C);
    if (g_tp) W.tp_h = dalloc<uint16_t>(std::max<size_t>(64, S_buf) * c.F);   // (one expert's tokens; the token path's <= 64 rows)
    W.wtr = dalloc<float>(S_buf);
    W.sg = dalloc<float>(S_buf), W.su = dalloc<float>(S_buf), W.wt = dalloc<float>(S_buf);
    W.gidx = dalloc<int>(S_buf), W.inv = dalloc<int>(S_buf);
    W.blob = dalloc<const uint8_t*>(S8), W.tok = dalloc<int>(S8);
    for (int i = 0; i < Work::kPackRing; ++i) {
        CU(cudaHostAlloc(&W.pack_h[i], 8192, cudaHostAllocPortable));
        W.pack_d[i] = dalloc<uint8_t>(8192);
        if (g_pool) CU(cudaHostAlloc(&W.y_h[i], (size_t) 64 * C * 4, cudaHostAllocPortable));
    }
    W.T_host = (int) std::max<size_t>(T_buf, W.T_sup > 0 ? (size_t) ctx : 0);   // (a lent scratch can take up to ctx)
    const size_t S_host = (size_t) W.T_host * c.K;
    for (int i = 0; i < 2; ++i) {
        CU(cudaHostAlloc(&W.gidx_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.inv_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.sg_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.su_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.wt_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.wtr_h[i], S_host * 4, cudaHostAllocPortable));
    }
    CU(cudaHostAlloc(&W.x_h, (size_t) 8 * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.emb_h, Tm * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.pos_h, (size_t) W.T_host * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.route_h, S_host * 8, cudaHostAllocPortable));
    W.argmax_d = dalloc<int>(Tm);
    CU(cudaHostAlloc(&W.argmax_h, Tm * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.logits_h, (size_t) c.n_rows * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.cand_h, (size_t) (2 + 2 * kCandCap) * 4, cudaHostAllocPortable));
    {   // (IK2_RG_SLOTS: the slots - tokens x picks - up to which a prompt's experts go by groups; 0: never)
        // (96 tokens by default: past ~128 the FP16 GEMMs are the faster way)
        const int want = getenv("IK2_RG_SLOTS") ? std::atoi(getenv("IK2_RG_SLOTS")) : 96 * c.K;
        W.rg_slots = std::max(0, std::min(want, (int) std::max<size_t>(64, T_buf)));
        W.rg_groups = c.E + W.rg_slots / k::kGroupMax;
        W.rg_bytes = (size_t) W.rg_groups * (8 + 4 + 8 * k::kGroupMax) + (size_t) W.rg_slots * 16;
        CU(cudaHostAlloc(&W.rg_h, W.rg_bytes, cudaHostAllocPortable));
        W.rg_d = dalloc<uint8_t>(W.rg_bytes);
    }
    W.cand_d = dalloc<int>(2 + 2 * kCandCap);
    W.cand_part = dalloc<int>(k::cand_scratch(c.n_vocab));
    const int n_ev = (c.E + Work::kMissBatch - 1) / Work::kMissBatch;
    W.miss_ev.resize(n_ev);
    for (int i = 0; i < n_ev; ++i) CU(cudaEventCreateWithFlags(&W.miss_ev[i], cudaEventDisableTiming));
    for (int r = 0; r < Work::kPfRing; ++r) {
        W.pf_evs[r].resize(n_ev);
        for (int i = 0; i < n_ev; ++i) CU(cudaEventCreateWithFlags(&W.pf_evs[r][i], cudaEventDisableTiming));
    }
    W.cnt.assign(c.E + 1, 0);
    W.start.assign(c.E + 1, 0);
    W.fill.assign(c.E + 1, 0);
    W.hit_exps.reserve(c.E);
    W.miss_exps.reserve(c.E);
    W.slot_ptrs.assign(c.E, nullptr);
    W.seen_stamp.assign(c.E, 0);
    W.gi_stamp.assign(c.E, 0);
    W.seen_late.assign(c.E, 0);
    W.gi_idx.assign(c.E, 0);
    W.logits = dalloc<float>((size_t) 16 * c.n_rows);
    if (c.n_pass > 1) {
        W.lp_pre = dalloc<float>(Tm * C);
        W.lp_bank = dalloc<float>((size_t) (n_core / c.block) * Tm * C);   // (the pass's blocks after its first, its final stream)
        W.lp_msg = dalloc<float>(4 * Tm * C);
        W.lp_msg_all = dalloc<float>((size_t) (c.core1 - c.core0) * 4 * C);
        W.lp_in = dalloc<float>(3 * Tm * C);
        W.lp_g = dalloc<float>(3 * Tm * (2 * (size_t) c.rank + 2));
        W.lp_a = dalloc<float>(3 * Tm * (size_t) c.rank);
        W.lp_f = dalloc<float>(2 * Tm * C);
    }
    for (auto& Vi : M.V) {   // the conv histories start at zero
        if (M.L[Vi.l].dev != d) continue;
        CU(cudaMemset(Vi.h_k, 0, 3 * KD * 4));
        CU(cudaMemset(Vi.h_v, 0, 3 * KD * 4));
        CU(cudaMemset(Vi.h_attn, 0, 3 * C * 4));
        CU(cudaMemset(Vi.h_mlp, 0, 3 * C * 4));
    }
    ExpertCache& EC = g_dev[d].EC;
    CU(cudaMemGetInfo(&fr, &tot));
    // (what is left free: 2 GiB beside a cache that streams experts in and out; a quarter of one when every expert
    //  stays in VRAM - nothing more is allocated but a token's graph)
    //  --vram-reserve-mib given: that much instead (a small card: 2 GiB of 8 is a third of the cache)
    const size_t room = vram_reserve_given ? std::max((size_t) (std::max(0.0, vram_reserve_mib) * 1048576.0),
                                                      g_fast_want ? (size_t) 256 << 20 : (size_t) 0)
                                           : (g_fast_want ? (size_t) 256 << 20 : (size_t) 2 << 30);
    const size_t budget = std::min((size_t) (cache_gb * 1073741824.0), fr > room ? fr - room : (size_t) 0);
    EC.tp = g_tp ? d : -1;
    EC.init(M.L, dual() ? -1 : d, c.n_layer, c.E, budget);
    std::printf("expert cache, GPU %d: %d slots across %zu pools (%.1f GiB); load %.1f s in all\n", d, EC.total_slots(),
                EC.pools.size(), EC.total_bytes() / 1073741824.0, now() - t_start);
    for (size_t p = 0; p < EC.pools.size(); ++p)
        std::printf("  pool %zu: %d slots of %.1f MiB (%.1f GiB, %d layers)\n", p, EC.pools[p].n_slots,
                    EC.pools[p].slot_bytes / 1048576.0,
                    (double) EC.pools[p].n_slots * EC.pools[p].slot_bytes / 1073741824.0, EC.pools[p].n_layers);
    if (g_kv_grow && !EC.pools.empty()) {   // (what --kv-grow can hold here, said at load rather than mid-generation)
        size_t full_v = 0;
        for (const auto& Vi : M.V) full_v += !M.L[Vi.l].sliding;
        int pf = 0;   // (the pool kv_ensure takes them from: the first full layer's)
        for (const auto& Vi : M.V)
            if (!M.L[Vi.l].sliding) { pf = std::max(0, EC.layer_pool[Vi.l]); break; }
        const auto& P0 = EC.pools[pf];
        const size_t per_tok = 2 * full_v * kv_tok_b((size_t) c.KV * c.D);
        const long most = per_tok ? (long) ((size_t) std::max(0, P0.n_slots - 2 * c.E - 80) * P0.slot_bytes / per_tok) : 0;
        if (full_v && most < ctx)
            std::printf("kv-grow: this card holds the full layers' K / V for about %ld tokens, under --ctx %d\n", most, ctx);
    }
    // the link's time an expert for the CPU lane's choice: one slot copied in from pinned host memory, timed (the
    // estimate 0.36 ms is PCIe 5's; an RTX 2060 SUPER on PCIe 3.0 x8 takes ~0.55 ms a 3.6 MiB expert)
    if (small_card && g_pool && !getenv("IK2_UP_MS") && !EC.pools.empty() && EC.pools[0].n_slots > 0 && M.L[0].host[0]) {
        const auto& P0 = EC.pools[0];
        float best = 1e9f;
        cudaEvent_t e0, e1;
        CU(cudaEventCreate(&e0));
        CU(cudaEventCreate(&e1));
        for (int i = 0; i < 4; ++i) {
            CU(cudaEventRecord(e0, g_s));
            CU(cudaMemcpyAsync(P0.data, M.L[0].host[0], P0.slot_bytes, cudaMemcpyHostToDevice, g_s));
            CU(cudaEventRecord(e1, g_s));
            CU(cudaEventSynchronize(e1));
            float ms = 0;
            CU(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
        }
        CU(cudaEventDestroy(e0));
        CU(cudaEventDestroy(e1));
        g_t_up = 1.1 * best * 1e-3;   // (+10%: the copies queue behind each other in a layer)
        std::printf("CPU lane: the link takes %.2f ms an expert (%.1f GB/s)\n", g_t_up * 1e3, P0.slot_bytes / (best * 1e6));
    }
    const char* wf = getenv("IK2_WARM");
    if ((wf && *wf) || g_ep) {   // the cache filled with the most picked experts (IK2_POP_OUT's file or all owned for EP)
        FILE* in = (wf && *wf) ? std::fopen(wf, "rb") : nullptr;
        int32_t hdr[2] = {0, 0};
        std::vector<uint64_t> cnt;
        if (in && std::fread(hdr, 4, 2, in) == 2 && hdr[0] == c.n_layer && hdr[1] == c.E) {
            cnt.resize((size_t) c.n_layer * c.E);
            if (std::fread(cnt.data(), 8, cnt.size(), in) != cnt.size()) cnt.clear();
        }
        if (in) std::fclose(in);
        if (cnt.empty() && g_ep) cnt.assign((size_t) c.n_layer * c.E, 1);
        if (cnt.empty()) {
            std::printf("warm: the cache starts empty\n");
        } else {
            const double tw = now();
            const uint64_t m0 = EC.misses, b0 = EC.bytes_up;
            int n = 0;
            for (size_t p = 0; p < EC.pools.size(); ++p) {
                std::vector<int> idx;
                for (int l = 0; l < c.n_layer; ++l)
                    if (EC.layer_pool[l] == (int) p)
                        for (int e = 0; e < c.E; ++e)
                            if (!g_ep || ep_owner(l, e) == d) idx.push_back(l * c.E + e);
                // the most picked first; the chosen ones then go in least picked first, so that the evictions that
                // start from the oldest take the least picked
                std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
                const int k = std::min((int) idx.size(), EC.pools[p].n_active);
                for (int i = k - 1; i >= 0; --i) {
                    const int l = idx[i] / c.E, e = idx[i] % c.E;
                    EC.get(M.L[l], l, e, EC.tick, g_cs, nullptr, true, true);
                    ++n;
                }
            }
            CU(cudaStreamSynchronize(g_cs));
            ++EC.tick;   // (taken in the tick the first prefetch uses, they would all count as in use: none evictable)
            // the profile also seeds the prefetch's picture of which experts a pass uses (W.use, as 100k picks a
            // layer): without it the first long-enough request prefetches all 18k experts
            W.use.assign((size_t) c.n_layer * c.E, 0), W.use_tot.assign(c.n_layer, 0);
            for (int l = 0; l < c.n_layer; ++l) {
                long double sum = 0;
                for (int e = 0; e < c.E; ++e) sum += cnt[(size_t) l * c.E + e];
                if (sum <= 0) continue;
                for (int e = 0; e < c.E; ++e) {
                    W.use[(size_t) l * c.E + e] = (uint64_t) (100000.0L * cnt[(size_t) l * c.E + e] / sum + 0.5L);
                    W.use_tot[l] += W.use[(size_t) l * c.E + e];
                }
            }
            std::printf("warm: %d experts (%.1f GiB) copied in %.1f s\n", n, (EC.bytes_up - b0) / 1073741824.0,
                        now() - tw);
            EC.misses = m0, EC.bytes_up = b0;
        }
    }
    if (g_fast_want && d == 0) {   // the resident path: every expert in now, and where each is on the device
        auto blocks = [](const Mat& m) { return m.raw || m.q8; };   // (kept as the GGUF's blocks, whole or planar)
        bool all = !EC.pools.empty(), raw = blocks(M.out_w);
        for (const auto& P : EC.pools) all = all && P.n_active >= P.n_layers * c.E;
        for (const auto& Ly : M.L)
            raw = raw && blocks(Ly.wo) && (Ly.sliding ? blocks(Ly.wqkvrg) : (blocks(Ly.wqrg) && blocks(Ly.wk) && blocks(Ly.wv)));
        if (all && raw) {
            const double tw = now();
            const uint64_t m0 = EC.misses, b0 = EC.bytes_up;
            int n = 0;
            for (int l = 0; l < c.n_layer; ++l)
                for (int e = 0; e < c.E; ++e)
                    if (!EC.has(l, e)) EC.get(M.L[l], l, e, EC.tick, g_cs, nullptr, true, true), ++n;
            CU(cudaStreamSynchronize(g_cs));
            ++EC.tick;
            for (int l = 0; l < c.n_layer; ++l) {
                Layer& Ly = M.L[l];
                const auto& P = EC.pools[EC.layer_pool[l]];
                std::vector<const uint8_t*> tab(c.E);
                for (int e = 0; e < c.E; ++e) tab[e] = P.data + (size_t) EC.loc[(size_t) l * c.E + e] * P.slot_bytes;
                Ly.tab_blob = dalloc<const uint8_t*>(c.E);
                CU(cudaMemcpy(Ly.tab_blob, tab.data(), (size_t) c.E * sizeof(void*), cudaMemcpyHostToDevice));
                Ly.tab_h = tab;
                for (int r = 0; r < 3; ++r)
                    if (!Ly.scale[r].empty()) Ly.tab_sc[r] = dev_f32(Ly.scale[r]);
            }
            EC.misses = m0, EC.bytes_up = b0;
            g_fast = true;
            std::printf("resident: every expert in VRAM (%d copied in %.1f s); tokens take the resident path%s\n", n,
                        now() - tw, g_graph ? ", a CUDA graph each" : "");
        } else {
            std::printf("resident path off: %s\n", all ? "a dense matrix is not kept as blocks" : "the cache does not hold every expert");
        }
    }
    std::fflush(stdout);
    }
    if (g_ep && n_gpu == 2) {
        bool ep_all = true;
        for (int d = 0; d < 2; ++d) {
            auto& EC = dev_cache(d);
            for (int l = 0; l < c.n_layer; ++l) {
                for (int e = 0; e < c.E; ++e) {
                    if (ep_owner(l, e) == d && !EC.has(l, e)) {
                        ep_all = false;
                        break;
                    }
                }
                if (!ep_all) break;
            }
        }
        if (ep_all) {
            for (int d = 0; d < 2; ++d) {
                use_dev(d);
                auto& EC = dev_cache(d);
                for (int l = 0; l < c.n_layer; ++l) {
                    Layer& Ly = M.L[l];
                    const auto& P = EC.pools[EC.layer_pool[l]];
                    std::vector<const uint8_t*> tab(c.E, nullptr);
                    for (int e = 0; e < c.E; ++e) {
                        if (ep_owner(l, e) == d) {
                            tab[e] = P.data + (size_t) EC.loc[(size_t) l * c.E + e] * P.slot_bytes;
                        }
                    }
                    Ly.ep_tab_blob[d] = dalloc<const uint8_t*>(c.E);
                    CU(cudaMemcpy((void*) Ly.ep_tab_blob[d], tab.data(), (size_t) c.E * sizeof(void*), cudaMemcpyHostToDevice));
                    for (int r = 0; r < 3; ++r) {
                        if (!Ly.scale[r].empty()) Ly.ep_tab_sc[d][r] = dev_f32(Ly.scale[r]);
                    }
                }
            }
            use_dev(0);
            g_ep_fast = true;
            std::printf("resident: every expert in VRAM across 2 GPUs; zero-wait GPU EP dispatch enabled\n");
            std::fflush(stdout);
        } else {
            std::printf("resident EP path off: not all owned experts resident in VRAM\n");
            std::fflush(stdout);
        }
    }
    // the caches' counters, summed over the GPUs
    auto ec_sum = [&](auto f) {
        double v = 0;
        for (auto& D : g_dev) v += (double) f(D.EC);
        return v;
    };

    if (serve) {
        // --serve: requests on stdin, a line each - "GEN max_new temperature top_p seed stop,stop ids,ids,..." - and
        // the answer on stdout as "@TOK id" lines then "@END reason prompt_tokens cached_tokens prompt_s gen_s".  The
        // tokens whose state the model holds are kept: a request that starts with them runs only what follows.
        std::vector<int> hist;
        std::vector<uint64_t> hist_sig;   // a picture row's hash at its position (shorter than hist: the rest are tokens)
        std::vector<float> last;
        double t_exp = 0;
        // Snapshots of every short conv's history (plus sliding-layer KV windows in host RAM) at the end of a
        // request's prompt (before its generation): an agent's next turn usually agrees with the last one up to
        // there and no further (the model's own reply comes back re-rendered, its thinking dropped), so the state
        // goes back to that point instead of the start. Full-layer KV needs nothing: positions rewrite cleanly.
        struct Snap {
            int pos = -1;
            std::vector<float*> h;   // [layer * 4 + conv]
            std::vector<std::vector<uint8_t>> kv_k, kv_v;
        };
        const size_t kv_row_b = kv_tok_b(KD);
        std::vector<Snap> snaps(8);
        for (auto& sn : snaps) {
            sn.kv_k.resize(M.V.size());
            sn.kv_v.resize(M.V.size());
            for (size_t l = 0; l < M.V.size(); ++l) {   // (l: a visit)
                auto& Ly = bind_visit(M, (int) l);
                use_dev(Ly.dev);
                const size_t cc[4] = {KD, KD, C, C};
                for (int i = 0; i < 4; ++i) sn.h.push_back(dalloc<float>(3 * cc[i]));
                if (Ly.sliding && Ly.kv_cap < ctx) {
                    const size_t bytes = (size_t) c.window * kv_row_b;
                    sn.kv_k[l].resize(bytes);
                    sn.kv_v[l].resize(bytes);
                }
            }
        }
        int snap_next = 0;
        auto snap_copy = [&](Snap& sn, bool save) {
            const size_t cc[4] = {KD, KD, C, C};
            const cudaMemcpyKind kv_dir = save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            const size_t row_b = kv_row_b;
            for (size_t l = 0; l < M.V.size(); ++l) {   // (l: a visit)
                Layer& Ly = bind_visit(M, (int) l);
                use_dev(Ly.dev);
                float* hs[4] = {Ly.h_k, Ly.h_v, Ly.h_attn, Ly.h_mlp};
                for (int i = 0; i < 4; ++i) {
                    float* sv = sn.h[l * 4 + i];
                    CU(cudaMemcpyAsync(save ? sv : hs[i], save ? hs[i] : sv, 3 * cc[i] * 4, cudaMemcpyDeviceToDevice, g_s));
                }
                if (sn.pos > 0 && Ly.sliding && Ly.kv_cap < ctx) {
                    const int w = std::min(sn.pos, c.window);
                    const int k_start = sn.pos - w;
                    const int s0 = k_start % Ly.kv_cap;
                    const int n1 = std::min(w, Ly.kv_cap - s0);
                    const int n2 = w - n1;
                    uint8_t* dk = kv_b8() ? Ly.kc8 : (uint8_t*) Ly.kc;
                    uint8_t* dv = kv_b8() ? Ly.vc8 : (uint8_t*) Ly.vc;
                    uint8_t* hk = sn.kv_k[l].data();
                    uint8_t* hv = sn.kv_v[l].data();
                    CU(cudaMemcpyAsync(save ? hk : dk + (size_t) s0 * row_b, save ? dk + (size_t) s0 * row_b : hk,
                                       (size_t) n1 * row_b, kv_dir, g_s));
                    CU(cudaMemcpyAsync(save ? hv : dv + (size_t) s0 * row_b, save ? dv + (size_t) s0 * row_b : hv,
                                       (size_t) n1 * row_b, kv_dir, g_s));
                    if (n2 > 0) {
                        CU(cudaMemcpyAsync(save ? hk + (size_t) n1 * row_b : dk,
                                           save ? dk : hk + (size_t) n1 * row_b, (size_t) n2 * row_b, kv_dir, g_s));
                        CU(cudaMemcpyAsync(save ? hv + (size_t) n1 * row_b : dv,
                                           save ? dv : hv + (size_t) n1 * row_b, (size_t) n2 * row_b, kv_dir, g_s));
                    }
                }
            }
            for (auto& D : g_dev) CU(cudaStreamSynchronize(D.s));
        };
        auto reset = [&]() {
            for (auto& sn : snaps) sn.pos = -1;
            for (int vis = 0; vis < (int) M.V.size(); ++vis) {
                Layer& Ly = bind_visit(M, vis);
                use_dev(Ly.dev);
                CU(cudaMemset(Ly.h_k, 0, 3 * KD * 4));
                CU(cudaMemset(Ly.h_v, 0, 3 * KD * 4));
                CU(cudaMemset(Ly.h_attn, 0, 3 * C * 4));
                CU(cudaMemset(Ly.h_mlp, 0, 3 * C * 4));
            }
            hist.clear();
            hist_sig.clear();
            last.clear();
        };
        auto parse_ints = [](const std::string& t) {
            std::vector<int> v;
            for (const char* p = t.c_str(); *p;) {
                if (*p == '-' && p[1] == 0) break;
                v.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
            return v;
        };
        std::signal(SIGUSR1, on_usr1);
        // The engine protocol (serve/server.py's StrataEngine): "GEN <max_new> [key=value ...] <ids>" in; "PP ..." per
        // prompt pass, "T <id>" per token and "DONE <generated> <prompt> <prompt_ms> <decode_ms> <stop|length|cancel>
        // 0 0 <reused>" out; "STOP" mid-request ends it at the next token (or between the prompt's passes), "QUIT"
        // ends the engine.  The tokens a request ends at: --eos, else the GGUF's eos and <|endoftext|>.
        if (eos_ids.empty()) {
            if (const auto* e = S.get("tokenizer.ggml.eos_token_id")) eos_ids.push_back((int) e->u);
            eos_ids.push_back(199999);   // (Inkling 2's <|endoftext|>)
        }
        if (serve_strata) {
            std::printf("INFO engine=ik2 engine_kind=ik2 model=Inkling-2 kv=%s ctx=%d gpus=%d\n", g_kv_i8 ? "i8" : g_kv_fp8 ? "fp8" : "f16", ctx, n_gpu);
            std::printf("READY %d stop\n", ctx);
        } else {
            std::printf("@READY ctx %d\n", ctx);
        }
        std::fflush(stdout);
        // stdin on a thread: a STOP while a request runs is seen at once (the request loop looks at g_cancel)
        // (static: the reader thread is detached and may outlive this scope at QUIT)
        static std::mutex in_m;
        static std::condition_variable in_cv;
        static std::deque<std::string> in_q;
        static bool in_eof = false;
        static std::atomic<bool> busy{false};
        std::thread([]() {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    if (busy) g_cancel = 1;
                    continue;
                }
                std::lock_guard<std::mutex> lk(in_m);
                in_q.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_m);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
        auto next_line = [&](std::string& l) {
            std::unique_lock<std::mutex> lk(in_m);
            in_cv.wait(lk, [&] { return !in_q.empty() || in_eof; });
            if (in_q.empty()) return false;
            l = std::move(in_q.front());
            in_q.pop_front();
            return true;
        };
        std::string line;
        std::mt19937_64 rng(1234);
        while (next_line(line)) {
            std::istringstream in(line);
            std::string cmd, stops_s, ids_s, emb_path;
            int max_new = 0, top_k = 0;
            double temp = 0, top_p = 1;
            long long seed = -1;
            in >> cmd;
            if (cmd == "QUIT") {
                pop_dump();
                break;
            }
            // GENI: GEN with the request's pictures - "GENI <max_new> [key=value ...] <pictures file> <ids>", the file
            // the image encoder's rows (SVE1 records back to back), one row an image token of the ids in order
            const bool geni = cmd == "GENI";
            if (cmd != "GEN" && !geni) {   // (PAUSE / WRAP and the like: not offered in READY, so not sent)
                if (!serve_strata) {
                    std::printf("@END error unknown_command 0 0 0 0\n");
                    std::fflush(stdout);
                }
                continue;
            }
            std::vector<int> stops;
            if (serve_strata) {
                in >> max_new;
                std::string w;
                while (in >> w) {
                    const size_t eq = w.find('=');
                    if (eq == std::string::npos) {
                        if (geni && emb_path.empty() && ids_s.empty() && w.find_first_not_of("0123456789,-") != std::string::npos)
                            emb_path = w;   // (the pictures' file comes before the ids)
                        else
                            ids_s = w;
                        continue;
                    }
                    const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                    if (k == "temperature") temp = std::atof(v.c_str());
                    else if (k == "top_p") top_p = std::atof(v.c_str());
                    else if (k == "top_k") top_k = std::atoi(v.c_str());
                    else if (k == "seed") seed = std::atoll(v.c_str());
                }
                stops = eos_ids;
            } else {
                in >> max_new >> temp >> top_p >> seed >> stops_s >> ids_s;
                stops = parse_ints(stops_s);
            }
            const std::vector<int> ids = parse_ints(ids_s);
            if (ids.empty() || (int) ids.size() + max_new > ctx) {
                if (serve_strata)
                    std::printf("ERR %s\nDONE 0 %zu 0 0 cancel 0 0 0\n",
                                ids.empty() ? "an empty prompt" :
                                    ("the prompt and max_tokens pass the context (" + std::to_string(ctx) + ")").c_str(),
                                ids.size());
                else
                    std::printf("@END error %s %zu 0 0 0\n", ids.empty() ? "empty_prompt" : "context_length", ids.size());
                std::fflush(stdout);
                continue;
            }
            g_img.clear();
            std::vector<uint64_t> sig(ids.size(), 0);
            if (!emb_path.empty()) {
                const std::string err = load_pictures(emb_path, ids, image_token, c.C, g_img, sig);
                if (!err.empty()) {
                    g_img.clear();
                    std::printf("ERR %s\nDONE 0 %zu 0 0 cancel 0 0 0\n", err.c_str(), ids.size());
                    std::fflush(stdout);
                    continue;
                }
            }
            busy = true;
            if (seed >= 0) rng.seed((uint64_t) seed);
            g_cancel = 0;
            size_t L = 0;   // (a picture's position is the same only with the same rows: its hash beside its token)
            auto hsig = [&](size_t i) { return i < hist_sig.size() ? hist_sig[i] : (uint64_t) 0; };
            while (L < hist.size() && L < ids.size() && hist[L] == ids[L] && hsig(L) == sig[L]) ++L;
            if (L < hist.size() || L == ids.size()) {
                // back to the latest snapshot at or before L that leaves a token to run (one past it is dropped:
                // its position's state follows from a prefix the request does not share)
                int best = -1;
                for (int i = 0; i < (int) snaps.size(); ++i)
                    if (snaps[i].pos >= 0 && snaps[i].pos <= (int) L && snaps[i].pos < (int) ids.size() &&
                        (best < 0 || snaps[i].pos > snaps[best].pos))
                        best = i;
                if (L == hist.size() && L == ids.size() && !last.empty()) {
                    // (the same tokens again: the last logits answer)
                } else if (best >= 0) {
                    const int sp = snaps[best].pos;
                    snap_copy(snaps[best], false);
                    for (auto& sn : snaps)
                        if (sn.pos > sp) sn.pos = -1;
                    hist.resize(sp);
                    if (hist_sig.size() > (size_t) sp) hist_sig.resize(sp);
                    last.clear();
                    L = sp;
                } else {
                    reset(), L = 0;
                }
            }
            const int cached = (int) L;
            const double t0 = now();
            const uint64_t miss0 = dev_cache(0).misses, bytes0 = dev_cache(0).bytes_up;
            for (auto& v : g_cnt_hist) v = 0;
            for (auto& v : g_src_hist) v = 0;
            bool cancelled = false;
            if (L < ids.size()) {
                const int T_sup = g_dev[0].W.T_sup;
                const int sub_chunk = std::min(g_dev[0].W.Tm, g_sub_max);
                if (T_sup > 0 && (n_gpu == 1 || dual()) && (int) (ids.size() - L) > 64) {
                    BigScratch bs;
                    const int T_pass = big_begin(M, g_dev[0].W, dev_cache(0), (int) (ids.size() - L), (int) ids.size(), bs);
                    for (size_t p0 = L; p0 < ids.size(); p0 += T_pass) {
                        if (g_cancel) {   // (a prompt half run: nothing of it is kept)
                            cancelled = true;
                            break;
                        }
                        const int n = (int) std::min<size_t>(T_pass, ids.size() - p0);
                        const bool last_c = p0 + n == ids.size();
                        std::vector<float> lo;
                        forward_prefill_sup(M, ids.data() + p0, n, (int) p0, sub_chunk,
                                            last_c ? OutMode::kLogits : OutMode::kNone, lo, nullptr, t_exp);
                        if (last_c) last = std::move(lo);
                        if (serve_strata) {   // (read, total, ms, tok/s, reused)
                            const double ms = (now() - t0) * 1e3;
                            std::printf("PP %zu %zu %.0f %.1f %d\n", p0 + n, ids.size(), ms,
                                        (p0 + n - L) / std::max(1e-3, ms / 1e3), (int) L);
                            std::fflush(stdout);
                        }
                    }
                    big_end(g_dev[0].W, dev_cache(0), bs);
                } else {
                    for (size_t p0 = L; p0 < ids.size(); p0 += chunk) {
                        if (g_cancel) {   // (a prompt half run: nothing of it is kept)
                            cancelled = true;
                            break;
                        }
                        const int n = (int) std::min<size_t>(chunk, ids.size() - p0);
                        const bool last_c = p0 + n == ids.size();
                        std::vector<float> lo;
                        forward(M, ids.data() + p0, n, (int) p0, false, last_c ? OutMode::kLogits : OutMode::kNone,
                                std::string(), lo, nullptr, t_exp);
                        if (last_c) last = std::move(lo);
                    }
                }
            }
            if (cancelled) {
                reset();
                if (serve_strata) std::printf("DONE 0 %zu %.1f 0 cancel 0 0 %d\n", ids.size(), (now() - t0) * 1e3, cached);
                else std::printf("@END cancelled %zu %d %.3f %.3f\n", ids.size(), cached, now() - t0, 0.0);
                std::fflush(stdout);
                busy = false;
                continue;
            }
            if (L < ids.size()) {
                hist = ids;
                hist_sig = sig;
                Snap& sn = snaps[snap_next];
                snap_next = (snap_next + 1) % (int) snaps.size();
                sn.pos = (int) hist.size();
                snap_copy(sn, true);
            }
            const double t1 = now();
            if (getenv("IK2_STATS"))
                std::printf("prefill stats: %zu tokens run, %llu expert copies (%.1f GiB) in %.2f s; experts that ran 1 / 2 / 3-7 / 8+ "
                            "tokens: %llu / %llu / %llu / %llu\n",
                            ids.size() - (size_t) cached, (unsigned long long) (dev_cache(0).misses - miss0),
                            (dev_cache(0).bytes_up - bytes0) / 1073741824.0, t1 - t0,
                            (unsigned long long) g_cnt_hist[0], (unsigned long long) g_cnt_hist[1],
                            (unsigned long long) g_cnt_hist[2], (unsigned long long) g_cnt_hist[3]);
            if (getenv("IK2_STATS"))
                std::printf("prefill sources: used and cached %llu, used and prefetched %llu, used and copied after the "
                            "routing %llu, prefetched and unused %llu\n",
                            (unsigned long long) g_src_hist[0], (unsigned long long) g_src_hist[1],
                            (unsigned long long) g_src_hist[2], (unsigned long long) g_src_hist[3]);
            // softmax(logits / T) over the top_p mass of the 1000 largest of the (logit, token) pairs given (the
            // whole vocabulary, or the candidates the resident token path found on the device: every logit of the
            // highest bins that hold 1000 - the same 1000 either way; ties: the lower token first)
            auto pick_top = [&](std::vector<std::pair<float, int>>& cd) -> int {
                const int kk = std::min((int) cd.size(), top_k > 0 ? std::min(top_k, kCandK) : kCandK);
                std::partial_sort(cd.begin(), cd.begin() + kk, cd.end(),
                                  [](const std::pair<float, int>& a2, const std::pair<float, int>& b2) {
                                      return a2.first > b2.first || (a2.first == b2.first && a2.second < b2.second);
                                  });
                std::vector<double> pr(kk);
                double z = 0;
                for (int j = 0; j < kk; ++j) z += (pr[j] = std::exp((cd[j].first - cd[0].first) / temp));
                double acc = 0;
                int keep = kk;
                for (int j = 0; j < kk; ++j) {
                    acc += pr[j] / z;
                    if (acc >= top_p) {
                        keep = j + 1;
                        break;
                    }
                }
                double zk = 0;
                for (int j = 0; j < keep; ++j) zk += pr[j];
                double u = std::uniform_real_distribution<double>(0, zk)(rng);
                for (int j = 0; j < keep; ++j) {
                    u -= pr[j];
                    if (u <= 0) return cd[j].second;
                }
                return cd[keep - 1].second;
            };
            auto pick = [&](const float* r) -> int {
                if (temp <= 0) return (int) (std::max_element(r, r + c.n_vocab) - r);
                std::vector<std::pair<float, int>> cd(c.n_vocab);
                for (int v = 0; v < c.n_vocab; ++v) cd[v] = {r[v], v};
                return pick_top(cd);
            };
            auto pick_cand = [&]() -> int {
                std::vector<std::pair<float, int>> cd(g_cand.n);
                for (int i = 0; i < g_cand.n; ++i) cd[i] = {g_cand.val[i], g_cand.idx[i]};
                return pick_top(cd);
            };
            // IK2_CAND=0: sampling takes the whole vocabulary's logits from the device every token
            const bool use_cand = !(getenv("IK2_CAND") && !std::atoi(getenv("IK2_CAND")));
            bool cand = false;   // the last token's logits came as candidates (g_cand), not in `last`
            const char* reason = "length";
            int n_out = 0;
            int greedy_tok = temp <= 0 && !last.empty() ? pick(last.data()) : -1;
            for (int i = 0; i < max_new; ++i) {
                if (g_cancel) {
                    reason = "cancelled";
                    break;
                }
                const int tok = temp <= 0 ? greedy_tok : cand ? pick_cand() : pick(last.data());
                ++n_out;
                std::printf(serve_strata ? "T %d\n" : "@TOK %d\n", tok);
                std::fflush(stdout);
                if (std::find(stops.begin(), stops.end(), tok) != stops.end()) {
                    reason = "stop";
                    break;
                }
                if (i + 1 == max_new) break;
                if (temp <= 0) {
                    std::vector<float> lo;
                    std::vector<int> ag;
                    forward(M, &tok, 1, (int) hist.size(), false, OutMode::kArgmax, std::string(), lo, &ag, t_exp);
                    greedy_tok = ag[0];
                    last.clear();
                } else {
                    std::vector<float> lo;
                    forward(M, &tok, 1, (int) hist.size(), false, use_cand ? OutMode::kCand : OutMode::kLogits,
                            std::string(), lo, nullptr, t_exp);
                    cand = use_cand && g_cand.n >= 0;
                    last = std::move(lo);
                }
                hist.push_back(tok);
            }
            if (serve_strata)
                std::printf("DONE %d %zu %.1f %.1f %s 0 0 %d\n", n_out, ids.size(), (t1 - t0) * 1e3, (now() - t1) * 1e3,
                            std::strcmp(reason, "cancelled") == 0 ? "cancel" : reason, cached);
            else
                std::printf("@END %s %zu %d %.3f %.3f\n", reason, ids.size(), cached, t1 - t0, now() - t1);
            std::fflush(stdout);
            busy = false;
        }
        return 0;
    }
    if ((int) prompt.size() + gen > ctx) die("prompt + gen past --ctx");
    // the prompt
    std::vector<float> logits;
    double t_exp = 0;
    const double tp = now();
    const bool all = !dump.empty();
    int first_best = -1;
    const int T_sup = g_dev[0].W.T_sup;
    const int sub_chunk = std::min(g_dev[0].W.Tm, g_sub_max);
    if (T_sup > 0 && (n_gpu == 1 || dual()) && !drafter && !all && (int) prompt.size() > 64) {
        BigScratch bs;
        const int T_pass = big_begin(M, g_dev[0].W, dev_cache(0), (int) prompt.size(), (int) prompt.size(), bs);
        if (T_pass != T_sup) std::printf("prefill passes of %d tokens (scratch lent by the expert cache)\n", T_pass);
        for (int p0 = 0; p0 < (int) prompt.size(); p0 += T_pass) {
            const int n = std::min(T_pass, (int) prompt.size() - p0);
            const bool last = p0 + n == (int) prompt.size();
            const OutMode om = (last && gen > 0) ? OutMode::kArgmax : OutMode::kNone;
            std::vector<float> lo;
            std::vector<int> ag;
            forward_prefill_sup(M, prompt.data() + p0, n, p0, sub_chunk, om, lo, &ag, t_exp);
            if (last && !ag.empty()) first_best = ag[0];
        }
        big_end(g_dev[0].W, dev_cache(0), bs);
    } else {
        for (int p0 = 0; p0 < (int) prompt.size(); p0 += chunk) {
            const int n = std::min(chunk, (int) prompt.size() - p0);
            const bool last = p0 + n == (int) prompt.size();
            const OutMode om = all ? OutMode::kLogits : (!last ? OutMode::kNone : ((gen > 0 || drafter) ? OutMode::kArgmax : OutMode::kNone));
            std::vector<float> lo;
            std::vector<int> ag;
            forward(M, prompt.data() + p0, n, p0, all, om, all ? dump : std::string(), lo, &ag, t_exp);
            if (all) logits.insert(logits.end(), lo.begin(), lo.end());
            else if (last && !ag.empty()) first_best = ag[0];
            if (drafter) {
                ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, n, p0);
                if (last) ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n - 1);
            }
        }
    }
    const double prompt_s = now() - tp;
    std::printf("prompt: %zu tokens in %.2f s (%.1f tok/s); experts %.2f s of it, uploads %.2f s; cache %llu hits, "
                "%llu misses, %.1f GiB copied\n",
                prompt.size(), prompt_s, prompt.size() / prompt_s, t_exp, ec_sum([](ExpertCache& e) { return e.t_up; }),
                (unsigned long long) ec_sum([](ExpertCache& e) { return e.hits; }),
                (unsigned long long) ec_sum([](ExpertCache& e) { return e.misses; }),
                ec_sum([](ExpertCache& e) { return e.bytes_up; }) / 1073741824.0);
    std::fflush(stdout);
    if (all) {
        // log-probabilities of every prompt position
        std::vector<float> lp(logits.size());
        for (size_t t = 0; t < prompt.size(); ++t) {
            const float* r = logits.data() + t * c.n_vocab;
            double m = -1e30, s = 0;
            for (int v = 0; v < c.n_vocab; ++v) m = std::max(m, (double) r[v]);
            for (int v = 0; v < c.n_vocab; ++v) s += std::exp(r[v] - m);
            const double ls = m + std::log(s);
            for (int v = 0; v < c.n_vocab; ++v) lp[t * c.n_vocab + v] = (float) (r[v] - ls);
        }
        FILE* f = std::fopen((dump + "/logprobs.f32").c_str(), "wb");
        std::fwrite(lp.data(), 4, lp.size(), f);
        std::fclose(f);
        const float* r_last = logits.data() + (prompt.size() - 1) * c.n_vocab;
        first_best = (int) (std::max_element(r_last, r_last + c.n_vocab) - r_last);
        logits.clear();
    }
    // greedy generation
    g_pf = Prof{};
    std::vector<int> out;
    const double tg = now();
    const uint64_t h0 = (uint64_t) ec_sum([](ExpertCache& e) { return e.hits; }),
                   m0 = (uint64_t) ec_sum([](ExpertCache& e) { return e.misses; });
    int n_steps = 0;
    const int verify_check = getenv("IK2_VERIFY_CHECK") ? std::atoi(getenv("IK2_VERIFY_CHECK")) : 0;
    double t_draft = 0, t_verify = 0;
    std::vector<int> acc_hist(ik2::draft::kBlock, 0);
    // IK2_FORCE=FILE (comma-separated ids): the tokens fed are these instead of the model's own picks, and how often
    // its pick agrees with the next one is counted - two builds then run the same tokens (a fair speed comparison)
    // and the agreement measures their numerics on real text
    std::vector<int> force;
    int agree = 0;
    if (getenv("IK2_FORCE")) {
        FILE* ff = std::fopen(getenv("IK2_FORCE"), "r");
        if (!ff) die("IK2_FORCE: cannot read the file");
        int v;
        while (std::fscanf(ff, "%d,", &v) == 1) force.push_back(v);
        std::fclose(ff);
    }
    // IK2_LOGITS_OUT=FILE: every generated token's logits [n_vocab] F32 appended there (with IK2_FORCE: two builds'
    // logits over the same tokens, to compare)
    FILE* lo = getenv("IK2_LOGITS_OUT") ? std::fopen(getenv("IK2_LOGITS_OUT"), "wb") : nullptr;
    for (int i = 0, best = first_best; i < gen && !drafter; ++i) {
        const int tok = i < (int) force.size() ? force[i] : best;
        if (i < (int) force.size()) agree += best == force[i];
        out.push_back(tok);
        std::printf("%d%s", tok, i + 1 < gen ? "," : "\n");
        std::fflush(stdout);
        if (i + 1 == gen) break;
        std::vector<int> ag;
        if (lo) logits.clear();
        const int pos = (int) prompt.size() + i;
        if (g_dp.ok && g_dp.graph && !lo && i > 0 && !g_dp.dirty && g_dp.planned <= g_dp_reset_at && !g_prof &&
            !g_trace && g_pop.empty()) {   // IK2_TOKGRAPH (its token the device plan's: see dp_tok)
            const int sl = i & 1;
            if (g_dp.exec_ep != k::scratch_epoch()) g_dp.exec_pos[0] = g_dp.exec_pos[1] = -1, g_dp.exec_ep = k::scratch_epoch();
            if (g_dp.exec_pos[sl] != pos) tg_capture(M, pos, sl);
            if (g_dp.exec_pos[sl] != pos) goto plain;   // (not captured: see tg_capture)
            const size_t rb = (size_t) c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
            Work& W0 = dev_work(0);
            k::decode_host(M.embd_type, M.embd + (size_t) tok * rb, c.C, W0.emb_h);
            CU(cudaMemcpyAsync(g_dp.emb, W0.emb_h, (size_t) c.C * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaGraphLaunch(g_dp.exec[sl], g_s));
            if (i + 2 < gen) tg_capture(M, pos + 1, sl ^ 1);   // (the next token's, while this one runs)
            CU(cudaMemcpyAsync(W0.argmax_h, W0.argmax_d, 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
            best = W0.argmax_h[0];
            continue;
        }
    plain:
        forward(M, &tok, 1, pos, false, lo ? OutMode::kLogits : OutMode::kArgmax, std::string(),
                logits, &ag, t_exp);
        if (lo) {
            std::fwrite(logits.data(), sizeof(float), (size_t) c.n_vocab, lo);
            best = (int) (std::max_element(logits.begin(), logits.begin() + c.n_vocab) - logits.begin());
        } else {
            best = ag[0];
        }
    }
    if (lo) std::fclose(lo);
    if (!force.empty())
        std::printf("forced: %d tokens fed, the model's pick agreed with the next one %d times (%.2f%%)\n",
                    (int) std::min<size_t>(force.size(), out.size()), agree,
                    100.0 * agree / std::max<size_t>(1, std::min<size_t>(force.size(), out.size())));

    if (drafter) {
        use_dev(0);
        // draft 7 after the anchor, verify the 8 in one call, keep the agreeing prefix and the target's next token
        float *raw = dalloc<float>(c.C), *anchor_h = dalloc<float>(c.C);
        auto argmax_row = [&](const float* r) { return (int) (std::max_element(r, r + c.n_vocab) - r); };
        int anchor = first_best, P = (int) prompt.size();
        out.push_back(anchor);
        // the verify window: IK2_DRAFT_B tokens (2 .. kBlock: the anchor and B - 1 of the 7 drafts)
        const int B = std::max(2, std::min(ik2::draft::kBlock, getenv("IK2_DRAFT_B") ? std::atoi(getenv("IK2_DRAFT_B")) : ik2::draft::kBlock));
        const bool margin_check = getenv("IK2_MARGIN") != nullptr;
        while ((int) out.size() < gen && P + B <= ctx) {
            use_dev(0);
            const double t0 = now();
            embed_row(M, anchor, raw, anchor_h);
            int dr[ik2::draft::kBlock - 1];
            ik2::draft::propose(drafter, anchor, anchor_h, P, M.out_w.f16, c.n_rows, c.n_vocab, dr, M.out_w.q8,
                                M.out_w.q8d);
            const double t1 = now();
            int tk[ik2::draft::kBlock];
            tk[0] = anchor;
            for (int j = 1; j < B; ++j) tk[j] = dr[j - 1];
            g_rec = true;
            logits.clear();
            std::vector<int> ag;
            const bool need_logits = n_steps < verify_check || margin_check;
            forward(M, tk, B, P, true, need_logits ? OutMode::kLogits : OutMode::kArgmax, std::string(), logits, &ag,
                    t_exp);
            if (n_steps < verify_check) {
                // the same window again through each other path (the convs' history taken back first): the
                // logits' largest difference and the argmax disagreements against the grouped run
                const bool ge = g_group_exp, ga = g_group_attn;
                const int modes[3][2] = {{0, 0}, {1, 0}, {0, 1}};
                const char* names[3] = {"neither grouped", "experts grouped only", "attention grouped only"};
                for (int m = 0; m < 3; ++m) {
                    rollback(M, 0);
                    g_group_exp = modes[m][0], g_group_attn = modes[m][1];
                    std::vector<float> lb;
                    forward(M, tk, B, P, true, OutMode::kLogits, std::string(), lb, nullptr, t_exp);
                    double mx = 0;
                    int dis = 0;
                    for (int j = 0; j < B; ++j) {
                        const float* a = logits.data() + (size_t) j * c.n_vocab;
                        const float* b = lb.data() + (size_t) j * c.n_vocab;
                        for (int v = 0; v < c.n_vocab; ++v) mx = std::max(mx, (double) std::fabs(a[v] - b[v]));
                        dis += argmax_row(a) != argmax_row(b);
                    }
                    std::printf("verify check, step %d, grouped vs %s: max |dlogit| %.4f, argmax differs in %d of %d rows\n",
                                n_steps, names[m], mx, dis, B);
                }
                g_group_exp = ge, g_group_attn = ga;
                rollback(M, 0);
                logits.clear();
                forward(M, tk, B, P, true, OutMode::kLogits, std::string(), logits, nullptr, t_exp);
            }
            g_rec = false;
            int g[ik2::draft::kBlock];
            if (need_logits) {
                for (int j = 0; j < B; ++j) g[j] = argmax_row(logits.data() + (size_t) j * c.n_vocab);
            } else {
                for (int j = 0; j < B; ++j) g[j] = ag[j];
            }
            if (margin_check) {   // the target's top-2 margin at each verified position (near ties flip)
                for (int j = 0; j < B; ++j) {
                    const float* r = logits.data() + (size_t) j * c.n_vocab;
                    float b1 = -1e30f, b2 = -1e30f;
                    int i1 = -1, i2 = -1;
                    for (int v = 0; v < c.n_vocab; ++v) {
                        if (r[v] > b1) b2 = b1, i2 = i1, b1 = r[v], i1 = v;
                        else if (r[v] > b2) b2 = r[v], i2 = v;
                    }
                    if (b1 - b2 < 0.05f)
                        std::printf("near tie at position %d: %d %.4f, %d %.4f\n", P + j, i1, b1, i2, b2);
                }
            }
            int n = 0;
            while (n < B - 1 && dr[n] == g[n]) ++n;
            if (getenv("IK2_DEBUG_DRAFT")) {
                std::printf("step %d pos %d anchor %d: dr=[", n_steps, P, anchor);
                for (int j = 0; j < B - 1; ++j) std::printf("%d%s", dr[j], j + 2 < B ? " " : "");
                std::printf("] g=[");
                for (int j = 0; j < B; ++j) std::printf("%d%s", g[j], j + 1 < B ? " " : "");
                std::printf("] match=%d\n", n);
            }
            if (n + 1 < B) rollback(M, n + 1);
            use_dev(0);
            ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, n + 1, P);
            ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n);
            for (int j = 0; j < n && (int) out.size() < gen; ++j) out.push_back(dr[j]);
            if ((int) out.size() < gen) out.push_back(g[n]);
            anchor = g[n];
            P += n + 1;
            ++n_steps;
            ++acc_hist[n];
            t_draft += t1 - t0;
            t_verify += now() - t1;
        }
        for (size_t j = 0; j < out.size(); ++j) std::printf("%d%s", out[j], j + 1 < out.size() ? "," : "\n");
    }
    const double gen_s = now() - tg;
    if (drafter && n_steps) {
        std::printf("speculative: %d steps, %.2f tokens a step (accepted drafts: ", n_steps,
                    (double) (out.size() - 1) / n_steps);
        for (int j = 0; j < ik2::draft::kBlock; ++j) std::printf("%d%s", acc_hist[j], j + 1 < ik2::draft::kBlock ? " " : "");
        std::printf("); drafting %.1f ms a step, verifying %.1f ms\n", t_draft * 1e3 / n_steps, t_verify * 1e3 / n_steps);
    }
    if (g_pool)
        std::printf("CPU lane: %d threads, %llu experts computed on the CPU (%.2f ms each at the end), %llu waited for\n",
                    g_pool->size(), (unsigned long long) g_n_cpu, g_t_cpu * 1e3, (unsigned long long) g_n_wait);
    if (g_dp.M)
        std::printf("devplan: %llu experts on the CPU, %llu copied in, %llu not (queue full); the GPU waited for the CPU "
                    "(or a late host) %d times, %.1f ms in all\n", (unsigned long long) g_dp.cpu_experts, (unsigned long long) g_dp.copies,
                    (unsigned long long) g_dp.skipped, g_dp.ctl[4], g_dp.ctl[3] / 1000.0);
    if (g_prof)
        std::printf("profile (all calls): attention %.2f s, router %.2f s, uploads %.2f s, experts %.2f s, rest %.2f s\n",
                    g_pf.attn, g_pf.route, g_pf.upload, g_pf.experts, g_pf.rest);
    if (out.size() > 1)
        std::printf("generation: %d tokens in %.2f s (%.2f tok/s); cache %llu hits, %llu misses in it\n",
                    (int) out.size() - 1, gen_s, (out.size() - 1) / gen_s, (unsigned long long) (ec_sum([](ExpertCache& e) { return e.hits; }) - h0),
                    (unsigned long long) (ec_sum([](ExpertCache& e) { return e.misses; }) - m0));
    std::printf("generated:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    std::fflush(stdout);
    if (g_trace) std::fclose(g_trace);
    pop_dump();
    return 0;
}

}  // namespace ik2

namespace strata::ik2 {
int run(int argc, char** argv) {
    return ::ik2::run(argc, argv);
}
}  // namespace strata::ik2
