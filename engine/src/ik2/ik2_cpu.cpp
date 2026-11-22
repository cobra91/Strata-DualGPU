// Based on Project Maya (https://github.com/mw00/project-maya)
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Licensed under the MIT License.

// src/ik2/ik2_cpu.cpp - strata-ik2's CPU lane (the interface: ik2_cpu.hpp).
#include "ik2_cpu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ik2_iq_tables.h"

namespace ik2::cpu {

bool available() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
}

Pool::Pool(int n) {
    for (int i = 0; i < n; ++i)
        th_.emplace_back([this]() {
            for (;;) {
                std::function<void()> f;
                {
                    // spin a while before sleeping: an expert's chunks come ~1 ms apart, a wake-up costs tens of us
                    std::unique_lock<std::mutex> lk(m_, std::defer_lock);
                    for (int spin = 0;; ++spin) {
                        lk.lock();
                        if (stop_ || !q_.empty()) break;
                        if (spin > 20000) {
                            cv_.wait(lk, [this]() { return stop_ || !q_.empty(); });
                            break;
                        }
                        lk.unlock();
                        _mm_pause();
                    }
                    if (stop_ && q_.empty()) return;
                    f = std::move(q_.front());
                    q_.pop_front();
                }
                f();
                {
                    std::lock_guard<std::mutex> lk(m_);
                    if (--pending_ == 0) done_.notify_all();
                }
            }
        });
}
Pool::~Pool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
}
void Pool::post(std::function<void()> f) {
    {
        std::lock_guard<std::mutex> lk(m_);
        q_.push_back(std::move(f));
        ++pending_;
    }
    cv_.notify_one();
}
void Pool::wait() {
    std::unique_lock<std::mutex> lk(m_, std::defer_lock);
    // (the caller works the queue too, then spins: the work is short.  The count restarts after each task it runs, so
    // a queue that stays fed keeps it spinning - on purpose: the next task is microseconds away - and it sleeps only
    // after 200000 idle spins)
    for (int spin = 0; spin < 200000; ++spin) {
        lk.lock();
        if (pending_ == 0) return;
        if (!q_.empty()) {
            std::function<void()> f = std::move(q_.front());
            q_.pop_front();
            lk.unlock();
            f();
            lk.lock();
            if (--pending_ == 0) done_.notify_all();
            lk.unlock();
            spin = 0;
            continue;
        }
        lk.unlock();
        _mm_pause();
    }
    lk.lock();
    done_.wait(lk, [this]() { return pending_ == 0; });
}

bool g_iq3s_soa = false;
namespace {

// ggml_ue4m3_to_fp32 of every byte
struct ScaleTable {
    float v[256];
    ScaleTable() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) {
                v[x] = 0.0f;
                continue;
            }
            const int e = (x >> 3) & 0xF, m = x & 7;
            v[x] = (e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7)) * 0.5f;
        }
    }
};
const ScaleTable g_scale;

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// a 36-byte block: four UE4M3 scales, then 8 bytes a 16-value sub-block (value j < 8 the low nibble of byte j)
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
void nvfp4_rows_avx512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 36, xp += 64) {
#define IK2_SUB(sb, acc)                                                                       \
    {                                                                                          \
        const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + (sb) * 8));             \
        const __m128i lo = _mm_and_si128(raw, m4);                                             \
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);                          \
        const __m128i val = _mm_shuffle_epi8(tab, _mm_unpacklo_epi64(lo, hi));                 \
        const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(val));                        \
        const __m512 xs = _mm512_mul_ps(_mm512_loadu_ps(xp + (sb) * 16), _mm512_set1_ps(g_scale.v[b[sb]])); \
        acc = _mm512_fmadd_ps(v, xs, acc);                                                     \
    }
            IK2_SUB(0, acc0)
            IK2_SUB(1, acc1)
            IK2_SUB(2, acc0)
            IK2_SUB(3, acc1)
#undef IK2_SUB
        }
        out[r - r0] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// IQ3_S / IQ4_NL / Q8_0 on AVX2 + FMA: fp32 row dots straight over the GGUF's blocks
#define IK2_AVX2 __attribute__((target("avx2,fma,f16c")))
const uint32_t k_iq3s_grid[512] = {IK2_IQ3S_GRID};
const int8_t k_iq4nl[16] = {IK2_IQ4NL_VALUES};
// IQ3_S's grid as 4 floats an entry (8 KB, stays in L1) and the 16 sign patterns as xor masks
struct Iq3sTab {
    alignas(64) float grid4[512][4];
    alignas(64) uint32_t sign4[16][4];
    Iq3sTab() {
        for (int i = 0; i < 512; ++i)
            for (int j = 0; j < 4; ++j) grid4[i][j] = (float) ((k_iq3s_grid[i] >> (8 * j)) & 255);
        for (int n = 0; n < 16; ++n)
            for (int j = 0; j < 4; ++j) sign4[n][j] = (n >> j) & 1 ? 0x80000000u : 0u;
    }
};
const Iq3sTab g_iq3s;
IK2_AVX2 inline float f16_at(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}
IK2_AVX2 inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}
// a 110-byte superblock of 256: d, qs [64], qh [8], signs [32], scales [4] - or, the rows re-laid (g_iq3s_soa:
// k::Iq3sSoa's layout, set at load for the GPU's dp4a kernels), each part a run across the row's nb blocks
struct Iq3sBlk {
    const uint8_t *qs, *qh, *sg, *sc;
    float d;
};
inline Iq3sBlk iq3s_blk(const uint8_t* row, int nb, int sb) {
    if (!g_iq3s_soa) {
        const uint8_t* b = row + (size_t) sb * 110;
        return {b + 2, b + 66, b + 74, b + 106, f16_at(b)};
    }
    const int g = (int) ((8u - ((uint32_t) (uintptr_t) row & 7u)) & 7u);
    const uint8_t* qs = row + g;
    return {qs + 64 * sb, qs + 96 * nb + 8 * sb, qs + 64 * nb + 32 * sb, qs + 104 * nb + 4 * sb,
            f16_at(row + (2 * sb < g ? 2 * sb : g + 108 * nb + 2 * sb - g))};
}
void iq3s_rows_soa_host_impl(uint8_t* m, size_t R, int K) {
    const int nb = K / 256, n = 110 * nb;
    std::vector<uint8_t> t((size_t) n);
    for (size_t r = 0; r < R; ++r) {
        uint8_t* row = m + r * (size_t) n;
        std::memcpy(t.data(), row, (size_t) n);
        const int g = (int) ((8u - ((uint32_t) (uintptr_t) row & 7u)) & 7u);
        for (int k = 0; k < nb; ++k) {
            const uint8_t* b = t.data() + (size_t) k * 110;
            std::memcpy(row + (2 * k < g ? 2 * k : g + 108 * nb + 2 * k - g), b, 2);
            std::memcpy(row + g + 64 * k, b + 2, 64);
            std::memcpy(row + g + 96 * nb + 8 * k, b + 66, 8);
            std::memcpy(row + g + 64 * nb + 32 * k, b + 74, 32);
            std::memcpy(row + g + 104 * nb + 4 * k, b + 106, 4);
        }
    }
}
IK2_AVX2 float dot_iq3s(const uint8_t* row, const float* x, int K) {
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < K / 256; ++sb) {
        const Iq3sBlk B = iq3s_blk(row, K / 256, sb);
        const float d = B.d;
        for (int ib = 0; ib < 8; ++ib) {
            const float db = d * (float) (1 + 2 * ((B.sc[ib >> 1] >> (4 * (ib & 1))) & 15));
            const uint32_t qh = B.qh[ib];
            const uint8_t* qs = B.qs + 8 * ib;
            const uint8_t* sg = B.sg + 4 * ib;
            const float* xx = x + sb * 256 + ib * 32;
            __m256 s = _mm256_setzero_ps();
            for (int il = 0; il < 4; ++il) {
                const int i1 = qs[2 * il] | ((qh << (8 - 2 * il)) & 256), i2 = qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256);
                const __m256 w = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(g_iq3s.grid4[i1])),
                                                      _mm_load_ps(g_iq3s.grid4[i2]), 1);
                const __m256 m = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps((const float*) g_iq3s.sign4[sg[il] & 15])),
                                                      _mm_load_ps((const float*) g_iq3s.sign4[sg[il] >> 4]), 1);
                s = _mm256_fmadd_ps(_mm256_xor_ps(w, m), _mm256_loadu_ps(xx + 8 * il), s);
            }
            acc = _mm256_fmadd_ps(s, _mm256_set1_ps(db), acc);
        }
    }
    return hsum8(acc);
}
// an 18-byte block of 32: d, qs [16] (values 0..15 the low nibbles, 16..31 the high); the book by two 8-entry permutes
IK2_AVX2 float dot_iq4nl(const uint8_t* row, const float* x, int K) {
    __m256 acc = _mm256_setzero_ps();
    alignas(32) float book[16];
    for (int i = 0; i < 16; ++i) book[i] = (float) k_iq4nl[i];
    const __m256 b_lo = _mm256_load_ps(book), b_hi = _mm256_load_ps(book + 8);
    for (int bk = 0; bk < K / 32; ++bk) {
        const uint8_t* b = row + (size_t) bk * 18;
        uint64_t q0, q1;
        std::memcpy(&q0, b + 2, 8);
        std::memcpy(&q1, b + 10, 8);
        const __m256i n[4] = {
            _mm256_and_si256(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) q0)), _mm256_set1_epi32(15)),
            _mm256_and_si256(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) q1)), _mm256_set1_epi32(15)),
            _mm256_srli_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) q0)), 4),
            _mm256_srli_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) q1)), 4)};
        __m256 s = _mm256_setzero_ps();
        for (int q = 0; q < 4; ++q) {
            const __m256 lo = _mm256_permutevar8x32_ps(b_lo, n[q]), hi = _mm256_permutevar8x32_ps(b_hi, n[q]);
            const __m256 w = _mm256_blendv_ps(lo, hi, _mm256_castsi256_ps(_mm256_slli_epi32(n[q], 28)));
            s = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + bk * 32 + 8 * q), s);
        }
        acc = _mm256_fmadd_ps(s, _mm256_set1_ps(f16_at(b)), acc);
    }
    return hsum8(acc);
}
// a 34-byte block of 32: d, qs [32] int8
IK2_AVX2 float dot_q8(const uint8_t* row, const float* x, int K) {
    __m256 acc = _mm256_setzero_ps();
    for (int bk = 0; bk < K / 32; ++bk) {
        const uint8_t* b = row + (size_t) bk * 34;
        __m256 s = _mm256_setzero_ps();
        for (int q = 0; q < 4; ++q) {
            uint64_t v;
            std::memcpy(&v, b + 2 + 8 * q, 8);
            const __m256 w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128((long long) v)));
            s = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + bk * 32 + 8 * q), s);
        }
        acc = _mm256_fmadd_ps(s, _mm256_set1_ps(f16_at(b)), acc);
    }
    return hsum8(acc);
}
// ... and by int8 dot products (experts_batch, IK2_CPU_Q8 default on): the input made int8 with a scale per 32 (as
// the GPU's dp4a kernels), a 32-value sub-block one maddubs - IQ3_S's unsigned grid bytes against the input with
// the weights' signs applied (sign_epi8), IQ4_NL's book by one pshufb.  The fp32 dots were compute-bound (one
// thread 1.8 GB/s of weights).
IK2_AVX2 void quant_q8_32(const float* x, int K, int8_t* q, float* sc) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int b = 0; b < K / 32; ++b) {
        const float* xb = x + b * 32;
        __m256 m = _mm256_setzero_ps();
        for (int j = 0; j < 4; ++j) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(xb + 8 * j)));
        __m128 a = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        a = _mm_max_ps(a, _mm_movehl_ps(a, a));
        a = _mm_max_ss(a, _mm_movehdup_ps(a));
        const float mx = _mm_cvtss_f32(a);
        sc[b] = mx > 1e-30f ? mx * (1.0f / 127.0f) : 0.0f;   // (a subnormal max: zero codes, a zero scale)
        const __m256 f = _mm256_set1_ps(mx > 1e-30f ? 127.0f / mx : 0.0f);
        __m256i v[4];
        for (int j = 0; j < 4; ++j) v[j] = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + 8 * j), f));   // (nearest)
        const __m256i p01 = _mm256_packs_epi32(v[0], v[1]), p23 = _mm256_packs_epi32(v[2], v[3]);
        const __m256i p = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p01, p23), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*) (q + b * 32), p);
    }
}
IK2_AVX2 float dot_iq3s_q8(const uint8_t* row, const int8_t* xq, const float* xs, int K) {
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                                          3, 3, 3, 3, 3);
    const __m256i bits = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const __m256i one8 = _mm256_set1_epi8(1), one16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < K / 256; ++sb) {
        const Iq3sBlk B = iq3s_blk(row, K / 256, sb);
        const float d = B.d;
        for (int ib = 0; ib < 8; ++ib) {
            const float db = d * (float) (1 + 2 * ((B.sc[ib >> 1] >> (4 * (ib & 1))) & 15));
            const uint32_t qh = B.qh[ib];
            const uint8_t* qs = B.qs + 8 * ib;
            uint32_t s32;
            std::memcpy(&s32, B.sg + 4 * ib, 4);
            const uint32_t* G = k_iq3s_grid;
            const __m256i g = _mm256_setr_epi32(   // (8 scalar lookups: an AVX2 gather measured 1.5x slower on Zen 2)
                (int) G[qs[0] | ((qh << 8) & 256)], (int) G[qs[1] | ((qh << 7) & 256)], (int) G[qs[2] | ((qh << 6) & 256)],
                (int) G[qs[3] | ((qh << 5) & 256)], (int) G[qs[4] | ((qh << 4) & 256)], (int) G[qs[5] | ((qh << 3) & 256)],
                (int) G[qs[6] | ((qh << 2) & 256)], (int) G[qs[7] | ((qh << 1) & 256)]);
            const __m256i sm = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) s32), shuf), bits), bits);
            const int blk = sb * 8 + ib;
            const __m256i x = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (xq + blk * 32)), _mm256_or_si256(sm, one8));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(g, x), one16);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(db * xs[blk]), acc);
        }
    }
    return hsum8(acc);
}
IK2_AVX2 float dot_iq4nl_q8(const uint8_t* row, const int8_t* xq, const float* xs, int K) {
    const __m256i book = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) k_iq4nl));
    const __m256i m4 = _mm256_set1_epi8(15), one16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int bk = 0; bk < K / 32; ++bk) {
        const uint8_t* b = row + (size_t) bk * 18;
        const __m128i q = _mm_loadu_si128((const __m128i*) (b + 2));
        const __m256i idx = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), m4);   // (values 0..15, 16..31)
        const __m256i w = _mm256_shuffle_epi8(book, idx);
        const __m256i x = _mm256_loadu_si256((const __m256i*) (xq + bk * 32));
        const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)), one16);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(f16_at(b) * xs[bk]), acc);
    }
    return hsum8(acc);
}
// ... several tokens' dots of one row (a verify window's tokens on one expert): each sub-block's weights made once
// (IQ3_S's grid vector and sign mask, IQ4_NL's book lookup), then one maddubs a token
template <int NT>
IK2_AVX2 void dot_iq3s_q8_n(const uint8_t* row, const int8_t* const* xq, const float* const* xs, int K, float* out) {
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                                          3, 3, 3, 3, 3);
    const __m256i bits = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const __m256i one8 = _mm256_set1_epi8(1), one16 = _mm256_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int sb = 0; sb < K / 256; ++sb) {
        const Iq3sBlk B = iq3s_blk(row, K / 256, sb);
        const float d = B.d;
        for (int ib = 0; ib < 8; ++ib) {
            const float db = d * (float) (1 + 2 * ((B.sc[ib >> 1] >> (4 * (ib & 1))) & 15));
            const uint32_t qh = B.qh[ib];
            const uint8_t* qs = B.qs + 8 * ib;
            uint32_t s32;
            std::memcpy(&s32, B.sg + 4 * ib, 4);
            const uint32_t* G = k_iq3s_grid;
            const __m256i g = _mm256_setr_epi32(
                (int) G[qs[0] | ((qh << 8) & 256)], (int) G[qs[1] | ((qh << 7) & 256)], (int) G[qs[2] | ((qh << 6) & 256)],
                (int) G[qs[3] | ((qh << 5) & 256)], (int) G[qs[4] | ((qh << 4) & 256)], (int) G[qs[5] | ((qh << 3) & 256)],
                (int) G[qs[6] | ((qh << 2) & 256)], (int) G[qs[7] | ((qh << 1) & 256)]);
            const __m256i sv = _mm256_or_si256(
                _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) s32), shuf), bits), bits), one8);
            const int blk = sb * 8 + ib;
            for (int t = 0; t < NT; ++t) {
                const __m256i x = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (xq[t] + blk * 32)), sv);
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(g, x), one16);
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(db * xs[t][blk]), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8(acc[t]);
}
template <int NT>
IK2_AVX2 void dot_iq4nl_q8_n(const uint8_t* row, const int8_t* const* xq, const float* const* xs, int K, float* out) {
    const __m256i book = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) k_iq4nl));
    const __m256i m4 = _mm256_set1_epi8(15), one16 = _mm256_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int bk = 0; bk < K / 32; ++bk) {
        const uint8_t* b = row + (size_t) bk * 18;
        const __m128i q = _mm_loadu_si128((const __m128i*) (b + 2));
        const __m256i w = _mm256_shuffle_epi8(book, _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), m4));
        const __m256i wa = _mm256_sign_epi8(w, w);
        const float d = f16_at(b);
        for (int t = 0; t < NT; ++t) {
            const __m256i x = _mm256_loadu_si256((const __m256i*) (xq[t] + bk * 32));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(wa, _mm256_sign_epi8(x, w)), one16);
            acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * xs[t][bk]), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8(acc[t]);
}
template <int NT>
IK2_AVX2 void rows_q8_n_t(int type, const uint8_t* W, int K, int r0, int r1, const int8_t* const* xq,
                          const float* const* xs, float* const* out) {
    const size_t rb = type == 21 ? (size_t) K / 256 * 110 : (size_t) K / 32 * 18;   // (21 or 20: rows_q8_n checks)
    float v[NT];
    for (int r = r0; r < r1; ++r) {
        if (type == 21) dot_iq3s_q8_n<NT>(W + (size_t) r * rb, xq, xs, K, v);
        else dot_iq4nl_q8_n<NT>(W + (size_t) r * rb, xq, xs, K, v);
        for (int t = 0; t < NT; ++t) out[t][r - r0] = v[t];
    }
}
#undef IK2_AVX2
#endif

void nvfp4_rows_scalar(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        double acc = 0.0;
        for (int i = 0; i < nb; ++i, b += 36)
            for (int sb = 0; sb < 4; ++sb) {
                float a = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t by = b[4 + sb * 8 + j];
                    a += kv[by & 0xF] * x[i * 64 + sb * 16 + j] + kv[by >> 4] * x[i * 64 + sb * 16 + j + 8];
                }
                acc += (double) g_scale.v[b[sb]] * a;
            }
        out[r - r0] = (float) acc;
    }
}

}  // namespace

void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool fast = available();
    if (fast) {
        nvfp4_rows_avx512(W, K, r0, r1, x, out);
        return;
    }
#endif
    nvfp4_rows_scalar(W, K, r0, r1, x, out);
}

bool available_any() { return available() || available_for(20); }
bool available_for(int type) {
    if (type == 40) return available();
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (type == 20 || type == 21 || type == 8)
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#endif
    return false;
}
void rows(int type, const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (type == 21 || type == 20 || type == 8) {
        if (!available_for(type)) {   // (AVX2 + FMA + F16C: the dots are built for them)
            std::fprintf(stderr, "ik2 CPU lane: this CPU has no AVX2 / FMA / F16C for GGUF type %d\n", type);
            std::abort();
        }
        if (K % (type == 21 ? 256 : 32)) {   // (a row of whole blocks, as GGUF lays them out)
            std::fprintf(stderr, "ik2 CPU lane: a row of %d values is not whole blocks of type %d\n", K, type);
            std::abort();
        }
        const size_t rb = type == 21 ? (size_t) K / 256 * 110 : (size_t) K / 32 * (type == 20 ? 18 : 34);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* w = W + (size_t) r * rb;
            out[r - r0] = type == 21 ? dot_iq3s(w, x, K) : type == 20 ? dot_iq4nl(w, x, K) : dot_q8(w, x, K);
        }
        return;
    }
#endif
    if (type != 40) {   // (unreachable through available_for(): a type this build has no CPU dot for)
        std::fprintf(stderr, "ik2 CPU lane: no CPU dot for GGUF type %d in this build\n", type);
        std::abort();
    }
    nvfp4_rows(W, K, r0, r1, x, out);
}

// rows by the int8 dots (IQ3_S / IQ4_NL only; xq / xs from quant_q8)
void rows_q8(int type, const uint8_t* W, int K, int r0, int r1, const int8_t* xq, const float* xs, float* out) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if ((type != 21 && type != 20) || !available_for(type)) {
        std::fprintf(stderr, "ik2 CPU lane: no int8 CPU dot for GGUF type %d here\n", type);
        std::abort();
    }
    if (K % (type == 21 ? 256 : 32)) {
        std::fprintf(stderr, "ik2 CPU lane: a row of %d values is not whole blocks of type %d\n", K, type);
        std::abort();
    }
    const size_t rb = type == 21 ? (size_t) K / 256 * 110 : (size_t) K / 32 * 18;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* w = W + (size_t) r * rb;
        out[r - r0] = type == 21 ? dot_iq3s_q8(w, xq, xs, K) : dot_iq4nl_q8(w, xq, xs, K);
    }
#else
    std::fprintf(stderr, "ik2 CPU lane: no int8 CPU dots in this build\n");
    std::abort();
#endif
}
void quant_q8(const float* x, int K, int8_t* q, float* sc) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    quant_q8_32(x, K, q, sc);
#else
    std::fprintf(stderr, "ik2 CPU lane: no int8 CPU dots in this build\n");
    std::abort();
#endif
}
// nt tokens' rows of one matrix (1 <= nt <= 8): out[t] [r1 - r0]
void rows_q8_n(int type, const uint8_t* W, int K, int r0, int r1, int nt, const int8_t* const* xq,
               const float* const* xs, float* const* out) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if ((type != 21 && type != 20) || !available_for(type) || K % (type == 21 ? 256 : 32) || nt < 1 || nt > 8) {
        std::fprintf(stderr, "ik2 CPU lane: rows_q8_n: type %d, K %d, %d tokens\n", type, K, nt);
        std::abort();
    }
    switch (nt) {
        case 1: rows_q8_n_t<1>(type, W, K, r0, r1, xq, xs, out); break;
        case 2: rows_q8_n_t<2>(type, W, K, r0, r1, xq, xs, out); break;
        case 3: rows_q8_n_t<3>(type, W, K, r0, r1, xq, xs, out); break;
        case 4: rows_q8_n_t<4>(type, W, K, r0, r1, xq, xs, out); break;
        case 5: rows_q8_n_t<5>(type, W, K, r0, r1, xq, xs, out); break;
        case 6: rows_q8_n_t<6>(type, W, K, r0, r1, xq, xs, out); break;
        case 7: rows_q8_n_t<7>(type, W, K, r0, r1, xq, xs, out); break;
        default: rows_q8_n_t<8>(type, W, K, r0, r1, xq, xs, out); break;
    }
#else
    std::fprintf(stderr, "ik2 CPU lane: no int8 CPU dots in this build\n");
    std::abort();
#endif
}
bool g_cpu_q8 = !getenv("IK2_CPU_Q8") || std::atoi(getenv("IK2_CPU_Q8"));
bool batch_multi_ok(int tg, int tu, int td, int C, int F) {   // (jobs of several tokens: the int8 dots' path only)
    return g_cpu_q8 && tg == 21 && tu == 21 && td == 20 && C % 256 == 0 && F % 128 == 0 && available_for(21);
}

namespace {
// The spin pool's state is one 64-bit word, generation << 32 | next task: a task is claimed by compare-and-swap only
// while the word carries the claimer's generation (a worker late from a run never takes or skips one of the next),
// the runs' descriptions in a ring of 4 by generation.
// (n atomic: a worker late from an earlier run may read it while the caller rewrites the slot four runs on - its CAS
// then fails on the generation; fn and ctx are read only after a CAS that succeeded, ordered by it)
struct SpinRun { void (*fn)(int, void*) = nullptr; void* ctx = nullptr; std::atomic<int> n{0}; };
struct SpinPool {
    std::vector<std::thread> th;
    std::atomic<uint64_t> state{0};
    std::atomic<int> done{0};
    uint32_t gen = 0;
    SpinRun run[4];
    bool started = false;
} g_spin;
void spin_claim(uint32_t g) {
    uint64_t v = g_spin.state.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (v >> 32) != g) return;
        const SpinRun& r = g_spin.run[g & 3];
        const int i = (int) (uint32_t) v;
        if (i >= r.n.load(std::memory_order_relaxed)) return;
        if (g_spin.state.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            r.fn(i, r.ctx);
            g_spin.done.fetch_add(1, std::memory_order_release);
            v = g_spin.state.load(std::memory_order_acquire);
        }
    }
}
void spin_run(int n, void (*fn)(int, void*), void* ctx) {
    if (n <= 0) return;
    const uint32_t g = ++g_spin.gen;
    SpinRun& r = g_spin.run[g & 3];
    r.fn = fn, r.ctx = ctx;
    r.n.store(n, std::memory_order_relaxed);
    g_spin.done.store(0, std::memory_order_relaxed);
    g_spin.state.store((uint64_t) g << 32, std::memory_order_release);
    spin_claim(g);
    while (g_spin.done.load(std::memory_order_acquire) < n) _mm_pause();
}
template <class F>
void spin_for(int n, F f) {
    spin_run(n, [](int i, void* c) { (*static_cast<F*>(c))(i); }, &f);
}
}  // namespace

void spin_start(int workers) {
    if (g_spin.started) return;
    g_spin.started = true;
    for (int w = 0; w < workers; ++w)
        g_spin.th.emplace_back([] {
            uint32_t seen = (uint32_t) (g_spin.state.load() >> 32);
            for (;;) {
                uint32_t g;
                // spin while runs come often (a decode token's layers, ~0.1 ms apart), then back off to short sleeps
                // (a prompt, or no request: no core burnt at 100%)
                for (unsigned idle = 0; (g = (uint32_t) (g_spin.state.load(std::memory_order_acquire) >> 32)) == seen; ++idle) {
                    if (idle < (1u << 18)) _mm_pause();
                    else std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                seen = g;
                spin_claim(g);
            }
        });
    for (auto& t : g_spin.th) t.detach();
}
void experts_batch(const BatchJob* J, int n, int C, int F, float gcap, float ucap) {
    // (one caller at a time on the spin pool: the device plan's service and a verify window's host path)
    static std::mutex one;
    std::lock_guard<std::mutex> lk(one);
    if (n <= 0) return;
    thread_local std::vector<float> gu_tl;   // (the caller's: the workers below see its pointer, not their own copy)
    if (gu_tl.size() < (size_t) n * 2 * F) gu_tl.resize((size_t) n * 2 * F);
    float* const gu = gu_tl.data();
    constexpr int RP = 64;
    if (C % RP || F % RP) {   // (the tasks are whole 64-row chunks)
        std::fprintf(stderr, "ik2 CPU lane: C %d / F %d not multiples of %d\n", C, F, RP);
        std::abort();
    }
    const int pa = 2 * F / RP, pc = C / RP, pb = (F + 127) / 128;
    bool q8 = g_cpu_q8 && C % 256 == 0 && F % 128 == 0;   // (F % 128: whole 32-row tasks for any RG of 32, 64, 128)
    for (int j = 0; q8 && j < n; ++j) q8 = J[j].tg == 21 && J[j].tu == 21 && J[j].td == 20;
    if (q8) {   // the int8 dots: the inputs made int8 (x here, the activations by the tasks that make them)
        // (a job's nt tokens - 1 for the device plan's, up to 8 for a verify window's - share each weight's decode)
        int ntm = 1;
        for (int j = 0; j < n; ++j) {
            if (J[j].nt > 8) {   // (BatchJob holds 8 tokens)
                std::fprintf(stderr, "ik2 CPU lane: a job of %d tokens (8 at most)\n", J[j].nt);
                std::abort();
            }
            ntm = std::max(ntm, J[j].nt);
        }
        thread_local std::vector<int8_t> q_tl;
        thread_local std::vector<float> s_tl, g_tl;
        const size_t per = (size_t) C + F;   // (a token's x codes, then its activations')
        const size_t slots = (size_t) n * ntm;
        if (q_tl.size() < slots * per) q_tl.resize(slots * per), s_tl.resize(slots * per / 32);
        if (g_tl.size() < slots * 2 * F) g_tl.resize(slots * 2 * F);
        int8_t* const qb = q_tl.data();
        float* const sb = s_tl.data();
        float* const gb = g_tl.data();
        auto xin = [&](int j, int t) { return t == 0 || J[j].nt <= 1 ? J[j].x : J[j].xt[t]; };
        auto yout = [&](int j, int t) { return t == 0 || J[j].nt <= 1 ? J[j].y : J[j].yt[t]; };
        for (int j = 0; j < n; ++j)
            for (int t = 0; t < std::max(1, J[j].nt); ++t) {
                const size_t sl = (size_t) j * ntm + t;
                quant_q8(xin(j, t), C, qb + sl * per, sb + sl * per / 32);
            }
        constexpr int RG = 32;   // (gate_up rows a task: a 32-block of the activations; 64 measured the same)
        const int pg = F / RG;
        spin_for(n * pg, [&](int i) {   // a task: RG rows of gate and the same of up, their activation, made int8
            const int j = i / pg, f0 = (i % pg) * RG, nt = std::max(1, J[j].nt);
            const int8_t* xq[8];
            const float* xs[8];
            float *og[8], *ou[8];
            for (int t = 0; t < nt; ++t) {
                const size_t sl = (size_t) j * ntm + t;
                xq[t] = qb + sl * per, xs[t] = sb + sl * per / 32;
                og[t] = gb + sl * 2 * F + f0, ou[t] = gb + sl * 2 * F + F + f0;
            }
            rows_q8_n(21, J[j].gate, C, f0, f0 + RG, nt, xq, xs, og);
            rows_q8_n(21, J[j].up, C, f0, f0 + RG, nt, xq, xs, ou);
            for (int t = 0; t < nt; ++t) {
                const size_t sl = (size_t) j * ntm + t;
                float* g = gb + sl * 2 * F;
                for (int f = f0; f < f0 + RG; ++f) {
                    const float a = g[f] * J[j].sg, u = g[F + f] * J[j].su;
                    g[f] = gcap * std::tanh(a / gcap) * (1.0f / (1.0f + std::exp(-a))) * ucap * std::tanh(u / ucap);
                }
                quant_q8(g + f0, RG, qb + sl * per + C + f0, sb + (sl * per + C + f0) / 32);
            }
        });
        spin_for(n * pc, [&](int i) {
            const int j = i / pc, r0 = (i % pc) * RP, nt = std::max(1, J[j].nt);
            const int8_t* xq[8];
            const float* xs[8];
            float* oy[8];
            for (int t = 0; t < nt; ++t) {
                const size_t sl = (size_t) j * ntm + t;
                xq[t] = qb + sl * per + C, xs[t] = sb + (sl * per + C) / 32, oy[t] = yout(j, t) + r0;
            }
            rows_q8_n(20, J[j].down, F, r0, r0 + RP, nt, xq, xs, oy);
        });
        return;
    }
    for (int j = 0; j < n; ++j)
        if (J[j].nt > 1) {
            std::fprintf(stderr, "ik2 CPU lane: several tokens a job need the int8 dots (IQ3_S / IQ4_NL)\n");
            std::abort();
        }
    spin_for(n * pa, [&](int i) {   // gate's rows [0, F), up's [F, 2F)
        const int j = i / pa, r0 = (i % pa) * RP;
        float* g = gu + (size_t) j * 2 * F;
        if (r0 < F) rows(J[j].tg, J[j].gate, C, r0, r0 + RP, J[j].x, g + r0);
        else rows(J[j].tu, J[j].up, C, r0 - F, r0 - F + RP, J[j].x, g + r0);
    });
    spin_for(n * pb, [&](int i) {   // the activation in place of gate's rows
        const int j = i / pb, f0 = (i % pb) * 128, f1 = std::min(F, f0 + 128);
        float* g = gu + (size_t) j * 2 * F;
        for (int f = f0; f < f1; ++f) {
            const float a = g[f] * J[j].sg, u = g[F + f] * J[j].su;
            g[f] = gcap * std::tanh(a / gcap) * (1.0f / (1.0f + std::exp(-a))) * ucap * std::tanh(u / ucap);
        }
    });
    spin_for(n * pc, [&](int i) {   // down's rows
        const int j = i / pc, r0 = (i % pc) * RP;
        rows(J[j].td, J[j].down, F, r0, r0 + RP, gu + (size_t) j * 2 * F, J[j].y + r0);
    });
}

void post_expert(Pool& pool, ExpertJob* job, int C, int F, float gcap, float ucap) {
    job->gu.resize((size_t) 2 * F);
    job->h.resize(F);
    const int n = std::max(1, pool.size());
    job->left = 2 * n;   // gate's chunks and up's
    auto down = [&pool, job, C, F, n]() {
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) C * c / n), r1 = (int) ((long) C * (c + 1) / n);
            pool.post([job, F, r0, r1]() { rows(job->td, job->down, F, r0, r1, job->h.data(), job->y + r0); });
        }
    };
    for (int part = 0; part < 2; ++part)
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) F * c / n), r1 = (int) ((long) F * (c + 1) / n);
            pool.post([job, C, F, gcap, ucap, part, r0, r1, down]() {
                rows(part ? job->tu : job->tg, part ? job->up : job->gate, C, r0, r1, job->x, job->gu.data() + (size_t) part * F + r0);
                if (--job->left == 0) {   // the last chunk: the activation, then down's rows
                    for (int i = 0; i < F; ++i) {
                        const float g = job->gu[i] * job->sg, u = job->gu[(size_t) F + i] * job->su;
                        job->h[i] = gcap * std::tanh(g / gcap) * (1.0f / (1.0f + std::exp(-g))) * ucap * std::tanh(u / ucap);
                    }
                    down();
                }
            });
        }
}

void iq3s_rows_soa_host(uint8_t* m, size_t R, int K) { iq3s_rows_soa_host_impl(m, R, K); }
}  // namespace ik2::cpu
