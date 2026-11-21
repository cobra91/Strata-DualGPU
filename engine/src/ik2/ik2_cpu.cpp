// Based on Project Maya (https://github.com/mw00/project-maya)
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Licensed under the MIT License.

// src/ik2/ik2_cpu.cpp - strata-ik2's CPU lane (the interface: ik2_cpu.hpp).
#include "ik2_cpu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
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
    for (int spin = 0; spin < 200000; ++spin) {   // (the caller works the queue too, then spins: the work is short)
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
// a 110-byte superblock of 256: d, qs [64], qh [8], signs [32], scales [4]
IK2_AVX2 float dot_iq3s(const uint8_t* row, const float* x, int K) {
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < K / 256; ++sb) {
        const uint8_t* b = row + (size_t) sb * 110;
        const float d = f16_at(b);
        for (int ib = 0; ib < 8; ++ib) {
            const float db = d * (float) (1 + 2 * ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15));
            const uint32_t qh = b[66 + ib];
            const uint8_t* qs = b + 2 + 8 * ib;
            const uint8_t* sg = b + 74 + 4 * ib;
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
        const size_t rb = type == 21 ? (size_t) K / 256 * 110 : (size_t) K / 32 * (type == 20 ? 18 : 34);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* w = W + (size_t) r * rb;
            out[r - r0] = type == 21 ? dot_iq3s(w, x, K) : type == 20 ? dot_iq4nl(w, x, K) : dot_q8(w, x, K);
        }
        return;
    }
#endif
    nvfp4_rows(W, K, r0, r1, x, out);
}

namespace {
// The spin pool's state is one 64-bit word, generation << 32 | next task: a task is claimed by compare-and-swap only
// while the word carries the claimer's generation (a worker late from a run never takes or skips one of the next),
// the runs' descriptions in a ring of 4 by generation.
struct SpinRun { void (*fn)(int, void*) = nullptr; void* ctx = nullptr; int n = 0; };
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
        if (i >= r.n) return;
        if (g_spin.state.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            r.fn(i, r.ctx);
            g_spin.done.fetch_add(1, std::memory_order_release);
            v = g_spin.state.load(std::memory_order_acquire);
        }
    }
}
void spin_run(int n, void (*fn)(int, void*), void* ctx) {
    const uint32_t g = ++g_spin.gen;
    g_spin.run[g & 3] = SpinRun{fn, ctx, n};
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
                while ((g = (uint32_t) (g_spin.state.load(std::memory_order_acquire) >> 32)) == seen) _mm_pause();
                seen = g;
                spin_claim(g);
            }
        });
    for (auto& t : g_spin.th) t.detach();
}
void experts_batch(const BatchJob* J, int n, int C, int F, float gcap, float ucap) {
    if (n <= 0) return;
    static std::vector<float> gu;
    if (gu.size() < (size_t) n * 2 * F) gu.resize((size_t) n * 2 * F);
    constexpr int RP = 64;
    const int pa = 2 * F / RP, pc = C / RP, pb = (F + 127) / 128;
    spin_for(n * pa, [&](int i) {   // gate's rows [0, F), up's [F, 2F)
        const int j = i / pa, r0 = (i % pa) * RP;
        float* g = gu.data() + (size_t) j * 2 * F;
        if (r0 < F) rows(J[j].tg, J[j].gate, C, r0, r0 + RP, J[j].x, g + r0);
        else rows(J[j].tu, J[j].up, C, r0 - F, r0 - F + RP, J[j].x, g + r0);
    });
    spin_for(n * pb, [&](int i) {   // the activation in place of gate's rows
        const int j = i / pb, f0 = (i % pb) * 128, f1 = std::min(F, f0 + 128);
        float* g = gu.data() + (size_t) j * 2 * F;
        for (int f = f0; f < f1; ++f) {
            const float a = g[f] * J[j].sg, u = g[F + f] * J[j].su;
            g[f] = gcap * std::tanh(a / gcap) * (1.0f / (1.0f + std::exp(-a))) * ucap * std::tanh(u / ucap);
        }
    });
    spin_for(n * pc, [&](int i) {   // down's rows
        const int j = i / pc, r0 = (i % pc) * RP;
        rows(J[j].td, J[j].down, F, r0, r0 + RP, gu.data() + (size_t) j * 2 * F, J[j].y + r0);
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

}  // namespace ik2::cpu
