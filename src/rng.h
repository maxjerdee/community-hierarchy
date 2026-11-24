// Random draws that are identical on every platform.
//
// std::mt19937 is specified exactly by the standard, so the raw stream is
// portable.  What is NOT portable is everything built on top of it:
// std::uniform_real_distribution, std::uniform_int_distribution and
// std::shuffle are all implementation-defined, and libstdc++ (native builds)
// and libc++ (the Emscripten/wasm build) consume different numbers of words in
// a different order.  Using them makes a seeded run diverge between builds from
// its first sweep -- same target distribution, different sample path.
//
// These three functions fix the algorithm, so a seed pins the run everywhere:
// native Linux/macOS/Windows and wasm all produce the same chain.  Use them
// instead of the standard facilities anywhere the sampler draws.
#ifndef COMMUNITY_HIERARCHY_RNG_H
#define COMMUNITY_HIERARCHY_RNG_H

#include <cstdint>
#include <cmath>
#include <iterator>
#include <random>

// Uniform on [0, 1) with full 53-bit resolution.  For the 32-bit generator this
// is the Mersenne Twister reference implementation's genrand_res53 (two draws);
// for the 64-bit one it is the top 53 bits of a single draw.  fit() uses both.
inline double next_uniform(std::mt19937& g) {
    const std::uint32_t a = static_cast<std::uint32_t>(g()) >> 5;   // 27 bits
    const std::uint32_t b = static_cast<std::uint32_t>(g()) >> 6;   // 26 bits
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);       // / 2^53
}

inline double next_uniform(std::mt19937_64& g) {
    return static_cast<double>(g() >> 11) * (1.0 / 9007199254740992.0);   // / 2^53
}

// Uniform on [0, n), unbiased.  Lemire's multiply-shift with rejection: one
// draw in the common case, and the rejection threshold makes the result exact
// rather than modulo-biased.
template <class Gen>
inline std::uint32_t next_below(Gen& g, std::uint32_t n) {
    if (n <= 1) return 0;
    std::uint64_t m = static_cast<std::uint64_t>(static_cast<std::uint32_t>(g())) * n;
    std::uint32_t low = static_cast<std::uint32_t>(m);
    if (low < n) {
        const std::uint32_t threshold = (0u - n) % n;               // 2^32 mod n
        while (low < threshold) {
            m = static_cast<std::uint64_t>(static_cast<std::uint32_t>(g())) * n;
            low = static_cast<std::uint32_t>(m);
        }
    }
    return static_cast<std::uint32_t>(m >> 32);
}

// Uniform integer on [lo, hi], inclusive, like std::uniform_int_distribution.
template <class Gen>
inline int next_int(Gen& g, int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + static_cast<int>(next_below(g, static_cast<std::uint32_t>(hi - lo + 1)));
}

// Fisher-Yates, walking downwards, like std::shuffle but with the swap order
// pinned.  (std::shuffle's traversal and its use of the distribution are both
// unspecified, which is what makes it non-portable.)
template <class RandomIt, class Gen>
void shuffle_portable(RandomIt first, RandomIt last, Gen& g) {
    const auto n = last - first;
    for (auto i = n - 1; i > 0; --i) {
        const auto j = static_cast<decltype(i)>(next_below(g, static_cast<std::uint32_t>(i + 1)));
        if (i != j) std::iter_swap(first + i, first + j);
    }
}

// Normal(0, sigma) by Box-Muller.  std::normal_distribution is doubly
// unportable: the algorithm differs between libraries AND it caches a second
// variate between calls, so the draw depends on call history.  This consumes
// exactly two uniforms every time.
template <class Gen>
inline double next_normal(Gen& g, double sigma) {
    double u1 = next_uniform(g);
    if (u1 < 1e-300) u1 = 1e-300;                       // log(0) guard
    const double u2 = next_uniform(g);
    return sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586476925286766559 * u2);
}

// Call-compatible with the old `uniform_random(rng)` distribution object.
struct UniformRandom {
    template <class Gen>
    double operator()(Gen& g) const { return next_uniform(g); }
};

#endif  // COMMUNITY_HIERARCHY_RNG_H
