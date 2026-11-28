/* Implementations of helper functions */

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "globals.h"
#include "helpers.h"

// log C(a, b) = lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1), for real a, b.
double log_binom(double a, double b) {
  if (b == 0) {
    return 0;
  }
  return std::lgamma(a + 1) - std::lgamma(b + 1) - std::lgamma(a - b + 1);
}

// log n!.  Tabulated per thread (chains are threads): the same values std::lgamma
// gives, computed once.  lgamma dominated the single-node and merge-split moves.
static thread_local std::vector<double> LOG_FACTORIAL_TABLE;
double log_factorial(int n) {
  if (n < 0) return std::lgamma(n + 1);
  if (n >= static_cast<int>(LOG_FACTORIAL_TABLE.size())) {
    const std::size_t old = LOG_FACTORIAL_TABLE.size();
    const std::size_t want = std::max<std::size_t>(static_cast<std::size_t>(n) + 1, 2 * old + 64);
    LOG_FACTORIAL_TABLE.resize(want);
    for (std::size_t k = old; k < want; ++k) LOG_FACTORIAL_TABLE[k] = std::lgamma(static_cast<double>(k) + 1.0);
  }
  return LOG_FACTORIAL_TABLE[static_cast<std::size_t>(n)];
}

// log Gamma(x + 1) for real x >= 0: log x! for integer x (from the table), its
// continuous extension otherwise.  For edge weights, which are real in
// weighted_SBM mode (an int argument would silently truncate them).
double log_factorial_real(double x) {
  const double r = std::round(x);
  if (r == x && r >= 0.0 && r < 1e7) return log_factorial(static_cast<int>(r));
  return std::lgamma(x + 1.0);
}

// log n!! for even n = 2k: k log 2 + log k!.
double log_factorial2(int n) {
  if (n % 2 != 0) {
    throw std::invalid_argument("log_factorial2: argument must be even, got " + std::to_string(n));
  }
  return n / 2 * std::log(2) + std::lgamma(n / 2 + 1);
}

// log of the negative binomial pmf of x with mean `mean` and concentration
// (shape) c:
//   P(x | mean, c) = C(x + c - 1, x) c^c mean^x / (c + mean)^(c + x),
// i.e. a Poisson(lambda) count with lambda ~ Gamma(shape c, mean `mean`).  c = 1
// is geometric; beyond 1/CONCENTRATION_EPSILON the Poisson limit is used.
//
// The c-dependent pieces lgamma(x + c), lgamma(c) and c log c are cached per
// thread, for TWO concentrations: calculate_H_M and flip_nodes alternate between
// concentration_in and concentration_out on every block pair, and a single slot
// was rebuilt on every alternation (flips ~7x slower whenever
// variation_in != variation_out).
namespace {
struct NegBinomCache {
  double c = -1.0, lgc = 0.0, clogc = 0.0;
  std::vector<double> lgxc;   // lgxc[x] = lgamma(x + c)
};
}  // namespace

double log_neg_binom(int x, double mean, double concentration) {
  if (x < 0) {
    return -std::numeric_limits<double>::infinity();
  }
  // P(X=0 | mean=0) = 1 for any count distribution; log = 0.
  if (mean <= 0.0) {
    return (x == 0) ? 0.0 : -std::numeric_limits<double>::infinity();
  }
  double concentration_used = concentration;
  if (concentration < CONCENTRATION_EPSILON){
    concentration_used = CONCENTRATION_EPSILON; // Avoid zero concentration
  }
  if (concentration_used > 1/CONCENTRATION_EPSILON){
    // For very large concentration, approximate with Poisson
    return -mean + x * std::log(mean) - log_factorial(x);
  }
  // log_binom(x + c - 1, x) = lgamma(x + c) - lgamma(x + 1) - lgamma(c).
  static thread_local NegBinomCache slots[2];
  static thread_local int last_filled = 1;
  NegBinomCache* t;
  if (slots[0].c == concentration_used) {
    t = &slots[0];
  } else if (slots[1].c == concentration_used) {
    t = &slots[1];
  } else {
    last_filled = 1 - last_filled;          // evict the slot filled longer ago
    t = &slots[last_filled];
    t->c = concentration_used;
    t->lgc = std::lgamma(concentration_used);
    t->clogc = concentration_used * std::log(concentration_used);
    t->lgxc.clear();
  }
  if (x >= static_cast<int>(t->lgxc.size())) {
    const std::size_t old = t->lgxc.size();
    const std::size_t want = std::max<std::size_t>(static_cast<std::size_t>(x) + 1, 2 * old + 64);
    t->lgxc.resize(want);
    for (std::size_t k = old; k < want; ++k) t->lgxc[k] = std::lgamma(static_cast<double>(k) + concentration_used);
  }
  return (t->lgxc[static_cast<std::size_t>(x)] - log_factorial(x) - t->lgc) + t->clogc +
         x * std::log(mean) - (concentration_used + x) * std::log(concentration_used + mean);
}

// Overflow protected version of log(\sum exp(x_i)); -inf for no terms or all -inf.
double log_sum_exp(const std::vector<double>& xs) {
  if (xs.size() == 0) {
    return -std::numeric_limits<double>::infinity();
  }
  double xs_max = *std::max_element(xs.begin(), xs.end());
  if (!std::isfinite(xs_max)) return xs_max;
  double result = 0;
  for (double x : xs) {
    result += std::exp(x - xs_max);
  }
  return xs_max + std::log(result);
}

double davidson_log_z(double s_a, double s_b, double log_nu) {
  const double t = davidson_log_tie(s_a, s_b, log_nu);
  const double mx = std::max(std::max(s_a, s_b), t);
  return mx + std::log(std::exp(s_a - mx) + std::exp(s_b - mx) + std::exp(t - mx));
}

double concentration_to_variation(double concentration) {
  if (concentration < 0.0) {
    throw std::runtime_error("concentration_to_variation: concentration cannot be negative, got " + std::to_string(concentration));
  }
  
  // Hardcoded cases
  if (concentration == 0.0) {
    return 1.0;
  }
  if (std::isinf(concentration)) {
    return 0.0;
  }
  
  return std::exp(log_binom(concentration - 0.5, -0.5));
}

double variation_to_concentration(double variation, double tol, int max_iter) {
  // Hardcoded cases
  if (variation == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  if (variation == 1.0) {
    return 0.0;
  }
  if (variation < 0.0 || variation > 1.0) {
    throw std::runtime_error("variation_to_concentration: variation must be in [0, 1], got " + std::to_string(variation));
  }

  // f(c) = concentration_to_variation(c) - variation (find root)
  auto f = [variation](double c) -> double {
    return concentration_to_variation(c) - variation;
  };

  // lower bound: small positive (avoid zero)
  double lo = 1e-12;
  double f_lo = f(lo);
  
  // find an upper bound where f(hi) < 0 (variation decreases with concentration)
  double hi = 1.0;
  double f_hi = f(hi);
  int it = 0;
  while (f_hi > 0 && it < 200) {
    hi *= 2.0;
    f_hi = f(hi);
    it++;
    if (hi > 1e12) {
      break;
    }
  }

  // variation(c) is strictly decreasing from 1 (c -> 0) to 0 (c -> inf), so a
  // failed bracket means the target lies beyond one of the two ends.  Return
  // that end's limit.  (Returning a fixed "reasonable" value here, as this used
  // to, silently scored e.g. variation = 1e-9 -- essentially Poisson -- as
  // concentration 1, i.e. a geometric distribution.)
  if (f_hi > 0) {
    return std::numeric_limits<double>::infinity();  // variation below f(hi): Poisson limit
  }
  if (f_lo < 0) {
    return 0.0;                                      // variation above f(lo): c -> 0 limit
  }

  // bisection
  for (int i = 0; i < max_iter; i++) {
    double mid = 0.5 * (lo + hi);
    double f_mid = f(mid);
    if (std::abs(f_mid) <= tol) {
      return mid;
    }
    // shrink interval
    if (f_lo * f_mid <= 0) {
      hi = mid;
      f_hi = f_mid;
    } else {
      lo = mid;
      f_lo = f_mid;
    }
  }

  return 0.5 * (lo + hi);
}

// CDF-space proposals.  To move a parameter a with prior P(a) and CDF F, map it
// to p = F(a), take a Gaussian random-walk step in p reflected into [0, 1], and
// map back with F^{-1}.  The step kernel K(p, p') is symmetric and the prior is
// uniform in p, so in a-space the proposal density is K(F(a), F(a')) P(a'): the
// prior cancels from the Hastings ratio, leaving only the likelihood.  This
// relies on F and F^{-1} being exact inverses of each other (see
// gaussian_proposal).
//
// Fold any real x onto [0, 1] by reflection at 0 and 1 (period 2).
static double reflect_unit(double x) {
  x = std::fmod(x, 2.0);           // any distance: fold onto (-2, 2), then onto [0, 2)
  if (x < 0.0) x += 2.0;           // (fmod(p + 2, 2) failed for steps below -2)
  if (x > 1.0) x = 2.0 - x;
  return x;
}

double uniform_proposal(double p, double move_sigma) {
  return reflect_unit(p + next_normal(rng, move_sigma));
}
// Half-Cauchy prior of width w: P(a|w) = 2 w / (pi (w^2 + a^2))
double half_cauchy_cdf(double a, double w) {
  return 2 * std::atan(a / w) / M_PI;
}
double half_cauchy_inv_cdf(double p, double w) {
  return w * std::tan(M_PI * p / 2);
}
double half_cauchy_proposal(double a, double w, double move_sigma) {
  double p = half_cauchy_cdf(a, w);
  double p_new = uniform_proposal(p, move_sigma);
  return half_cauchy_inv_cdf(p_new, w);
}
// Exponential prior with mean mu: P(a|mu) = (1/mu) exp(-a/mu) for a >= 0
double exponential_cdf(double a, double mu) {
  if (a <= 0.0) {
    return 0.0;
  }
  return 1.0 - std::exp(-a / mu);
}

double exponential_inv_cdf(double p, double mu) {
  // p in [0,1); for p == 1 return +inf
  if (p >= 1.0) {
    return std::numeric_limits<double>::infinity();
  }
  if (p <= 0.0) {
    return 0.0;
  }
  return -mu * std::log(1.0 - p);
}

double exponential_proposal(double a, double mu, double move_sigma) {
  double p = exponential_cdf(a, mu);
  double p_new = uniform_proposal(p, move_sigma);
  return exponential_inv_cdf(p_new, mu);
}

// ---------------------------------------------------------------------------
// Gaussian CDF and quantile, accurate in both tails.
//
// The score move (MCMC::propose_score_move) is a CDF-space random walk on the
// exact Gaussian conditional of one score, and its Hastings ratio assumes that
// the CDF and the quantile are exact inverses.  The former float-precision
// inverse_erf round-tripped to ~1e-7 in the bulk and to ~1% beyond 3.3 sigma,
// and 0.5 (1 + erf) saturated to exactly 0 or 1 beyond ~8.3 sigma.
//
// normal_lower_tail(z) = Phi(-|z|) via erfc keeps full relative precision in
// either tail.  normal_quantile_lower(t), 0 < t <= 0.5, inverts it: Acklam's
// rational approximation (P. J. Acklam, "An algorithm for computing the inverse
// normal cumulative distribution function", 2003; |relative error| < 1.15e-9)
// followed by one Halley step on erfc, as Acklam recommends, which brings it to
// double precision (round trip |Phi(x) - t| / t ~ 1e-15, checked down to
// t = 1e-300 in tests/mcmc_core_test.cpp).
// ---------------------------------------------------------------------------
namespace {
const double SQRT2 = 1.41421356237309504880;
const double SQRT2PI = 2.50662827463100050242;

double normal_lower_tail(double z) {       // Phi(-|z|)
  return 0.5 * std::erfc(std::fabs(z) / SQRT2);
}

double normal_quantile_lower(double t) {   // Phi^{-1}(t) <= 0 for 0 < t <= 0.5
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                             6.680131188771972e+01, -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                             3.754408661907416e+00};
  if (!(t > 0.0)) return -std::numeric_limits<double>::infinity();
  double x;
  if (t < 0.02425) {                       // lower tail
    const double q = std::sqrt(-2.0 * std::log(t));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else {                                 // central region
    const double q = t - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  }
  // One Halley step on f(x) = Phi(x) - t, with Phi(x) = erfc(-x / sqrt 2) / 2:
  // u = f / f' = e / phi(x).  Below t ~ 1e-308 exp(x^2 / 2) overflows and the
  // (then 1e-9-accurate) Acklam value is kept.
  const double e = 0.5 * std::erfc(-x / SQRT2) - t;
  const double u = e * SQRT2PI * std::exp(0.5 * x * x);
  if (std::isfinite(u)) x -= u / (1.0 + 0.5 * x * u);
  return std::min(x, 0.0);
}
}  // namespace

// Gaussian CDF: computed through the nearer tail, so the lower tail keeps full
// relative precision (the upper tail is limited by 1 - Phi rounding, ~1e-16).
double gaussian_cdf(double a, double mu, double sigma) {
  const double z = (a - mu) / sigma;
  const double t = normal_lower_tail(z);
  return z <= 0.0 ? t : 1.0 - t;
}

// Gaussian quantile, accurate to ~1e-15 relative over (0, 1): the nearer tail
// probability min(p, 1 - p) is inverted, and the quantile is antisymmetric.
double gaussian_inv_cdf(double p, double mu, double sigma) {
  const double x = (p <= 0.5) ? normal_quantile_lower(p) : -normal_quantile_lower(1.0 - p);
  return mu + sigma * x;
}

// Score proposal: a reflected random walk in the CDF space of N(mu, sigma^2).
// The prior cancels only if CDF and quantile are exact inverses, so the CDF
// value p is never formed near 1.  It is carried as the tail probability
// t = Phi(-|z|) and a side, using that reflection into [0, 1] commutes with
// p -> 1 - p: from the upper side, p' = 1 - reflect(t - step).  Both tails then
// keep full relative precision (a raw p saturates to 1 beyond ~8.3 sigma, which
// made distinct scores map to the same point).  Same random draw as
// uniform_proposal.
double gaussian_proposal(double a, double mu, double sigma, double move_sigma) {
  const double z = (a - mu) / sigma;
  const bool upper = z > 0.0;
  const double t = normal_lower_tail(z);
  const double step = next_normal(rng, move_sigma);
  // x = the new CDF value seen from the current side (lower: p', upper: 1 - p')
  // (x = 0 needs the step to cancel t exactly; it is clamped so that the score
  // stays finite.)
  const double x = std::max(reflect_unit(t + (upper ? -step : step)),
                            std::numeric_limits<double>::min());
  // Map back from whichever tail x lies in: x <= 0.5 -> same side, else the other.
  const double z_new_side = (x <= 0.5) ? normal_quantile_lower(x) : -normal_quantile_lower(1.0 - x);
  const double z_new = upper ? -z_new_side : z_new_side;
  return mu + sigma * z_new;
}

// Value consistency check: compare the recomputed (actual) and the cached
// (bookkeeping) value; with throw_error, fail when
//     |actual - bookkeeping| > tolerance + NUMERICAL_CHECK_REL_TOL * |actual|
// or either is NaN.  The relative part covers the round-off of incremental
// updates on large totals and, unlike the former tolerance * actual, also
// applies to negative values (H_W, H_Adir).  Returns actual (repairs the cache).
double value_check(double actual, double bookkeeping, double tolerance, bool throw_error, const std::string& name) {
  const double diff = std::abs(actual - bookkeeping);
  if (throw_error && (std::isnan(actual) || std::isnan(bookkeeping) ||
                      (actual != bookkeeping &&
                       !(diff <= tolerance + NUMERICAL_CHECK_REL_TOL * std::abs(actual))))) {
    throw std::runtime_error("Bookkeeping mismatch for " + name +
        ": cached=" + std::to_string(bookkeeping) +
        ", actual=" + std::to_string(actual));
  }
  return actual;
}
