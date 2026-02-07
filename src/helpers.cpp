/* Implementations of helper functions */

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "globals.h"
#include "helpers.h"
#include "MCMC_core.h" // Include full definition of MCMC

// Logarithm of the binomial coefficient
double log_binom(double a, double b) {
  if (b == 0) {
    return 0;
  }
  return std::lgamma(a + 1) - std::lgamma(b + 1) - std::lgamma(a - b + 1);
}

// Logarithm of the factorial
double log_factorial(int n) { return std::lgamma(n + 1); }

// Logarithm of the double factorial
double log_factorial2(int n) {
  // Check that n is an even integer
  if (n % 2 != 0) {
    std::cout << "Error: log_factorial2 called with odd integer.\n";
    exit(0);
  }
  return n / 2 * std::log(2) + std::lgamma(n / 2 + 1);
}

// Logarithm of the negative binomial distribution of x with a given mean and concentration parameter, 
// P(x|mean,concentration) = binom(x+concentration-1, x) * concentration^concentration mean^x/(concentration + mean)^(concentration + x)
// concentration = 1 is a geometric distribution, concentration = inf is a Poisson distribution
double log_neg_binom(int x, double mean, double concentration) {
  if (x < 0) {
    return -std::numeric_limits<double>::infinity();
  }
  double concentration_used = concentration;
  if (concentration < CONCENTRATION_EPSILON){
    concentration_used = CONCENTRATION_EPSILON; // Avoid zero concentration
  }
  if (concentration_used > 1/CONCENTRATION_EPSILON){
    // For very large concentration, approximate with Poisson
    return -mean + x * std::log(mean) - log_factorial(x);
  }
  return log_binom(x + concentration_used - 1, x) + concentration_used * std::log(concentration_used) +
         x * std::log(mean) - (concentration_used + x) * std::log(concentration_used + mean);
}

// Overflow protected version of log(e^a + e^b)
double log_sum_exp(double a, double b) {
  double a_b_max = std::max(a, b);
  return a_b_max + std::log(std::exp(a - a_b_max) + std::exp(b - a_b_max));
}

// Overflow protected version of log(\sum exp(x_i))
double log_sum_exp(std::vector<double> xs) {
  if (xs.size() == 0) {
    return -std::numeric_limits<double>::infinity();
  }
  double xs_max = *std::max_element(xs.begin(), xs.end());
  double result = 0;
  for (double x : xs) {
    result += std::exp(x - xs_max);
  }
  return xs_max + std::log(result);
}

double inverse_erf(double x) {
    if (x < -1 || x > 1) {
        std::cerr << "Error: inverse_erf called with argument outside [-1, 1].\n";
        exit(1);
    }
  // Reference: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c

  const double absx = std::abs(x);
  if (absx == 1.0) {
    return std::copysign(std::numeric_limits<double>::infinity(), x);
  }

  // Asymptotic correction near |x| -> 1 to enforce divergence
  if (absx > 0.999) {
    const double eps = 1.0 - absx;
    const double log_pi = std::log(std::acos(-1.0));
    const double L = -std::log(eps) - 0.5 * log_pi;
    const double z = std::sqrt(L - 0.5 * std::log(L));
    return std::copysign(z, x);
  }

  float p, r, t;
  t = fmaf (x, 0.0f - x, 1.0f);
  t = std::log(t);
  if (fabsf(t) > 6.125f) { // maximum ulp error = 2.35793
    p =              3.03697567e-10f; //  0x1.4deb44p-32 
    p = fmaf (p, t,  2.93243101e-8f); //  0x1.f7c9aep-26 
    p = fmaf (p, t,  1.22150334e-6f); //  0x1.47e512p-20 
    p = fmaf (p, t,  2.84108955e-5f); //  0x1.dca7dep-16 
    p = fmaf (p, t,  3.93552968e-4f); //  0x1.9cab92p-12 
    p = fmaf (p, t,  3.02698812e-3f); //  0x1.8cc0dep-9 
    p = fmaf (p, t,  4.83185798e-3f); //  0x1.3ca920p-8 
    p = fmaf (p, t, -2.64646143e-1f); // -0x1.0eff66p-2 
    p = fmaf (p, t,  8.40016484e-1f); //  0x1.ae16a4p-1 
  } else { // maximum ulp error = 2.35002
    p =              5.43877832e-9f;  //  0x1.75c000p-28 
    p = fmaf (p, t,  1.43285448e-7f); //  0x1.33b402p-23 
    p = fmaf (p, t,  1.22774793e-6f); //  0x1.499232p-20 
    p = fmaf (p, t,  1.12963626e-7f); //  0x1.e52cd2p-24 
    p = fmaf (p, t, -5.61530760e-5f); // -0x1.d70bd0p-15 
    p = fmaf (p, t, -1.47697632e-4f); // -0x1.35be90p-13 
    p = fmaf (p, t,  2.31468678e-3f); //  0x1.2f6400p-9 
    p = fmaf (p, t,  1.15392581e-2f); //  0x1.7a1e50p-7 
    p = fmaf (p, t, -2.32015476e-1f); // -0x1.db2aeep-3 
    p = fmaf (p, t,  8.86226892e-1f); //  0x1.c5bf88p-1 
  }
  r = x * p;
  return r;
}


double concentration_to_variation(double concentration) {
  if (concentration < 0.0) {
    std::cerr << "Error: Concentration parameter cannot be negative.\n";
    exit(1);
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
    std::cerr << "Error: Variation parameter must be in [0, 1].\n";
    exit(1);
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

  if (f_lo * f_hi > 0) {
    // fallback: return a reasonable default
    // In Python this recursively calls itself which would cause infinite loop,
    // so just return mid-range value
    return 1.0;
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

// Sample from a set of outcomes given weights (not necessarily normalized),
// also return the entropy of that choice
std::pair<int, double> sample_weights(std::vector<double> weights) {
  // Sample according to these weights (this is where all the randomness comes
  // from), use the global rng mersenne twister
  // Normalize weights
  double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  for (double &weight : weights) {
    weight /= sum;
  }

  double p = uniform_random(rng);

  double culm_weight = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) {
    culm_weight += weights[i];
    if (p <= culm_weight) {
      return std::make_pair(static_cast<int>(i), -std::log(weights[i]));
    }
  }
  // If no match found, return the last element as a fallback (due to rounding
  // issues)
  return std::make_pair(weights.size() - 1, -std::log(weights.back()));
}

// Sample from a set of log weights (not necessarily normalized),
// also return the entropy of that choice
std::pair<int, double> sample_log_weights(std::vector<double> log_weights) {
  // Normalize the weights
  double log_sum = log_sum_exp(log_weights);
  std::vector<double> weights;
  weights.reserve(log_weights.size()); // Reserve memory for weights
  for (double log_weight : log_weights) {
    weights.push_back(std::exp(log_weight - log_sum));
  }
  
  return sample_weights(weights);
}

// Template implementation for sample_log_weights_map
template <typename T>
std::pair<T, double> sample_log_weights_map(const std::unordered_map<T, double>& log_weights_map) {
    std::vector<double> log_weights;
    std::vector<T> keys;
    for (const auto& [key, log_weight] : log_weights_map) {
        log_weights.push_back(log_weight);
        keys.push_back(key);
    }
    auto [index, entropy] = sample_log_weights(log_weights);
    return {keys[index], entropy};
}

// Explicit instantiation for int keys
template std::pair<int, double> sample_log_weights_map<int>(const std::unordered_map<int, double>& log_weights_map);

double positive_uniform_proposal(double a, double move_sigma) {
  double a_new = a + std::normal_distribution<double>(0, move_sigma)(rng);
  // Impose reflective boundary conditions p_new \in [0,inf)
  a_new = fabs(a_new);
  return a_new;
}

// Transformations of parameters to the [0,1] interval according to a given prior distribution to simplify the sampling process
// In order to obtain a local move of a that respects the prior distribution P(a),
// We compute p = cdf(a), apply a small uniform Gaussian perturbation to p, 
double uniform_proposal(double p, double move_sigma) {
  double p_new = p + std::normal_distribution<double>(0, move_sigma)(rng);
  // Impose reflective boundary conditions p_new \in [0,1]
  p_new = fmod(p_new + 2.0, 2.0);
  if(p_new > 1.0){
      p_new = 2.0 - p_new;
  }
  return p_new;
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
// Cauchy prior with width w and mean mu: P(a|mu,w) = 1 / (pi w (1 + ((a - mu) / w)^2))
double cauchy_cdf(double a, double mu, double w) {
  return 0.5 + std::atan((a - mu) / w) / M_PI;
}
double cauchy_inv_cdf(double p, double mu, double w) {
  return mu + w * std::tan(M_PI * (p - 0.5));
}
double cauchy_proposal(double a, double mu, double w, double move_sigma) {
  double p = cauchy_cdf(a, mu, w);
  double p_new = uniform_proposal(p, move_sigma);
  return cauchy_inv_cdf(p_new, mu, w);
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

// Gaussian prior with mean mu and width sigma: P(a|mu,sigma) = (1/sqrt(2*pi*sigma^2)) exp(-(a-mu)^2/(2*sigma^2))
double gaussian_cdf(double a, double mu, double sigma) {
  // CDF of Gaussian: 0.5 * (1 + erf((a - mu) / (sigma * sqrt(2))))
  return 0.5 * (1.0 + std::erf((a - mu) / (sigma * std::sqrt(2.0))));
}

double gaussian_inv_cdf(double p, double mu, double sigma) {
  // Inverse CDF using inverse error function
  if(verbose){
    print("gaussian_inv_cdf called with p=" + std::to_string(p) + ", mu=" + std::to_string(mu) + ", sigma=" + std::to_string(sigma));
    print("inverse_erf argument: " + std::to_string(2.0 * p - 1.0));
    print("inverse_erf value: " + std::to_string(inverse_erf(2.0 * p - 1.0)));
    print("Result: " + std::to_string(mu + sigma * std::sqrt(2.0) * inverse_erf(2.0 * p - 1.0)));
  }
  return mu + sigma * std::sqrt(2.0) * inverse_erf(2.0 * p - 1.0);
}

double gaussian_proposal(double a, double mu, double sigma, double move_sigma) {
  if(verbose){
    print("gaussian_proposal called with a=" + std::to_string(a) + ", mu=" + std::to_string(mu) + ", sigma=" + std::to_string(sigma) + ", move_sigma=" + std::to_string(move_sigma));
  }
  double p = gaussian_cdf(a, mu, sigma);
  double p_new = uniform_proposal(p, move_sigma);
  if(verbose){
    print("gaussian_proposal intermediate p=" + std::to_string(p) + ", p_new=" + std::to_string(p_new));
  }
  double result = gaussian_inv_cdf(p_new, mu, sigma);
  if(result < -1e10 || result > 1e10){
    std::cerr << "Warning: gaussian_proposal result is very large: " << result << ". Check if the parameters are set correctly.\n";
    std::cerr << "Input parameters were: a=" << a << ", mu=" << mu << ", sigma=" << sigma << ", move_sigma=" << move_sigma << "\n";
    std::cerr << "Intermediate values were: p=" << p << ", p_new=" << p_new << "\n";
  }
  return result;
}



// Compute the log(mean) and standard error of that assessment given log values
std::pair<double, double> log_mean_and_se(std::vector<double> log_vals){
  // Compute the mean of the log values
  double log_E_val = log_sum_exp(log_vals) - std::log(log_vals.size()); // Log of mean of values
  std::vector<double> twice_log_vals;
  for (double log_val : log_vals) { twice_log_vals.push_back(2 * log_val); } // 2 * log_vals
  double log_E_val_2 = log_sum_exp(twice_log_vals) - std::log(log_vals.size()); // Log of mean of squared values
  double log_std = 0.5 * (std::log(std::exp(0) - std::exp(2 * log_E_val - log_E_val_2)) + log_E_val_2); // Log of standard deviation of values
  double log_E_val_err_est = std::exp(log_std - 0.5 * std::log(log_vals.size()) - log_E_val); // Log of standard error of mean of values

  return std::make_pair(log_E_val, log_E_val_err_est); // Return the log of the mean and standard error of the mean of the values
}

// Value consistency check: compare actual vs bookkeeping value, optionally throw error
double value_check(double actual, double bookkeeping, double tolerance, bool throw_error, const std::string& name) {
  if (throw_error && ((std::abs(actual - bookkeeping) > tolerance && std::abs(actual - bookkeeping) > tolerance*actual)|| std::isnan(actual) || std::isnan(bookkeeping))) {
    std::cerr << "Error: " << name << " does not match the actual value.\n";
    std::cerr << name << ": " << bookkeeping << ", " << name << "_actual: " << actual << "\n";
    exit(1);
  }
  return actual;
}

// Sample from a bernoulli distribution with given probability
std::pair<int, double> sample_bernoulli(double p) {
  double r = uniform_random(rng);
  if (r < p) {
    return std::make_pair(1, -std::log(p));
  } else {
    return std::make_pair(0, -std::log(1 - p));
  }
}

// Return the indices perm which label the row sums from largest to smallest
template <typename T> std::vector<int> argsort(const std::vector<T> &array) {
  std::vector<int> indices(array.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&array](int left, int right) -> bool {
              // sort indices according to corresponding array element
              return array[left] > array[right];
            });

  return indices;
}

// Explicit instantiation of template function for int type
template std::vector<int> argsort<int>(const std::vector<int> &array);

// Printing functions (useful for debugging)
// Print function which calls print on all arguments
// template <typename T, typename... Args>
// void print(T t, Args... args){
//     std::cout << t << " ";
//     print(args...);
// }

void print(const MCMC& state) { 
  std::cout << state.to_string_verbose() << std::endl; 
}

// Fix format specifier in printf
void print_node(size_t i) {
  printf("Node %zu\n", i); // Use %zu for size_t
}

void print(std::vector<std::vector<int>> table) {
  for (size_t i = 0; i < table.size(); i++) {
    for (size_t j = 0; j < table[0].size(); j++) {
      std::cout << table[i][j] << ' ';
    }
    std::cout << "\n";
  }
}
void print(std::vector<std::vector<double>> table) {
  for (size_t i = 0; i < table.size(); i++) {
    for (size_t j = 0; j < table[0].size(); j++) {
      std::cout << table[i][j] << ' ';
    }
    std::cout << "\n";
  }
}
void print(std::vector<int> vec) {
  for (size_t n = 0; n < vec.size(); ++n)
    std::cout << vec[n] << ' ';
  std::cout << '\n';
}
void print(std::vector<double> vec) {
  for (size_t n = 0; n < vec.size(); ++n)
    std::cout << vec[n] << ' ';
  std::cout << '\n';
}
void print(int val) { std::cout << val << std::endl; }
void print(double val) { std::cout << val << std::endl; }
void print(std::string val) { std::cout << val << std::endl; }
