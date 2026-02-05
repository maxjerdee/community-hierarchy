/* Helper function declarations */

#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <vector>

class MCMC;

/* Math */
// Logarithm of the binomial coefficient
double log_binom(double a, double b);

// Logarithm of the factorial
double log_factorial(int n);

// Logarithm of the double factorial
double log_factorial2(int n);

// Logarithm of the negative binomial distribution of x with given mean and concentration parameter,
double log_neg_binom(int x, double mean, double concentration);

// Overflow protected version of log(e^a + e^b) for precomputing log_q_vals
double log_sum_exp(double a, double b);
double log_sum_exp(std::vector<double> xs);

// Compute the log(mean) and standard error of that assessment given log values
std::pair<double, double> log_mean_and_se(std::vector<double> log_vals);

// Inverse erf
double inverse_erf(double x);

// Value consistency check: compare actual vs bookkeeping value, optionally throw error
double value_check(double actual, double bookkeeping, double tolerance, bool throw_error, const std::string& name);

// Convert between variation (Gini) and concentration (Dirichlet) parameters
double concentration_to_variation(double concentration);
double variation_to_concentration(double variation, double tol = 1e-8, int max_iter = 100);

/* Generate proposals for 1 dimensional variables respecting given priors */
// Uniform distribution between 0 and 1
double uniform_proposal(double p, double move_sigma);

// Half-Cauchy prior of width w: P(a|w) = 2 w / (pi (w^2 + a^2))
double half_cauchy_cdf(double a, double w);
double half_cauchy_inv_cdf(double p, double w);
double half_cauchy_proposal(double a, double w, double move_sigma);

// Cauchy prior with width w and mean mu: P(a|mu,w) = 1 / (pi w (1 + ((a - mu) / w)^2))
double cauchy_cdf(double a, double mu, double width);
double cauchy_inv_cdf(double p, double mu, double width);
double cauchy_proposal(double a, double mu, double width, double move_sigma);

// Exponential prior with mean mu: P(a|mu) = (1/mu) exp(-a/mu) for a >= 0
double exponential_cdf(double a, double mu);
double exponential_inv_cdf(double p, double mu);
double exponential_proposal(double a, double mu, double move_sigma);

// Gaussian prior with mean mu and width sigma: P(a|mu,sigma) = (1/sqrt(2*pi*sigma^2)) exp(-(a-mu)^2/(2*sigma^2))
double gaussian_cdf(double a, double mu, double sigma);
double gaussian_inv_cdf(double p, double mu, double sigma);
double gaussian_proposal(double a, double mu, double sigma, double move_sigma);

// Positive uniform proposal
double positive_uniform_proposal(double a, double move_sigma);

/* Sampling */
// Sample from a set of outcomes given (log) weights (not necessarily normalized),
// also return the entropy of that choice
std::pair<int, double> sample_weights(std::vector<double> weights);
std::pair<int, double> sample_log_weights(std::vector<double> log_weights);

// Sample from unordered_map with template keys and log-weight values
// Returns pair of (key, entropy)
template <typename T>
std::pair<T, double> sample_log_weights_map(const std::unordered_map<T, double>& log_weights_map);

// Sample from a Bernoulli distribution with given probability
std::pair<int, double> sample_bernoulli(double p);

/* Sorting */
// Return the index permutation which labels the vector from largest to smallest
template <typename T> std::vector<int> argsort(const std::vector<T> &array);

/* Printing functions */
void print(const MCMC& state);
void print(std::vector<std::vector<int>> table);
void print(std::vector<std::vector<double>> table);
void print(std::vector<int> vec);
void print(std::vector<double> vec);
void print(int val);
void print(double val);
void print(std::string val);

#endif // HELPERS_H
