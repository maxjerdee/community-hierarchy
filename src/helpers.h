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
// log Gamma(x + 1): log x! for integer x, its continuous extension for real x
double log_factorial_real(double x);

// Logarithm of the double factorial
double log_factorial2(int n);

// Logarithm of the negative binomial distribution of x with given mean and concentration parameter,
double log_neg_binom(int x, double mean, double concentration);

// Overflow protected version of log(e^a + e^b) for precomputing log_q_vals
double log_sum_exp(double a, double b);
double log_sum_exp(std::vector<double> xs);

// Value consistency check: with throw_error, throw if
// |actual - bookkeeping| > tolerance + NUMERICAL_CHECK_REL_TOL * |actual| (or NaN);
// returns actual
double value_check(double actual, double bookkeeping, double tolerance, bool throw_error, const std::string& name);

// Convert between variation and concentration parameters.  The "variation" of a
// concentration (shape) c is the Gini coefficient of a Gamma(c) variable,
//     G(c) = Gamma(c + 1/2) / (sqrt(pi) Gamma(c + 1)),
// which falls from 1 (c -> 0) through 1/2 (c = 1) to 0 (c -> infinity).  It is used
// for the negative-binomial edge counts (H_M) and the Dirichlet-multinomial
// degrees (H_k), so that both overdispersions live on [0, 1].
double concentration_to_variation(double concentration);
double variation_to_concentration(double variation, double tol = 1e-8, int max_iter = 100);

/* Generate proposals for 1 dimensional variables respecting given priors */
// Uniform distribution between 0 and 1
double uniform_proposal(double p, double move_sigma);

// Half-Cauchy prior of width w: P(a|w) = 2 w / (pi (w^2 + a^2))
double half_cauchy_cdf(double a, double w);
double half_cauchy_inv_cdf(double p, double w);
double half_cauchy_proposal(double a, double w, double move_sigma);

// Exponential prior with mean mu: P(a|mu) = (1/mu) exp(-a/mu) for a >= 0
double exponential_cdf(double a, double mu);
double exponential_inv_cdf(double p, double mu);
double exponential_proposal(double a, double mu, double move_sigma);

// Gaussian N(mu, sigma^2): CDF and quantile accurate to ~1e-15 relative in both
// tails (erfc; Acklam's approximation plus one Halley step).
double gaussian_cdf(double a, double mu, double sigma);
double gaussian_inv_cdf(double p, double mu, double sigma);
// CDF-space random walk for N(mu, sigma^2) (the score move).  Its Hastings ratio
// assumes an exact CDF/quantile pair; the CDF value is carried through its
// nearer tail so that both tails keep full precision.  See helpers.cpp.
double gaussian_proposal(double a, double mu, double sigma, double move_sigma);

/* Printing functions */
void print(std::vector<std::vector<int>> table);
void print(std::vector<std::vector<double>> table);
void print(std::vector<int> vec);
void print(std::vector<double> vec);
void print(int val);
void print(double val);
void print(std::string val);

#endif // HELPERS_H
