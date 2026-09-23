#include "globals.h"

int n = 0;
int m = 0;
std::vector<std::unordered_set<int>> neighbors;

bool verbose = false;
bool debug = false;

bool timing = true;
thread_local std::chrono::high_resolution_clock::time_point start_time, end_time;
thread_local double duration = 0.0;

// Default-constructed: every MCMC constructor seeds it (MCMC_core.cpp).
thread_local std::mt19937 rng;

int MERGE_SPLIT_GIBBS_SWEEPS = 10;
int MERGE_SPLIT_PRESTAGE_MODE = 0;
int MERGE_SPLIT_MOVES = 7;  // merge | split | joint
double MERGE_SPLIT_WEIGHT = -1.0;  // <0 = linear-time default, see globals.h
