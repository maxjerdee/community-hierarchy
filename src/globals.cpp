#include "globals.h"

bool verbose = false;

bool timing = false;  // profiling only: see globals.h
thread_local std::chrono::high_resolution_clock::time_point start_time, end_time;
thread_local double duration = 0.0;

// Default-constructed: every MCMC constructor seeds it (MCMC_core.cpp).
thread_local std::mt19937 rng;

int MERGE_SPLIT_GIBBS_SWEEPS = 3;
int MERGE_SPLIT_PRESTAGE_MODE = 0;
int MERGE_SPLIT_MOVES = 7;  // merge | split | joint
double INDIVIDUAL_DEPTH_CENTRED_PROB = 0.5;   // see globals.h
int INDIVIDUAL_DEPTH_SCALE_MOVE = 1;           // see globals.h (default: residual rescale)
double MERGE_SPLIT_WEIGHT = 1.0;   // Peixoto's omega_collective = 1; <0 = equal-spend retuning by fit(), see globals.h
double MERGE_SPLIT_EQUAL_SPEND_C = 9.0;
