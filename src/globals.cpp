#include "globals.h"

int n = 0;
int m = 0;
std::vector<std::unordered_set<int>> neighbors;

bool verbose = false;
bool debug = false;

bool timing = true;
std::chrono::high_resolution_clock::time_point start_time, end_time;
double duration = 0.0;

std::random_device rd;
std::mt19937 rng(rd());
std::uniform_real_distribution<double> uniform_random(0.0, 1.0);