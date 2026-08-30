#ifndef FIT_RESULT_H
#define FIT_RESULT_H

#include <vector>
#include <limits>
#include <map>
#include <string>
#include <utility>

struct SampleState {
    int chain_id;
    double beta;
    std::vector<int> partition;
    std::vector<double> scores;
    int num_groups;
    double density_in, density_out;
    double variation_in, variation_out;
    double degree_correction;
    double individual_depth, group_depth;
    double ties_parameter;
    double H_A_b;
    double H_b, H_M, H_k, H_A_G_k_M_b, H_A_G_b;
    double H_Adir_G_A_s_nu;
    double H_s_G_b_depths;
    bool two_layer_mode = false;
    bool weighted_SBM = false;
    double H_W = 0.0;
};

// Return type for network_properties(): one row per parameter.
struct NetworkPropertiesResult {
    std::vector<std::string> parameter;
    std::vector<double> mean;
    std::vector<double> std;
    std::vector<double> hdi_lo;
    std::vector<double> hdi_hi;
};

// Return type for node_properties(): one row per node (integer-indexed).
struct NodePropertiesResult {
    std::string mode_used;          // "consensus" or "mdl"
    std::vector<double> score_mean;
    std::vector<double> score_std;
    std::vector<int>    group;
    std::vector<double> group_score_mean;
};

class FitResult {
public:
    FitResult() = default;
    FitResult(int num_nodes, std::vector<SampleState> samples);

    double mdl_value() const;
    std::vector<int> mdl_partition() const;
    std::vector<int> consensus_partition() const;
    std::vector<double> coincidence_matrix() const;
    std::vector<double> score_means() const;
    std::vector<double> score_stds() const;
    std::vector<double> direction_probabilities(int node_a, int node_b, bool ties_allowed = true) const;
    std::map<std::string, double> parameter_means() const;
    std::map<std::string, double> parameter_stds() const;

    // New posterior-summary methods (C++ core; language wrappers convert to DataFrames).
    NetworkPropertiesResult network_properties(double prob = 0.90) const;
    NodePropertiesResult    node_properties(const std::string& mode = "auto") const;

    int num_nodes() const { return num_nodes_; }
    int num_samples() const { return static_cast<int>(samples_.size()); }
    const std::vector<SampleState>& samples() const { return samples_; }

protected:
    int num_nodes_ = 0;
    std::vector<SampleState> samples_;

private:
    int best_index() const;
    static std::pair<double, double> hdi_bounds(std::vector<double> arr, double prob);
};

#endif // FIT_RESULT_H
