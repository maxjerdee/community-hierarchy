#ifndef FIT_RESULT_H
#define FIT_RESULT_H

#include <vector>
#include <limits>

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

    int num_nodes() const { return num_nodes_; }
    int num_samples() const { return static_cast<int>(samples_.size()); }
    const std::vector<SampleState>& samples() const { return samples_; }

protected:
    int num_nodes_ = 0;
    std::vector<SampleState> samples_;

private:
    int best_index() const;
};

#endif // FIT_RESULT_H
