#include "MCMC_core.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <regex>
#include <cmath>
#include <map>

// Simple GML parser for directed graphs with edge types
// Returns a pair: first = wins (dominant edges), second = ties (neutral edges)
std::pair<std::vector<std::unordered_map<int, double>>, std::vector<std::unordered_map<int, double>>> read_gml(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<std::unordered_map<int, double>> wins;
    std::vector<std::unordered_map<int, double>> ties;
    std::string line;
    int max_node_id = -1;
    
    // First pass: find maximum node id
    while (std::getline(file, line)) {
        std::smatch match;
        if (std::regex_search(line, match, std::regex("id\\s+(\\d+)"))) {
            int node_id = std::stoi(match[1]);
            if (node_id > max_node_id) {
                max_node_id = node_id;
            }
        }
    }
    
    // Initialize neighbors vectors
    wins.resize(max_node_id + 1);
    ties.resize(max_node_id + 1);
    
    // Second pass: read edges
    file.clear();
    file.seekg(0);
    bool in_edge = false;
    int source = -1, target = -1;
    std::string edge_type = "";
    
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        
        if (line.find("edge") != std::string::npos) {
            in_edge = true;
            source = -1;
            target = -1;
            edge_type = "";
        } else if (in_edge && line.find("source") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("source\\s+(\\d+)"))) {
                source = std::stoi(match[1]);
            }
        } else if (in_edge && line.find("target") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("target\\s+(\\d+)"))) {
                target = std::stoi(match[1]);
            }
        } else if (in_edge && line.find("type") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("type\\s+\"(\\w+)\""))) {
                edge_type = match[1];
            }
        } else if (in_edge && line == "]") {
            if (source != -1 && target != -1) {
                // Add edge based on type
                if (edge_type == "dominant") {
                    wins[source][target] = 1.0;
                } else if (edge_type == "neutral") {
                    ties[source][target] = 1.0;
                    ties[target][source] = 1.0; // Undirected
                }
            }
            in_edge = false;
        }
    }
    
    file.close();
    return std::make_pair(wins, ties);
}

int main() {
    // Read real network from GML file
    // std::vector<std::unordered_map<int, double>> neighbors = read_gml("examples/data/dolphins.gml");
    // auto [wins, ties] = read_gml("examples/data/friends.gml");
    auto [wins, ties] = read_gml("examples/data/highschool_51.gml");
    // auto [wins, ties] = read_gml("examples/data/highschool_30.gml");
    // auto [wins, ties] = read_gml("examples/data/highschool_84.gml");
    // auto [wins, ties] = read_gml("examples/data/test_2.gml");

    // Initialize partition with random groups
    std::vector<int> partition(wins.size());
    for (std::size_t i = 0; i < wins.size(); i++) {
        partition[i] = rand() % 4;  // Randomly assign to group 0 or 1
    }
    

    // // Simple 3-node test network
    // std::vector<std::unordered_map<int, double>> out_neighbors(3);
    // out_neighbors[0][1] = 1.0;
    // // out_neighbors[1][0] = 1.0;
    // out_neighbors[1][2] = 1.0;
    // // out_neighbors[2][1] = 1.0;

    // std::vector<int> partition = {0, 0, 1};


    // // Simple 2-node test network
    // std::vector<std::unordered_map<int, double>> wins(2);
    // std::vector<std::unordered_map<int, double>> ties(2);
    // wins[0][1] = 1.0;
    // ties[0][1] = 1.0;
    // ties[1][0] = 1.0;

    // std::vector<int> partition = {0, 1};


    std::cout << "Loaded graph with " << wins.size() << " nodes" << std::endl;

    // Print wins and ties
    for (std::size_t i = 0; i < wins.size(); i++){
        for (const auto& [j, weight] : wins[i]){
            std::cout << "Win: " << i << " -> " << j << " (weight " << weight << ")" << std::endl;
        }
        for (const auto& [j, weight] : ties[i]){
            std::cout << "Tie: " << i << " <-> " << j << " (weight " << weight << ")" << std::endl;
        }
    }

    // // Full general model
    // MCMC m(neighbors,
    //     partition,
    //     2,              // num_groups (passed as value)
    //     false,          // num_groups_fixed
    //     true,           // assortative
    //     0.5,            // variation_in (passed as value)
    //     false,          // variation_in_fixed
    //     0.5,            // variation_out (passed as value)
    //     false,          // variation_out_fixed
    //     0.5,            // degree_correction (passed as value)
    //     false,          // degree_correction_fixed
    //     -0.5,            // mean_degree_scaling (passed as value)
    //     false,          // mean_degree_scaling_fixed
    //     46);            // seed

    // Canonical typical DCSBM
    MCMC m(wins,
        ties,
        partition,
        2,              // num_groups (passed as value)
        false,          // num_groups_fixed
        false,           // assortative
        0.5,            // variation_in (passed as value)
        true,          // variation_in_fixed
        0.5,            // variation_out (passed as value)
        true,          // variation_out_fixed
        0.5,            // degree_correction (passed as value)
        true,          // degree_correction_fixed
        0.0,            // mean_degree_scaling (passed as value)
        true,          // mean_degree_scaling_fixed
        1.0,            // individual_depth (passed as value)
        false,           // individual_depth_fixed
        0.0,            // group_depth (passed as value)
        false,           // group_depth_fixed
        1.0,            // ties_parameter (passed as value)
        false,          // ties_parameter_fixed
        42);            // seed

    /*
    Configuration from python:
    Model configuration: assortative=False, variation_in=0.5, variation_out=0.5, degree_correction=0.5, mean_degree_scaling=0, individual_depth = None, group_depth = None, ties_parameter = None, num_groups=None
    
    */
    
    int num_samples = 100000;
    int num_samples_used = num_samples / 2;
    
    // Track statistics
    std::unordered_map<int, int> num_groups_counts;
    double mean_degree_scaling_sum = 0.0;
    double degree_correction_sum = 0.0;
    double variation_in_sum = 0.0;
    double variation_out_sum = 0.0;
    double ties_parameter_sum = 0.0;
    std::unordered_map<std::string, int> partition_counts;
    
    // Histograms: bins of 0.1 width
    std::map<int, int> mean_degree_scaling_hist;  // range -1.0 to 0.0 (bins -10 to 0)
    std::map<int, int> degree_correction_hist;     // range 0.0 to 1.0 (bins 0 to 10)
    std::map<int, int> variation_in_hist;          // range 0.0 to 1.0 (bins 0 to 10)
    std::map<int, int> variation_out_hist;         // range 0.0 to 1.0 (bins 0 to 10)
    std::map<int, int> ties_parameter_hist;        // range 0.0 to 10.0 (bins 0 to 100)
    // Depth histograms: bins of width 0.5 over [0,10]
    // Bin indices 0..19 correspond to [0,0.5),[0.5,1.0),...,[9.5,10.0]
    std::map<int, int> individual_depth_hist;      // bins 0 to 19
    std::map<int, int> group_depth_hist;           // bins 0 to 19
    // Score histogram: bins of width 0.1 over [-3,3]
    // Bin indices 0..59 correspond to [-3,-2.9),...,[2.9,3.0]
    std::map<int, int> first_score_hist;           // bins 0 to 59
    
    for(int i = 0; i < num_samples; i++){
        double wang_landau_modification_factor = 0.1 / (1 + i/1000.0); // Decrease modification factor over time
        printf("MCMC Sweep %d, wang_landau_modification_factor: %f\n", i + 1, wang_landau_modification_factor);
        // m.MCMC_sweep(0.0, true); // Testing merge-split
        // m.MCMC_sweep(0.0, false); // Testing single site
        // m.MCMC_sweep(1.0, false, wang_landau_modification_factor, 15); // Testing WL algorithm
        m.MCMC_sweep(1.0, false);
        
        // Only count samples from the second half towards histograms and statistics
        if(i >= num_samples - num_samples_used) {
            // Track the number of groups
            num_groups_counts[m.num_groups]++;
            
            // Track parameter values
            mean_degree_scaling_sum += m.mean_degree_scaling;
            degree_correction_sum += m.degree_correction;
            variation_in_sum += m.variation_in;
            variation_out_sum += m.variation_out;
            ties_parameter_sum += m.ties_parameter;
            
            // Convert partition to string and count frequency
            std::string partition_str = "";
            for(int j = 0; j < m.partition.size(); j++) {
                partition_str += std::to_string(m.partition[j]);
                if(j < m.partition.size() - 1) partition_str += ",";
            }
            partition_counts[partition_str]++;
            
            // Histogram binning
            int mds_bin = std::max(-10, std::min(0, static_cast<int>(std::floor(m.mean_degree_scaling * 10))));
            int dc_bin = std::max(0, std::min(10, static_cast<int>(std::floor(m.degree_correction * 10))));
            int vi_bin = std::max(0, std::min(10, static_cast<int>(std::floor(m.variation_in * 10))));
            int vo_bin = std::max(0, std::min(10, static_cast<int>(std::floor(m.variation_out * 10))));
            int tp_bin = std::max(0, std::min(99, static_cast<int>(std::floor(m.ties_parameter * 10))));
            mean_degree_scaling_hist[mds_bin]++;
            degree_correction_hist[dc_bin]++;
            variation_in_hist[vi_bin]++;
            variation_out_hist[vo_bin]++;
            ties_parameter_hist[tp_bin]++;

            // Depth histogram binning (0 to 10, bin size 0.5)
            int id_bin = std::max(0, std::min(19, static_cast<int>(std::floor(m.individual_depth * 2.0))));
            int gd_bin = std::max(0, std::min(19, static_cast<int>(std::floor(m.group_depth * 2.0))));
            individual_depth_hist[id_bin]++;
            group_depth_hist[gd_bin]++;

            // First score histogram binning (-3 to 3, bin size 0.1)
            int score_bin = std::max(0, std::min(59, static_cast<int>(std::floor((m.scores[0] + 3.0) * 10.0))));
            first_score_hist[score_bin]++;
        }
        
        // Print current partition
        std::cout << "Current partition: ";
        for(const auto& group : m.partition){
            std::cout << group << " ";
        }
        std::cout << std::endl;
        // Print the first 10 scores
        std::cout << "Current scores: ";
        for(int s = 0; s < std::min(10, static_cast<int>(m.scores.size())); s++){
            std::cout << m.scores[s] << " ";
        }
        std::cout << std::endl;
        // Print the current depths
        std::cout << "Current individual_depth: " << m.individual_depth << ", group_depth: " << m.group_depth << std::endl;
        // Print the current H_A_b, H_Adir_G_A_s, H_s_G_b_depths
        std::cout << "Current H_A_b: " << m.H_A_b << ", H_Adir_G_A_s_nu: " << m.H_Adir_G_A_s_nu << ", H_s_G_b_depths: " << m.H_s_G_b_depths << std::endl;
    }
    // Print the full final information (parameters, entropies, partition)
    std::cout << "\n=== Final MCMC State ===" << std::endl;
    std::cout << "Number of groups: " << m.num_groups << std::endl;
    std::cout << "density_in: " << m.density_in << std::endl;
    std::cout << "density_out: " << m.density_out << std::endl;
    std::cout << "variation_in: " << m.variation_in << std::endl;
    std::cout << "variation_out: " << m.variation_out << std::endl;
    std::cout << "degree_correction: " << m.degree_correction << std::endl;
    std::cout << "weight_in: " << m.weight_in << std::endl;
    std::cout << "weight_out: " << m.weight_out << std::endl;
    std::cout << "concentration_in: " << m.concentration_in << std::endl;
    std::cout << "concentration_out: " << m.concentration_out << std::endl;
    std::cout << "degree_correction_concentration: " << m.degree_correction_concentration << std::endl;
    std::cout << "mean_degree_scaling: " << m.mean_degree_scaling << std::endl;
    std::cout << "individual_depth: " << m.individual_depth << std::endl;
    std::cout << "group_depth: " << m.group_depth << std::endl;
    std::cout << "ties_parameter: " << m.ties_parameter << std::endl;
    std::cout << "H_A_b: " << m.H_A_b << std::endl;
    std::cout << "H_b: " << m.H_b << std::endl;
    std::cout << "H_M: " << m.H_M << std::endl;
    std::cout << "H_k: " << m.H_k << std::endl;
    std::cout << "H_A_G_k_M_b: " << m.H_A_G_k_M_b << std::endl;
    std::cout << "H_Adir_G_A_s_nu: " << m.H_Adir_G_A_s_nu << std::endl;
    std::cout << "H_s_G_b_depths: " << m.H_s_G_b_depths << std::endl;

    std::cout << "Final partition: ";
    for(const auto& group : m.partition){
        std::cout << group << " ";
    }
    std::cout << std::endl;

    std::cout << "Final scores: ";
    for(const auto& score : m.scores){
        std::cout << score << " ";
    }
    std::cout << std::endl;
    
    // Print timing summary
    std::cout << "\n=== Timing Summary ===" << std::endl;
    std::cout << "Time spent in H_k computation: " << m.time_H_k << " seconds" << std::endl;
    std::cout << "Time spent in H_M computation: " << m.time_H_M << " seconds" << std::endl;
    std::cout << "Time spent in H_A_G_k_M_b computation: " << m.time_H_A_G_k_M_b << " seconds" << std::endl;
    std::cout << "Time spent in single-site proposals: " << m.time_single_site_proposals << " seconds" << std::endl;
    std::cout << "Time spent in merge proposals: " << m.time_merge_proposals << " seconds" << std::endl;
    std::cout << "Time spent in split proposals: " << m.time_split_proposals << " seconds" << std::endl;
    std::cout << "Time spent in parameter proposals: " << m.time_parameter_proposals << " seconds" << std::endl;

    // Print summary of number of groups distribution
    std::cout << "\n=== Number of Groups Distribution ===" << std::endl;
    
    // Sort by number of groups
    std::vector<std::pair<int, int>> sorted_counts(num_groups_counts.begin(), num_groups_counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end());
    
    for(const auto& [num_groups, count] : sorted_counts){
        std::cout << "num_groups = " << num_groups << ": " << count << " times (" 
                  << (100.0 * count / num_samples) << "%)" << std::endl;
    }

    // Print the log_density_num_groups vector if Wang-Landau is enabled
    if(m.wang_landau_modification_factor > 0){
        std::cout << "\n=== Log Density of States for Number of Groups ===" << std::endl;
        for(int num_groups = 0; num_groups < m.log_density_num_groups.size(); num_groups++){
            std::cout << "num_groups = " << num_groups << ": log_density = " << m.log_density_num_groups[num_groups] << std::endl;
        }
    }
    
    // Print parameter statistics
    std::cout << "\n=== Parameter Statistics ===" << std::endl;
    std::cout << "Mean mean_degree_scaling: " << (mean_degree_scaling_sum / num_samples_used) << std::endl;
    std::cout << "Mean degree_correction: " << (degree_correction_sum / num_samples_used) << std::endl;
    std::cout << "Mean variation_in: " << (variation_in_sum / num_samples_used) << std::endl;
    std::cout << "Mean variation_out: " << (variation_out_sum / num_samples_used) << std::endl;
    std::cout << "Mean ties_parameter: " << (ties_parameter_sum / num_samples_used) << std::endl;
    
    // Print mean_degree_scaling histogram
    std::cout << "\n=== mean_degree_scaling Histogram ===" << std::endl;
    for(int bin = -10; bin < 0; bin++){
        double lower = bin / 10.0;
        double upper = (bin + 1) / 10.0;
        int count = mean_degree_scaling_hist[bin];
        std::cout << "[" << lower << ", " << upper << "): " << count << " times (" 
                  << (100.0 * count / num_samples_used) << "%)" << std::endl;
    }
    
    // Print degree_correction histogram
    std::cout << "\n=== degree_correction Histogram ===" << std::endl;
    for(int bin = 0; bin < 10; bin++){
        double lower = bin / 10.0;
        double upper = (bin + 1) / 10.0;
        int count = degree_correction_hist[bin];
        std::cout << "[" << lower << ", " << upper << "): " << count << " times (" 
                  << (100.0 * count / num_samples_used) << "%)" << std::endl;
    }
    
    // Print variation_in histogram
    std::cout << "\n=== variation_in Histogram ===" << std::endl;
    for(int bin = 0; bin < 10; bin++){
        double lower = bin / 10.0;
        double upper = (bin + 1) / 10.0;
        int count = variation_in_hist[bin];
        std::cout << "[" << lower << ", " << upper << "): " << count << " times (" 
                  << (100.0 * count / num_samples_used) << "%)" << std::endl;
    }
    
    // Print variation_out histogram
    std::cout << "\n=== variation_out Histogram ===" << std::endl;
    for(int bin = 0; bin < 10; bin++){
        double lower = bin / 10.0;
        double upper = (bin + 1) / 10.0;
        int count = variation_out_hist[bin];
        std::cout << "[" << lower << ", " << upper << "): " << count << " times (" 
                  << (100.0 * count / num_samples_used) << "%)" << std::endl;
    }

    // Print ties_parameter histogram
    std::cout << "\n=== ties_parameter Histogram ===" << std::endl;
    for(int bin = 0; bin < 100; bin++){
        double lower = bin / 10.0;
        double upper = (bin + 1) / 10.0;
        int count = ties_parameter_hist[bin];
        if(count > 0) {  // Only print non-empty bins
            std::cout << "[" << lower << ", " << upper << "): " << count << " times (" 
                      << (100.0 * count / num_samples_used) << "%)" << std::endl;
        }
    }

    // Print individual_depth histogram
    std::cout << "\n=== individual_depth Histogram ===" << std::endl;
    for(int bin = 0; bin < 20; bin++){
        double lower = bin * 0.5;
        double upper = (bin + 1) * 0.5;
        int count = individual_depth_hist[bin];
        std::cout << "[" << lower << ", " << upper << "): " << count << " times ("
                  << (100.0 * count / num_samples_used) << "%)" << std::endl;
    }

    // Print group_depth histogram
    std::cout << "\n=== group_depth Histogram ===" << std::endl;
    for(int bin = 0; bin < 20; bin++){
        double lower = bin * 0.5;
        double upper = (bin + 1) * 0.5;
        int count = group_depth_hist[bin];
        std::cout << "[" << lower << ", " << upper << "): " << count << " times ("
                  << (100.0 * count / num_samples_used) << "%)" << std::endl;
    }
    
    // Print first score histogram
    std::cout << "\n=== First Score Histogram ===" << std::endl;
    for(int bin = 0; bin < 60; bin++){
        double lower = (bin * 0.1) - 3.0;
        double upper = ((bin + 1) * 0.1) - 3.0;
        int count = first_score_hist[bin];
        std::cout << "[" << lower << ", " << upper << "): " << count << " times ("
                  << (100.0 * count / num_samples_used) << "%)" << std::endl;
    }
    
    // Print partition frequency statistics
    std::cout << "\n=== Partition Frequency ===" << std::endl;
    std::cout << "Total unique partitions sampled: " << partition_counts.size() << std::endl;
    
    // Sort by frequency (descending)
    std::vector<std::pair<std::string, int>> sorted_partitions(partition_counts.begin(), partition_counts.end());
    std::sort(sorted_partitions.begin(), sorted_partitions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Print top 20 most frequent partitions
    int print_count = std::min(20, static_cast<int>(sorted_partitions.size()));
    for(int i = 0; i < print_count; i++) {
        const auto& [partition_str, count] = sorted_partitions[i];
        std::cout << "Partition [" << partition_str << "]: " << count << " times ("
                  << (100.0 * count / num_samples_used) << "%)" << std::endl;
    }
    
    if(sorted_partitions.size() > 20) {
        std::cout << "... and " << (sorted_partitions.size() - 20) << " more unique partitions" << std::endl;
    }
    
    // Print proposal statistics
    std::cout << "\n=== Proposal Statistics ===" << std::endl;
    std::cout << "Single-site proposals: " << m.single_site_proposals << " (accepted: " 
              << m.accepted_single_site_proposals << ", " 
              << (100.0 * m.accepted_single_site_proposals / std::max(1, m.single_site_proposals)) << "%)" << std::endl;
    std::cout << "Parameter proposals: " << m.parameter_proposals << " (accepted: " 
              << m.accepted_parameter_proposals << ", " 
              << (100.0 * m.accepted_parameter_proposals / std::max(1, m.parameter_proposals)) << "%)" << std::endl;
    std::cout << "  - degree_correction: " << m.degree_correction_proposals << " (accepted: " 
              << m.accepted_degree_correction_proposals << ", " 
              << (100.0 * m.accepted_degree_correction_proposals / std::max(1, m.degree_correction_proposals)) << "%)" << std::endl;
    std::cout << "  - variation_in: " << m.variation_in_proposals << " (accepted: " 
              << m.accepted_variation_in_proposals << ", " 
              << (100.0 * m.accepted_variation_in_proposals / std::max(1, m.variation_in_proposals)) << "%)" << std::endl;
    std::cout << "  - variation_out: " << m.variation_out_proposals << " (accepted: " 
              << m.accepted_variation_out_proposals << ", " 
              << (100.0 * m.accepted_variation_out_proposals / std::max(1, m.variation_out_proposals)) << "%)" << std::endl;
    std::cout << "  - mean_degree_scaling: " << m.mean_degree_scaling_proposals << " (accepted: " 
              << m.accepted_mean_degree_scaling_proposals << ", " 
              << (100.0 * m.accepted_mean_degree_scaling_proposals / std::max(1, m.mean_degree_scaling_proposals)) << "%)" << std::endl;
    std::cout << "  - individual_depth: " << m.individual_depth_proposals << " (accepted: " 
              << m.accepted_individual_depth_proposals << ", " 
              << (100.0 * m.accepted_individual_depth_proposals / std::max(1, m.individual_depth_proposals)) << "%)" << std::endl;
    std::cout << "  - group_depth: " << m.group_depth_proposals << " (accepted: " 
              << m.accepted_group_depth_proposals << ", " 
              << (100.0 * m.accepted_group_depth_proposals / std::max(1, m.group_depth_proposals)) << "%)" << std::endl;
    std::cout << "Merge proposals: " << m.merge_proposals << " (accepted: " 
              << m.accepted_merge_proposals << ", " 
              << (100.0 * m.accepted_merge_proposals / std::max(1, m.merge_proposals)) << "%)" << std::endl;
    std::cout << "Split proposals: " << m.split_proposals << " (accepted: " 
              << m.accepted_split_proposals << ", " 
              << (100.0 * m.accepted_split_proposals / std::max(1, m.split_proposals)) << "%)" << std::endl;
    
    std::cout << "MCMC stub constructed and MCMC_sweep invoked." << std::endl;
    return 0;
}
