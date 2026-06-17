#include "consensus.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

std::vector<int> ConsensusClustering::consensus_clustering(
    const std::vector<std::vector<int>>& clusterings,
    const std::string& metric
) {
    if (clusterings.empty()) {
        throw std::invalid_argument("clusterings cannot be empty");
    }
    
    int num_clusterings = clusterings.size();
    int num_nodes = clusterings[0].size();
    
    // Validate all clusterings have the same size
    for (const auto& clustering : clusterings) {
        if (clustering.size() != num_nodes) {
            throw std::invalid_argument("All clusterings must have the same number of nodes");
        }
    }
    
    // Validate metric
    if (metric != "L1" && metric != "L2") {
        throw std::invalid_argument("metric must be 'L1' or 'L2'");
    }
    
    // Compute aggregate coincidence matrix efficiently
    // Group nodes by cluster membership and only increment for same-group pairs
    std::vector<std::vector<double>> aggregate_matrix(num_nodes, std::vector<double>(num_nodes, 0.0));
    
    for (const auto& clustering : clusterings) {
        // Group nodes by their cluster assignment
        std::unordered_map<int, std::vector<int>> groups;
        for (int node = 0; node < num_nodes; ++node) {
            groups[clustering[node]].push_back(node);
        }
        
        // For each group, increment matrix entries only for members of that group
        for (std::unordered_map<int, std::vector<int>>::const_iterator it = groups.begin(); it != groups.end(); ++it) {
            const std::vector<int>& members = it->second;
            // // Print group members for debugging
            // std::cout << "Clustering group " << group_id << " members: ";
            // for (int node : members) {
            //     std::cout << node << " ";
            // }
            // std::cout << std::endl;
            for (int i : members) {
                for (int j : members) {
                    aggregate_matrix[i][j] += 1.0;
                }
            }
        }
    }
    
    // Normalize by number of clusterings
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            aggregate_matrix[i][j] /= num_clusterings;
        }
    }
    
    // Compute base distance once (treating all pairs as 0)
    double base_distance = 0.0;
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            double val = aggregate_matrix[i][j];
            if (metric == "L1") {
                base_distance += val;  // |aggregate[i][j] - 0|
            } else {  // L2
                base_distance += val * val;  // (aggregate[i][j] - 0)^2
            }
        }
    }
    // printf("Base distance (all pairs 0): %f\n", base_distance);
    
    // Find the best clustering
    double best_distance = std::numeric_limits<double>::infinity();
    std::vector<int> best_clustering = clusterings[0];
    int best_index = 0;
    
    int clustering_index = 0;
    for (const auto& clustering : clusterings) {
        // Group nodes by their cluster assignment
        std::unordered_map<int, std::vector<int>> groups;
        for (int node = 0; node < num_nodes; ++node) {
            groups[clustering[node]].push_back(node);
        }
        
        // Compute distance starting from base and adjusting for within-group pairs
        double distance = base_distance;
        
        // Adjust for within-group pairs (should be 1.0, not 0.0)
        for (std::unordered_map<int, std::vector<int>>::const_iterator it = groups.begin(); it != groups.end(); ++it) {
            const std::vector<int>& members = it->second;
            for (int i : members) {
                for (int j : members) {
                    double val = aggregate_matrix[i][j];
                    
                    if (metric == "L1") {
                        // Remove the |val - 0| and add |val - 1|
                        distance -= val;
                        distance += std::abs(val - 1.0);
                    } else {  // L2
                        // Remove the (val - 0)^2 and add (val - 1)^2
                        distance -= val * val;
                        double diff = val - 1.0;
                        distance += diff * diff;
                    }
                }
            }
        }
        
        if (distance < best_distance) {
            best_distance = distance;
            best_clustering = clustering;
            best_index = clustering_index;
        }
        
        clustering_index++;
    }
    
    // Count groups in best clustering
    std::unordered_map<int, int> group_count;
    for (int group_id : best_clustering) {
        group_count[group_id]++;
    }
    int num_groups = group_count.size();
    
    return best_clustering;
}
