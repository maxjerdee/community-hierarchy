// Script to allow for testing and debugging the MCMC core class without needing to run the full Python interface. This is useful for faster iteration during development and debugging of the core MCMC algorithm.

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
                if (edge_type == "neutral") {
                    if (ties[source].find(target) != ties[source].end()) {
                        ties[source][target] += 1.0;
                    } else {
                        ties[source][target] = 1.0;
                    }
                    if (ties[target].find(source) != ties[target].end()) {
                        ties[target][source] += 1.0;
                    } else {
                        ties[target][source] = 1.0;
                    }
                } else {
                    if (wins[source].find(target) != wins[source].end()) {
                        wins[source][target] += 1.0;
                    } else {
                        wins[source][target] = 1.0;
                    }
                }
            }
            in_edge = false;
        }
    }

    file.close();

    double total_wins = 0.0;
    for (const auto& row : wins) {
        for (const auto& [_, weight] : row) {
            total_wins += weight;
        }
    }

    double total_ties_stored = 0.0;
    for (const auto& row : ties) {
        for (const auto& [_, weight] : row) {
            total_ties_stored += weight;
        }
    }

    std::cout << "[mcmc_test] Parsed GML: total_wins=" << total_wins
              << ", total_ties=" << (total_ties_stored / 2.0)
              << " (stored_ties_sum=" << total_ties_stored << ")" << std::endl;

    return std::make_pair(wins, ties);
}

// Read the partition from group labels in the GML file (if present).
std::vector<int> read_partition(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }
    std::unordered_map<std::string, int> group_name_to_id;
    std::vector<int> partition;

    std::string line;
    int current_node_id = -1;

    while (std::getline(file, line)) {
        if (line.find("node [") != std::string::npos) {
            current_node_id = -1;
        } else if (current_node_id == -1 && line.find("id") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("id\\s+(\\d+)"))) {
                current_node_id = std::stoi(match[1]);
            }
        } else if (current_node_id != -1 && line.find("group") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("group\\s+(\"([^\"]+)\"|([^\\s\\]]+))"))) {
                std::string group_name = match[2].matched ? match[2].str() : match[3].str();
                if (group_name_to_id.find(group_name) == group_name_to_id.end()) {
                    group_name_to_id[group_name] = group_name_to_id.size();
                }
                int group_id = group_name_to_id[group_name];
                if (current_node_id >= static_cast<int>(partition.size())) {
                    partition.resize(current_node_id + 1, -1);
                }
                partition[current_node_id] = group_id;
            }
        }
    }

    file.close();
    return partition;
}

// Read node scores from the GML file (if present).
std::vector<double> read_scores(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<double> scores;
    std::string line;
    int current_node_id = -1;

    while (std::getline(file, line)) {
        if (line.find("node [") != std::string::npos) {
            current_node_id = -1;
        } else if (current_node_id == -1 && line.find("id") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("id\\s+(\\d+)"))) {
                current_node_id = std::stoi(match[1]);
            }
        } else if (current_node_id != -1 && line.find("score") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("score\\s+(\"?([-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?)\"?)"))) {
                double score = std::stod(match[2]);
                if (current_node_id >= static_cast<int>(scores.size())) {
                    scores.resize(current_node_id + 1, 0.0);
                }
                scores[current_node_id] = score;
            }
        }
    }

    file.close();
    return scores;
}


int main() {
    // The original input (english-football-restricted-wins_and_ties_integrated-scored.gml)
    // was removed in 1b78c87; english-football.gml is the current version of the dataset.
    const std::string gml_path = "examples/data/networks/english-football.gml";
    // const std::string gml_path = "examples/data/networks/friends.gml";
    // const std::string gml_path = "examples/data/networks/highschool_51.gml";
    // const std::string gml_path = "examples/data/networks/business_depts.gml";
    // const std::string gml_path = "examples/data/networks/highschool_30.gml";
    // const std::string gml_path = "examples/data/networks/highschool_84.gml";
    // const std::string gml_path = "examples/data/networks/example_3_nodes.gml";

    auto [wins, ties] = read_gml(gml_path);
    std::vector<int> partition;
    std::vector<double> scores;

    if (partition.empty()) {
        partition.resize(wins.size());
        for (std::size_t i = 0; i < wins.size(); i++) {
            partition[i] = rand() % 2;
        }
        std::cout << "No partition found in file, initialized randomly" << std::endl;
    }

    if (scores.size() < wins.size()) {
        scores.resize(wins.size(), 0.0);
    }

    std::cout << "Loaded graph with " << wins.size() << " nodes" << std::endl;

    MCMC m(wins,
        ties,
        {},             // group_adj: empty = single-layer mode (added in 568cf60)
        partition,
        false,          // partition_fixed
        scores,
        false,          // scores_fixed
        2,              // num_groups
        false,          // num_groups_fixed
        false,          // assortative
        0.5,            // variation_in
        true,           // variation_in_fixed
        0.5,            // variation_out
        true,           // variation_out_fixed
        0.5,            // degree_correction
        true,           // degree_correction_fixed
        1.0,            // individual_depth
        false,          // individual_depth_fixed
        1.0,            // group_depth
        false,          // group_depth_fixed
        1.0,            // ties_parameter
        false,          // ties_parameter_fixed
        42);            // seed

    std::cout << "Initial state: ";
    if (m.scores.size() > 0 && m.partition.size() > 0) {
        std::cout << "Node 0 -> Group " << m.partition[0] << ", Score " << m.scores[0] << " ";
    }
    if (m.scores.size() > 1 && m.partition.size() > 1) {
        std::cout << "Node 1 -> Group " << m.partition[1] << ", Score " << m.scores[1] << " ";
    }
    std::cout << std::endl;

    int num_samples = 100;
    int num_samples_used = num_samples / 2;

    std::unordered_map<int, int> num_groups_counts;
    double degree_correction_sum = 0.0;
    double variation_in_sum = 0.0;
    double variation_out_sum = 0.0;
    double ties_parameter_sum = 0.0;
    std::unordered_map<std::string, int> partition_counts;

    std::map<int, int> degree_correction_hist;
    std::map<int, int> variation_in_hist;
    std::map<int, int> variation_out_hist;
    std::map<int, int> ties_parameter_hist;
    std::map<int, int> individual_depth_hist;
    std::map<int, int> group_depth_hist;
    std::map<int, int> first_score_hist;

    for (int i = 0; i < num_samples; i++) {
        printf("MCMC Sweep %d\n", i + 1);
        m.MCMC_sweep(1.0, 100);

        std::cout << "Current individual_depth: " << m.individual_depth
                  << ", group_depth: " << m.group_depth << std::endl;
        std::cout << "Current H_A_b: " << m.H_A_b
                  << ", H_Adir_G_A_s_nu: " << m.H_Adir_G_A_s_nu
                  << ", H_s_G_b_depths: " << m.H_s_G_b_depths << std::endl;

        if (i >= num_samples - num_samples_used) {
            num_groups_counts[m.num_groups]++;

            degree_correction_sum += m.degree_correction;
            variation_in_sum += m.variation_in;
            variation_out_sum += m.variation_out;
            ties_parameter_sum += m.ties_parameter;

            std::string partition_str = "";
            for (int j = 0; j < static_cast<int>(m.partition.size()); j++) {
                partition_str += std::to_string(m.partition[j]);
                if (j < static_cast<int>(m.partition.size()) - 1) partition_str += ",";
            }
            partition_counts[partition_str]++;

            int dc_bin  = std::max(0, std::min(10, static_cast<int>(std::floor(m.degree_correction * 10))));
            int vi_bin  = std::max(0, std::min(10, static_cast<int>(std::floor(m.variation_in * 10))));
            int vo_bin  = std::max(0, std::min(10, static_cast<int>(std::floor(m.variation_out * 10))));
            int tp_bin  = std::max(0, std::min(99, static_cast<int>(std::floor(m.ties_parameter * 10))));
            int id_bin  = std::max(0, std::min(19, static_cast<int>(std::floor(m.individual_depth * 2.0))));
            int gd_bin  = std::max(0, std::min(19, static_cast<int>(std::floor(m.group_depth * 2.0))));
            int sc_bin  = std::max(0, std::min(59, static_cast<int>(std::floor((m.scores[0] + 3.0) * 10.0))));
            degree_correction_hist[dc_bin]++;
            variation_in_hist[vi_bin]++;
            variation_out_hist[vo_bin]++;
            ties_parameter_hist[tp_bin]++;
            individual_depth_hist[id_bin]++;
            group_depth_hist[gd_bin]++;
            first_score_hist[sc_bin]++;
        }
    }

    std::cout << "\n=== Final MCMC State ===" << std::endl;
    std::cout << "Number of groups: " << m.num_groups << std::endl;
    std::cout << "density_in: " << m.density_in << std::endl;
    std::cout << "density_out: " << m.density_out << std::endl;
    std::cout << "variation_in: " << m.variation_in << std::endl;
    std::cout << "variation_out: " << m.variation_out << std::endl;
    std::cout << "degree_correction: " << m.degree_correction << std::endl;
    std::cout << "concentration_in: " << m.concentration_in << std::endl;
    std::cout << "concentration_out: " << m.concentration_out << std::endl;
    std::cout << "degree_correction_concentration: " << m.degree_correction_concentration << std::endl;
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
    for (const auto& group : m.partition) {
        std::cout << group << " ";
    }
    std::cout << std::endl;

    std::cout << "Final scores: ";
    for (const auto& score : m.scores) {
        std::cout << score << " ";
    }
    std::cout << std::endl;

    std::cout << "\n=== Timing Summary ===" << std::endl;
    std::cout << "Time in H_k: " << m.time_H_k << "s" << std::endl;
    std::cout << "Time in H_M: " << m.time_H_M << "s" << std::endl;
    std::cout << "Time in H_A_G_k_M_b: " << m.time_H_A_G_k_M_b << "s" << std::endl;
    std::cout << "Time in single-site proposals: " << m.time_single_site_proposals << "s" << std::endl;
    std::cout << "Time in parameter proposals: " << m.time_parameter_proposals << "s" << std::endl;

    std::cout << "\n=== Number of Groups Distribution ===" << std::endl;
    std::vector<std::pair<int, int>> sorted_counts(num_groups_counts.begin(), num_groups_counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end());
    for (const auto& [ng, count] : sorted_counts) {
        std::cout << "num_groups = " << ng << ": " << count << " times ("
                  << (100.0 * count / num_samples) << "%)" << std::endl;
    }

    std::cout << "\n=== Parameter Statistics ===" << std::endl;
    std::cout << "Mean degree_correction: " << (degree_correction_sum / num_samples_used) << std::endl;
    std::cout << "Mean variation_in: " << (variation_in_sum / num_samples_used) << std::endl;
    std::cout << "Mean variation_out: " << (variation_out_sum / num_samples_used) << std::endl;
    std::cout << "Mean ties_parameter: " << (ties_parameter_sum / num_samples_used) << std::endl;

    std::cout << "\n=== degree_correction Histogram ===" << std::endl;
    for (int bin = 0; bin < 10; bin++) {
        double lo = bin / 10.0, hi = (bin + 1) / 10.0;
        int cnt = degree_correction_hist[bin];
        std::cout << "[" << lo << ", " << hi << "): " << cnt << " (" << (100.0 * cnt / num_samples_used) << "%)" << std::endl;
    }

    std::cout << "\n=== variation_in Histogram ===" << std::endl;
    for (int bin = 0; bin < 10; bin++) {
        double lo = bin / 10.0, hi = (bin + 1) / 10.0;
        int cnt = variation_in_hist[bin];
        std::cout << "[" << lo << ", " << hi << "): " << cnt << " (" << (100.0 * cnt / num_samples_used) << "%)" << std::endl;
    }

    std::cout << "\n=== variation_out Histogram ===" << std::endl;
    for (int bin = 0; bin < 10; bin++) {
        double lo = bin / 10.0, hi = (bin + 1) / 10.0;
        int cnt = variation_out_hist[bin];
        std::cout << "[" << lo << ", " << hi << "): " << cnt << " (" << (100.0 * cnt / num_samples_used) << "%)" << std::endl;
    }

    std::cout << "\n=== ties_parameter Histogram ===" << std::endl;
    for (int bin = 0; bin < 100; bin++) {
        int cnt = ties_parameter_hist[bin];
        if (cnt > 0) {
            double lo = bin / 10.0, hi = (bin + 1) / 10.0;
            std::cout << "[" << lo << ", " << hi << "): " << cnt << " (" << (100.0 * cnt / num_samples_used) << "%)" << std::endl;
        }
    }

    std::cout << "\n=== individual_depth Histogram ===" << std::endl;
    for (int bin = 0; bin < 20; bin++) {
        double lo = bin * 0.5, hi = (bin + 1) * 0.5;
        int cnt = individual_depth_hist[bin];
        std::cout << "[" << lo << ", " << hi << "): " << cnt << " (" << (100.0 * cnt / num_samples_used) << "%)" << std::endl;
    }

    std::cout << "\n=== group_depth Histogram ===" << std::endl;
    for (int bin = 0; bin < 20; bin++) {
        double lo = bin * 0.5, hi = (bin + 1) * 0.5;
        int cnt = group_depth_hist[bin];
        std::cout << "[" << lo << ", " << hi << "): " << cnt << " (" << (100.0 * cnt / num_samples_used) << "%)" << std::endl;
    }

    std::cout << "\n=== First Score Histogram ===" << std::endl;
    for (int bin = 0; bin < 60; bin++) {
        double lo = (bin * 0.1) - 3.0, hi = ((bin + 1) * 0.1) - 3.0;
        int cnt = first_score_hist[bin];
        std::cout << "[" << lo << ", " << hi << "): " << cnt << " (" << (100.0 * cnt / num_samples_used) << "%)" << std::endl;
    }

    std::cout << "\n=== Partition Frequency ===" << std::endl;
    std::cout << "Total unique partitions sampled: " << partition_counts.size() << std::endl;
    std::vector<std::pair<std::string, int>> sorted_partitions(partition_counts.begin(), partition_counts.end());
    std::sort(sorted_partitions.begin(), sorted_partitions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    int print_count = std::min(20, static_cast<int>(sorted_partitions.size()));
    for (int i = 0; i < print_count; i++) {
        const auto& [pstr, cnt] = sorted_partitions[i];
        std::cout << "Partition [" << pstr << "]: " << cnt << " (" << (100.0 * cnt / num_samples_used) << "%)" << std::endl;
    }
    if (static_cast<int>(sorted_partitions.size()) > 20) {
        std::cout << "... and " << (sorted_partitions.size() - 20) << " more unique partitions" << std::endl;
    }

    std::cout << "\n=== Proposal Statistics ===" << std::endl;
    std::cout << "Single-site: " << m.single_site_proposals
              << " (accepted: " << m.accepted_single_site_proposals
              << ", " << (100.0 * m.accepted_single_site_proposals / std::max(1, m.single_site_proposals)) << "%)" << std::endl;
    std::cout << "Score: " << m.score_proposals
              << " (accepted: " << m.accepted_score_proposals
              << ", " << (100.0 * m.accepted_score_proposals / std::max(1, m.score_proposals)) << "%)" << std::endl;
    std::cout << "Parameter: " << m.parameter_proposals
              << " (accepted: " << m.accepted_parameter_proposals
              << ", " << (100.0 * m.accepted_parameter_proposals / std::max(1, m.parameter_proposals)) << "%)" << std::endl;
    std::cout << "  - degree_correction: " << m.degree_correction_proposals
              << " (accepted: " << m.accepted_degree_correction_proposals
              << ", " << (100.0 * m.accepted_degree_correction_proposals / std::max(1, m.degree_correction_proposals)) << "%)" << std::endl;
    std::cout << "  - variation_in: " << m.variation_in_proposals
              << " (accepted: " << m.accepted_variation_in_proposals
              << ", " << (100.0 * m.accepted_variation_in_proposals / std::max(1, m.variation_in_proposals)) << "%)" << std::endl;
    std::cout << "  - variation_out: " << m.variation_out_proposals
              << " (accepted: " << m.accepted_variation_out_proposals
              << ", " << (100.0 * m.accepted_variation_out_proposals / std::max(1, m.variation_out_proposals)) << "%)" << std::endl;
    std::cout << "  - individual_depth: " << m.individual_depth_proposals
              << " (accepted: " << m.accepted_individual_depth_proposals
              << ", " << (100.0 * m.accepted_individual_depth_proposals / std::max(1, m.individual_depth_proposals)) << "%)" << std::endl;
    std::cout << "  - group_depth: " << m.group_depth_proposals
              << " (accepted: " << m.accepted_group_depth_proposals
              << ", " << (100.0 * m.accepted_group_depth_proposals / std::max(1, m.group_depth_proposals)) << "%)" << std::endl;
    std::cout << "  - ties_parameter: " << m.ties_parameter_proposals
              << " (accepted: " << m.accepted_ties_parameter_proposals
              << ", " << (100.0 * m.accepted_ties_parameter_proposals / std::max(1, m.ties_parameter_proposals)) << "%)" << std::endl;

    return 0;
}
