// Script to allow for testing and debugging the MCMC core class without needing to run the full Python interface. This is useful for faster iteration during development and debugging of the core MCMC algorithm.

#include "MCMC_core.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <regex>
#include <algorithm>
#include <cmath>
#include <filesystem>

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
                // Add edge based on type, incrementing weight if edge already exists (multigraph support)
                // Default to "dominant" if no type is specified
                if (edge_type == "neutral") {
                    // Increment weight for undirected edge in both directions
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
                    // Default to "dominant" (edge_type == "dominant" or edge_type == "")
                    if (wins[source].find(target) != wins[source].end()) {
                        wins[source][target] += 1.0;  // Increment weight for multigraph
                    } else {
                        wins[source][target] = 1.0;   // First edge between these nodes
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

// Read the partition from group labels in the GML file (if present). Returns a vector where index is node id and value is group id. If no group labels are found, returns an empty vector.
// node [
//     id 0
//     label "Barnsley"
//     group "league-one"
//   ]
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
        // Print line for debugging
        if (line.find("node [") != std::string::npos) {
            current_node_id = -1;
        } else if (current_node_id == -1 && line.find("id") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("id\\s+(\\d+)"))) {
                current_node_id = std::stoi(match[1]);
                std::cout << "Line: " << line << std::endl;
                std::cout << "Found node with id: " << current_node_id << std::endl;
            }
        } else if (current_node_id != -1 && line.find("group") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("group\\s+(\"([^\"]+)\"|([^\\s\\]]+))"))) {
                std::string group_name = match[2].matched ? match[2].str() : match[3].str();
                std::cout << "Line: " << line << std::endl;
                std::cout << "Found group label: " << group_name << std::endl;
                if (group_name_to_id.find(group_name) == group_name_to_id.end()) {
                    group_name_to_id[group_name] = group_name_to_id.size(); // Assign new group id
                }
                int group_id = group_name_to_id[group_name];
                if (current_node_id >= partition.size()) {
                    partition.resize(current_node_id + 1, -1); // Resize partition vector if needed
                }
                partition[current_node_id] = group_id;
            }
        }
    }

    file.close();
    return partition;
}

// Read the modularity_group integer attribute from node blocks in the GML file.
// Returns a vector where index is node id and value is the modularity_group integer.
// Returns an empty vector if the attribute is absent.
std::vector<int> read_modularity_groups(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<int> groups;
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
        } else if (current_node_id != -1 && line.find("modularity_group") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("modularity_group\\s+(\\d+)"))) {
                int group_id = std::stoi(match[1]);
                if (current_node_id >= static_cast<int>(groups.size())) {
                    groups.resize(current_node_id + 1, 0);
                }
                groups[current_node_id] = group_id;
            }
        }
    }

    file.close();
    return groups;
}

// Read win_percentage float attribute from node blocks in the GML file.
// Returns a vector where index is node id and value is win_percentage.
// Returns an empty vector if the attribute is absent.
std::vector<double> read_win_percentages(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<double> win_pcts;
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
        } else if (current_node_id != -1 && line.find("win_percentage") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(line, match, std::regex("win_percentage\\s+(\"?([-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?)\"?)"))) {
                double val = std::stod(match[1]);
                if (current_node_id >= static_cast<int>(win_pcts.size())) {
                    win_pcts.resize(current_node_id + 1, 0.0);
                }
                win_pcts[current_node_id] = val;
            }
        }
    }

    file.close();
    return win_pcts;
}

// Read node scores from the GML file (if present). Returns a vector where index is node id and value is score.
// Supports lines like:
//   score 0.25
//   score -1
//   score "0.25"
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

std::vector<std::string> read_node_labels(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<std::string> labels;
    std::string line;
    int current_node_id = -1;
    bool in_node = false;

    while (std::getline(file, line)) {
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
        trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);

        if (trimmed.find("node") != std::string::npos && trimmed.find("node") < 5) {
            in_node = true;
            current_node_id = -1;
        } else if (in_node && current_node_id == -1 && trimmed.find("id") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(trimmed, match, std::regex("^id\\s+(\\d+)"))) {
                current_node_id = std::stoi(match[1]);
            }
        } else if (in_node && current_node_id != -1 && trimmed.find("label") != std::string::npos) {
            std::smatch match;
            if (std::regex_search(trimmed, match, std::regex("label\\s+\"([^\"]+)\""))) {
                if (current_node_id >= static_cast<int>(labels.size())) {
                    labels.resize(current_node_id + 1);
                }
                labels[current_node_id] = match[1];
            } else if (std::regex_search(trimmed, match, std::regex("label\\s+(\\S+)"))) {
                if (current_node_id >= static_cast<int>(labels.size())) {
                    labels.resize(current_node_id + 1);
                }
                labels[current_node_id] = match[1];
            }
        } else if (in_node && trimmed == "]") {
            in_node = false;
        }
    }

    file.close();
    return labels;
}

static std::string csv_escape(const std::string& value) {
    bool needs_quotes = value.find(',') != std::string::npos ||
                        value.find('"') != std::string::npos ||
                        value.find('\n') != std::string::npos;
    if (!needs_quotes) {
        return value;
    }

    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped += c;
        }
    }
    escaped += '"';
    return escaped;
}


int main(int argc, char** argv) {
    // Parse named flags: --input-gml, --output-csv, --num-samples, --sweeps-per-sample, --seed
    // Positional fallback: argv[1]=input, argv[2]=output
    std::string gml_path;
    std::string output_csv_arg;
    int num_samples_arg = 100;
    int sweeps_per_sample_arg = 1000;
    int seed_arg = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--input-gml" || arg == "-i") && i + 1 < argc) {
            gml_path = argv[++i];
        } else if ((arg == "--output-csv" || arg == "-o") && i + 1 < argc) {
            output_csv_arg = argv[++i];
        } else if (arg == "--num-samples" && i + 1 < argc) {
            num_samples_arg = std::stoi(argv[++i]);
        } else if (arg == "--sweeps-per-sample" && i + 1 < argc) {
            sweeps_per_sample_arg = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed_arg = std::stoi(argv[++i]);
        } else if (arg[0] != '-') {
            // Positional argument
            if (gml_path.empty()) gml_path = arg;
            else if (output_csv_arg.empty()) output_csv_arg = arg;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if (gml_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --input-gml <input.gml> [--output-csv <output.csv>]"
                  << " [--num-samples N] [--sweeps-per-sample N] [--seed N]" << std::endl;
        return 1;
    }

    std::filesystem::path output_csv_path;
    if (!output_csv_arg.empty()) {
        output_csv_path = output_csv_arg;
    } else {
        std::filesystem::path input_path(gml_path);
        output_csv_path = std::filesystem::path("samples") / (input_path.stem().string() + "_samples.csv");
    }

    auto [wins, ties] = read_gml(gml_path);
    std::vector<std::string> node_labels = read_node_labels(gml_path);

    // Seed partition from modularity_group attribute; fall back to random if absent.
    std::vector<int> partition = read_modularity_groups(gml_path);
    if (partition.empty()) {
        partition.resize(wins.size());
        for (std::size_t i = 0; i < wins.size(); i++) {
            partition[i] = rand() % 2;
        }
        std::cout << "No modularity_group found in file, initialized partition randomly" << std::endl;
    } else {
        std::cout << "Seeded partition from modularity_group attribute" << std::endl;
    }

    // Derive num_groups from the loaded partition.
    int init_num_groups = *std::max_element(partition.begin(), partition.end()) + 1;

    // Seed scores from win_percentage attribute; fall back to zeros if absent.
    std::vector<double> scores = read_win_percentages(gml_path);
    if (scores.empty()) {
        scores.resize(wins.size(), 0.0);
        std::cout << "No win_percentage found in file, initialized scores to zero" << std::endl;
    } else {
        std::cout << "Seeded scores from win_percentage attribute" << std::endl;
    }

    // Ensure vectors align with graph size.
    if (partition.size() < wins.size()) {
        partition.resize(wins.size(), 0);
    }
    if (scores.size() < wins.size()) {
        scores.resize(wins.size(), 0.0);
    }
    if (node_labels.size() < wins.size()) {
        node_labels.resize(wins.size());
    }
    for (std::size_t i = 0; i < node_labels.size(); ++i) {
        if (node_labels[i].empty()) {
            node_labels[i] = "node_" + std::to_string(i);
        }
    }

    if (!output_csv_path.parent_path().empty()) {
        std::filesystem::create_directories(output_csv_path.parent_path());
    }
    std::ofstream samples_csv(output_csv_path);
    if (!samples_csv.is_open()) {
        std::cerr << "Error: Could not open output CSV " << output_csv_path << std::endl;
        return 1;
    }

    samples_csv << "beta";
    for (std::size_t i = 0; i < wins.size(); i++) {
        samples_csv << "," << csv_escape(node_labels[i] + "_group");
    }
    for (std::size_t i = 0; i < wins.size(); i++) {
        samples_csv << "," << csv_escape(node_labels[i] + "_score");
    }
    samples_csv << ",num_groups,density_in,density_out,variation_in,variation_out,degree_correction,mean_degree_scaling,individual_depth,group_depth,ties_parameter,H_A_b,H_b,H_M,H_k,H_A_G_k_M_b,H_Adir_G_A_s_nu,H_s_G_b_depths\n";

    std::cout << "Writing samples to: " << output_csv_path << std::endl;
    std::cout << "Loaded graph with " << wins.size() << " nodes" << std::endl;

    // Canonical typical DCSBM
    MCMC m(wins,
        ties,
        partition,
        false,          // partition_fixed
        scores,
        false,          // scores_fixed
        init_num_groups, // num_groups (seeded from modularity partition)
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
        1.0,            // group_depth (passed as value)
        false,           // group_depth_fixed
        1.0,            // ties_parameter (passed as value)
        false,          // ties_parameter_fixed
        seed_arg);      // seed

    int num_samples = num_samples_arg;
    
    for(int i = 0; i < num_samples; i++){
        std::cout << "\rSample " << (i + 1) << "/" << num_samples << "  " << std::flush;
        m.MCMC_sweep(1.0, false, -1.0, -1, sweeps_per_sample_arg);

        const double beta = 1.0;
        samples_csv << beta;
        for (std::size_t node = 0; node < m.partition.size(); ++node) {
            samples_csv << "," << m.partition[node];
        }
        for (std::size_t node = 0; node < m.scores.size(); ++node) {
            samples_csv << "," << m.scores[node];
        }
        samples_csv << "," << m.num_groups
                    << "," << m.density_in
                    << "," << m.density_out
                    << "," << m.variation_in
                    << "," << m.variation_out
                    << "," << m.degree_correction
                    << "," << m.mean_degree_scaling
                    << "," << m.individual_depth
                    << "," << m.group_depth
                    << "," << m.ties_parameter
                    << "," << m.H_A_b
                    << "," << m.H_b
                    << "," << m.H_M
                    << "," << m.H_k
                    << "," << m.H_A_G_k_M_b
                    << "," << m.H_Adir_G_A_s_nu
                    << "," << m.H_s_G_b_depths
                    << "\n";
        samples_csv.flush();
    }
    std::cout << std::endl;
    std::cout << "Done. Saved samples to " << output_csv_path << std::endl;
    return 0;
}
