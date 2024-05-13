/* This script uses parallel tempering on single site moves to sample from the posterior distribution
 * of the microcanonical Degree-Corrected Stochastic Block Model (DCSBM) as implemented in graph-tool.
 * 
 * 
 * Written by Max Jerdee  11 MAY 2024
 */

// After making the file (run make in the directory), the executable may be run for example as:
// ./SBM-BT -i ../../data/gml_files/AddHealth/highschool_1.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_1.csv
// ./SBM-BT -i ../../data/gml_files/AddHealth/highschool_57.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_57.csv
// ./SBM-BT -i ../../data/gml_files/misc/test.gml -o ../../data/samples/misc/SBM-BT/test.csv

#include <getopt.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <vector>
#include <iterator>
#include <random>
#include <vector>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <unordered_set>

/* Global variables */
// Model configuration:
const double MEAN_SIGMA = 3; // A priori mean value of sigma_I and sigma_C, the inter and intra community status spreads

const bool FIX_Q = false; // Whether to fix the number of groups q
// const bool FIX_Q = true; // Whether to fix the number of groups q

// Optimization (parallel tempering) configuration:
const double MIN_BETA = 0.1;
// const double MIN_BETA = 1;
// const double MAX_BETA = 0.1;
const double MAX_BETA = 1;
const double BETA_SPACING = 0.1;
const int THERMAL_SWAP_FREQUENCY = 1; // After how many iterations to consider a thermal swap (swap betas of two chains)

// Sampling configuration:
const int SAMPLES_BACK_BURNIN_CHECK = 500; // Number of samples back that we will check for an increase from to say that we have burned in enough to start recording samples 
// const int SAMPLES_BACK_BURNIN_CHECK = 100; // Number of samples back that we will check for an increase from to say that we have burned in enough to start recording samples 

const int ITERATIONS_BETWEEN_SAMPLES = 10000; // Total number of iterations between recorded samples
const int NUM_SAMPLES = 20000; // How many samples to take

const double NUMERICAL_CHECK_EPSILON = 0.00001; // Tolerance to use when checking that the bookkeeping is in sync.

const double SIGMA_C_TILDE_STEP_SIZE = 0.25;// Typical amount to change sigma_C in each proposal (in the transformed parameter space)
const double SIGMA_I_TILDE_STEP_SIZE = 0.1;
const double NU_TILDE_STEP_SIZE = 0.1;
const double TOTAL_STATUS_TILDE_STEP_SIZE = 0.5;
const double COMMUNITY_STATUS_TILDE_STEP_SIZE = 0.25;

std::vector<std::vector<double>> log_q_vals; // log q(m,n) values, to be precomputed

/* Custom types */
// Static information (information which remains constant [shared between the chains])
typedef std::tuple<
    int, // n
    int, // m
    std::vector<std::unordered_map<int,int>>, // wins
    std::vector<std::unordered_map<int,int>>, // ties
    std::vector<std::unordered_map<int,int>>, // losses
    std::vector<std::unordered_map<int,int>>, // neighbors (map the neighbor to the A[i,j] value [multiplicity])
    std::vector<int> // degree_list, \vec{k} (qxk_max matrix)
> ConstInfoType; 

// MCMC State (independent latent parameters over which we sample)
typedef std::tuple<
std::vector<int>, // partition, \vec{b}
double, // sigma_C, spread of community statuses
double, // sigma_I, spread of status inside communities
std::vector<double>, // community_status, \vec{c}, community statuses
std::vector<double>, // total_status, \vec{s}, total statuses
double // nu, the parameter that controls the likelihood of ties
> StateType;

const int sigma_C_StateTypeIndex = 1;
const int sigma_I_StateTypeIndex = 2;

// MCMC State bookkeeping (information that can be inferred from the MCMC state but is useful to have on hand)
typedef std::tuple<
    int, // q
    std::vector<int>, // node_counts, n^{(b)}
    std::vector<std::unordered_set<int>>, // group_members (list of the indices that belong to each group)
    std::vector<std::vector<int>>, // edge_counts, \mat{M} (qxq matrix)
    std::vector<int>, // edge_counts_sum, \vec{m} (q vector)
    std::vector<std::unordered_map<int,int>>, // degree_counts_by_group, degree_counts_by_group[r] is a map between the degree and its multiplicity in group r
    double, // H_A_G_Abar_s_nu
    double, // H_nu
    double, // H_s_G_c_b_sigmaI
    double, // H_c_G_sigmaC
    double, // H_sigmaC
    double, // H_sigmaI
    double, // H_Abar_G_M_b, Poisson encoding of the edges
    double, // H_k_G_eta, cost to transmit degrees given degree counts eta
    double, // H_eta_G_M_nb, cost to transmit the degree counts given the group sizes and the edge_count matrix
    double, // H_M_G_q, cost to transmit the edge_count matrix without conditioning on the diagonal or the margin
    double, // H_b_G_nb, cost to transmit the partition given the group sizes
    double, // H_nb_G_q, cost to transmit the group sizes
    double, // H_q, cost to transmit the number of colors
    double // energy (H = -log P)
    > StateBookkeepingType;

// Indices in the tuple at which to extract important information
const int qStateBookkeepingTypeIndex = 0;
const int energyStateBookkeepingTypeIndex = 19;

/* Function declarations */
// Initialize the state with a given partition
StateType initialize_state(std::vector<int>& partition_init, std::mt19937& rng);

// Randomly initialize the state 
StateType initialize_rand_state(ConstInfoType& const_info, std::mt19937& rng);

// Compute the bookkeeping information (including the energy) from the MCMC state
StateBookkeepingType get_state_bookkeeping(StateType& state, ConstInfoType& const_info);

// Check that the state and its bookkeeping are in alignment. If not print the error and exit.
void check_state_bookkeeping(StateType& state, StateBookkeepingType& state_bookkeeping, ConstInfoType& const_info);

// Generate proposed MCMC move: var_name, var_index, new_val, old_val, delta_q
std::tuple<std::string, int, double, double, int> generate_proposal(StateType& state, StateBookkeepingType& state_bookkeeping, std::mt19937& rng);

// Implement the proposed swap. Also return the change in energy, relevant for deciding whether to in fact accept this swap
double implement_proposal(std::string var_name, int var_index, double new_val, double old_val, int delta_q, StateType& state, StateBookkeepingType& state_bookkeeping, ConstInfoType& const_info, std::mt19937& rng);

// Parse an edgelist and community partition from a GML file
std::pair<std::vector<std::pair<int, int>>, std::vector<int>> parseGML(const std::string& filename);

// Helper functions for computation
// Function to compute log(binomial(a,b))
double log_binom(double a, double b);

// Overflow protected version of log(e^(sij/2) + 2 nu + e^(sji/2)) for evaluating H(A|Abar,s,nu)
double log_denom(double score_diff, double nu);

// Overflow protected version of log(e^a + e^b) for precomputing log_q_vals
double log_sum_exp(double a, double b);

double log_factorial(int n);
double log_factorial2(int n);

// Write samples to file
std::string state_csv_string(StateType& state, StateBookkeepingType& state_bookkeeping, ConstInfoType& const_info);

// Printing quantities (for debugging)
void print(ConstInfoType& const_info);
void print(StateBookkeepingType& state_bookkeeping);
void print(StateType& state);
void print(std::vector<std::vector<int>> table);
void print(std::vector<std::vector<double>> table);
void print(std::vector<int> vec);
void print(std::vector<double> vec);
void print(int val);
void print(double val);

//*********************** MAIN PROGRAM ***********************************************************************

int main(int argc, char *argv[]) {
    // Parse command line parameters
    int opt;
    std::string input_filename;
    std::string output_filename;
    // take command line arguments
    while ((opt = getopt(argc, argv, "i:o:D:M:T:")) != -1) {
        switch (opt) {
        case 'i':
            input_filename = optarg;
            break;
        case 'o':
            output_filename = optarg;
            break;
        default:
            exit(EXIT_FAILURE);
        }
    }
    // Write the header line of the samples csv
    std::ofstream outfile(output_filename);
    std::cout <<"Writing samples to "<< output_filename << "\n";
    outfile <<"partition,sigma_C,sigma_I,community_status,total_status,nu,H_A_G_Abar_s_nu,H_nu,H_s_G_c_b_sigmaI,H_c_G_sigmaC,H_sigmaC,H_sigmaI,H_Abar_G_M_b,H_k_G_eta,H_eta_G_M_nb,H_M_G_q,H_b_G_nb,H_nb_G_q,H_q,energy\n";

    bool warmup_done = false; // Start to record samples after a burn-in period, this is set to true when the energy starts to increase in the beta = 1 chain

    // RNG
    std::random_device rd;     // Only used once to initialise (seed) engine
    std::mt19937 rng(0);    // Seed
    // std::mt19937 rng(rd());    // Seed

    // Parse an edgelist and community partition from the given GML file. If no initial partition is given, initially assign all nodes to a single community.
    auto result = parseGML(input_filename);

    std::vector<std::pair<int, int>> edgelist = result.first;
    std::vector<int> partition_init = result.second;

    // From the edgelist, produce the background information constant across all of the chains:
    int n = partition_init.size();
    std::vector<std::unordered_map<int,int>> wins;
    std::vector<std::unordered_map<int,int>> ties;
    std::vector<std::unordered_map<int,int>> losses;
    std::vector<std::unordered_map<int,int>> neighbors; 
    for(int i = 0; i < n; i++){
        std::unordered_map<int,int> neighbors_i;
        neighbors.push_back(neighbors_i);
        std::unordered_map<int,int> wins_i;
        wins.push_back(wins_i);
        std::unordered_map<int,int> ties_i;
        ties.push_back(ties_i);
        std::unordered_map<int,int> losses_i;
        losses.push_back(losses_i);
    }
    std::vector<int> degree_list;

    for(std::pair<int,int> edge : edgelist){
        int i = edge.first;
        int j = edge.second;
        if(i == j){ 
            // std::cout << "Error: self-edge detected on node "<<i<<", not supported.\n";
            // exit(0);
            std::cout << "Warning: self-edge detected on node "<<i<<", not supported.\n";
        }else{
            // Record the interaction in both directions
            // If i is not already in the list of j's neighbors:
            if(neighbors[j].find(i) == neighbors[j].end()){
                neighbors[j][i] = 1;
            }else{ // No multiplicity increment for undirected edges
                // neighbors[j][i] += 1;
                ties[j][i] = 1; // Record a reciprocated interaction as a tie
            }
            // If j is not already in the list of i's neighbors:
            if(neighbors[i].find(j) == neighbors[i].end()){
                neighbors[i][j] = 1;
            }else{ // No multiplicity increment for undirected edges
                // neighbors[i][j] += 1;
                ties[i][j] = 1; // Record a reciprocated interaction as a tie
            }
        }
        // To only allow simple graphs:
        // // Check if j is not already in the list of i's neighbors:
        // if(std::find(neighbors[i].begin(), neighbors[i].end(), j) == neighbors[i].end()) {
        //     // Record the interaction in both directions
        //     neighbors[i].push_back(j);
        //     neighbors[j].push_back(i);
        // }
        // }else{ // If it is, there is a multiedge so throw error
        //     std::cout << "Error: multi-edge detected between nodes "<<i<<" and "<<j<<", not supported.\n";
        //     exit(0);
        // }
    }
    // Loop back through to record unreciprocated interactions as wins and losses
    for(std::pair<int,int> edge : edgelist){
        int i = edge.first;
        int j = edge.second;
        // Check that i and j did not reciprocate
        if(ties[i].find(j) == ties[i].end()) {
            // This edge indicates that i has an reciprocated friendship with j. A win for j, a loss for i. 
            losses[i][j] = 1;
            wins[j][i] = 1;
        }
    }
    // Loop through to count degrees and the total edges
    int m = 0;
    for(int i = 0; i < n; i++){
        degree_list.push_back(0);
        for(auto& [j, Aij] : neighbors[i]){
            degree_list[i] += Aij;
            m += 1;
        }
    }
    m /= 2; // Edges had been double counted
    std::cout << "Successfully parsed the GML file. n = "<<n<<", m = "<<m<<".\n";

    ConstInfoType const_info = std::make_tuple(n, m, wins, ties, losses, neighbors, degree_list);
    print(const_info);

    // Precompute the values of q(m,n) up to the total number of interactions 
    /* Boundary conditions from Xiaoran:
    # n >= 0 and m >= 0:
    if m > 0 and n == 0:
        return 0
    if m == 0 and n == 0:
        return 1
    if n == 1 and m > 0:
        return 1
    
    # m < 0 or n < 0
    if m < 0 or n < 0:
        return 0
    */
   std::cout << edgelist.size() << "\n";
    int max_m = 2*edgelist.size();
    std::cout << "Precomputing values of log q(m,n) up to m = " << max_m << "\n";
    for(int q_m = 0; q_m <= max_m; q_m++){
        std::vector<double> log_q_vals_r;
        for(int q_n = 0; q_n <= max_m; q_n++){
            if(q_n == 0){
                if(q_m == 0){
                    log_q_vals_r.push_back(0); // q = 1 for n = 0, m = 0
                }else{
                    log_q_vals_r.push_back(-INFINITY); // q = 0 for n = 0, m > 0
                }
            }else{
                log_q_vals_r.push_back(-INFINITY); // q = 0 initialization for n >= 1
            }
        }
        log_q_vals.push_back(log_q_vals_r);
    }
    // Perform recursion
    for(int q_n = 1; q_n <= max_m; q_n++){
        for(int q_m = 0; q_m <= max_m; q_m++){
            if(q_m >= q_n){
                // q(m,n) = q(m,n - 1) + q(m - n, n)
                log_q_vals[q_m][q_n] = log_sum_exp(log_q_vals[q_m][q_n - 1],log_q_vals[q_m - q_n][q_n]);
            }else{
                // If m < n, the restriction on the number of parts is irrelevant. 
                log_q_vals[q_m][q_n] = log_q_vals[q_m][q_n - 1];
            }
        }
    }
    
    // Checks
    // std::cout << "q(2,2) = " << std::exp(log_q_vals[2][2]) << "\n"; // 2
    // std::cout << "q(10,7) = " << std::exp(log_q_vals[10][7]) << "\n"; // 38
    std::cout << "Precomputation of q(m,n) values finished.\n";

    // Create the states
    std::vector<double> all_betas; // All of the betas available in order
    for(double beta = MIN_BETA; beta <= MAX_BETA; beta += BETA_SPACING){
        all_betas.push_back(beta);
    }
    int num_states = all_betas.size();

    // Initialize the states
    std::vector<StateType> states;
    std::vector<StateBookkeepingType> state_bookkeepings;
    std::vector<double> beta_is; // State i will evolve according to beta = all_betas[beta_is[i]]
    for(int i = 0; i < num_states; i++){
        beta_is.push_back(i);
        StateType state = initialize_state(partition_init, rng);
        StateBookkeepingType state_bookkeeping = get_state_bookkeeping(state, const_info);
        states.push_back(state);
        state_bookkeepings.push_back(state_bookkeeping);
        // print(state_bookkeepings[0]);
        // exit(0);
    }

    double p;
    std::vector<double> energy_history; // Write the histories of the energy (at beta = 1) in order to detect the burn in
    for(int sample_i = 0; sample_i < SAMPLES_BACK_BURNIN_CHECK; sample_i++){
        energy_history.push_back(INFINITY);
    }
    int num_warmups_taken = 0;
    int num_samples_taken = 0;
    while(num_samples_taken < NUM_SAMPLES){
        for(int t = 0; t < ITERATIONS_BETWEEN_SAMPLES; t++){
            // Perform a move on each state
            for(int state_i = 0; state_i < all_betas.size(); state_i++){
                auto [var_name, var_index, new_val, old_val, delta_q] = generate_proposal(states[state_i], state_bookkeepings[state_i], rng);
                double deltaH = implement_proposal(var_name, var_index, new_val, old_val, delta_q, states[state_i], state_bookkeepings[state_i], const_info, rng);
                // check_state_bookkeeping(states[state_i],state_bookkeepings[state_i], const_info); // Check bookkeeping after each move
                
                // std::cout << deltaH << " " << "\n";
                // Accept/reject
                double beta = all_betas[beta_is[state_i]];
                double p = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
                if(p < std::exp(-beta*deltaH)){
                    // Keep the flip
                    // std::cout << "deltaH: " << deltaH << " energy: " << std::get<energyStateBookkeepingTypeIndex>(state_bookkeepings[state_i]) << "\n";
                }else{
                    // Revert the flip (we could potentially speed this up by storing the proposed change in the bookkeeping and reverting it here, although I'm not sure if the added saving to memory overhead is worth it?)
                    deltaH = implement_proposal(var_name, var_index, old_val, new_val, -delta_q, states[state_i], state_bookkeepings[state_i], const_info, rng);
                    // check_state_bookkeeping(states[state_i],state_bookkeepings[state_i], const_info);
                }
            }
            if(t % THERMAL_SWAP_FREQUENCY == 0 && num_states >= 2){ // Consider a thermal swap
                int rand_state_i1 = std::uniform_int_distribution<int>(0, num_states - 1)(rng);
                int rand_state_i2 = std::uniform_int_distribution<int>(0, num_states - 2)(rng);
                if(rand_state_i2 >= rand_state_i1){ // Make sure not the same
                    rand_state_i2 += 1;
                }
                double beta_1 = all_betas[beta_is[rand_state_i1]];
                double beta_2 = all_betas[beta_is[rand_state_i2]];
                double energy_1 = std::get<energyStateBookkeepingTypeIndex>(state_bookkeepings[rand_state_i1]);
                double energy_2 = std::get<energyStateBookkeepingTypeIndex>(state_bookkeepings[rand_state_i2]);
                double p = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
                if(p < std::exp((beta_1 - beta_2)*(energy_1 - energy_2))){ // Accept the flip
                    double i_temp = beta_is[rand_state_i1];
                    beta_is[rand_state_i1] = beta_is[rand_state_i2];
                    beta_is[rand_state_i2] = i_temp;
                    // std::cout<<
                }
            }
        }
        // Run this when time to take a sample
        // Check that the local moves are still in snyc with a global check
        for(int state_i = 0; state_i < all_betas.size(); state_i++){
            check_state_bookkeeping(states[state_i],state_bookkeepings[state_i], const_info);
        }
        if(warmup_done){
            num_samples_taken += 1;
            std::cout << num_samples_taken << "/"<< NUM_SAMPLES<<" samples taken.\n";
            for(int state_i = 0; state_i < all_betas.size(); state_i++){
                double beta = all_betas[beta_is[state_i]];
                if(std::abs(beta - MAX_BETA) < 0.001){
                    outfile << state_csv_string(states[state_i], state_bookkeepings[state_i], const_info) << "\n"; // For some reason this is writing strange results at times?
                    std::cout << " energy: " << std::get<energyStateBookkeepingTypeIndex>(state_bookkeepings[state_i]) << " q: " << std::get<qStateBookkeepingTypeIndex>(state_bookkeepings[state_i])
                     << " sigma_C: " << std::get<sigma_C_StateTypeIndex>(states[state_i]) << " sigma_I: " << std::get<sigma_I_StateTypeIndex>(states[state_i])  << " ";
                    
                }
            }
        }else{
            // Check if the warmup is done (if we have an increase in energy in the beta = 1 chain)
            for(int state_i = 0; state_i < all_betas.size(); state_i++){
                double beta = all_betas[beta_is[state_i]];
                if(std::abs(beta - MAX_BETA) < 0.001){
                    // print(state_bookkeepings[state_i]);
                    double energy = std::get<energyStateBookkeepingTypeIndex>(state_bookkeepings[state_i]);
                    if(energy > energy_history[num_warmups_taken % SAMPLES_BACK_BURNIN_CHECK]){
                        warmup_done = true;
                    }else{
                        energy_history[num_warmups_taken % SAMPLES_BACK_BURNIN_CHECK] = energy;
                        num_warmups_taken += 1;
                        std::cout << " energy: " << std::get<energyStateBookkeepingTypeIndex>(state_bookkeepings[state_i]) << " q: " << std::get<qStateBookkeepingTypeIndex>(state_bookkeepings[state_i])
                     << " sigma_C: " << std::get<sigma_C_StateTypeIndex>(states[state_i]) << " sigma_I: " << std::get<sigma_I_StateTypeIndex>(states[state_i])  << " ";
                        std::cout << "Warming up, " << num_warmups_taken << " warmup samples taken.\n";
                    }
                }
            }
        }
    }
    // Close file
    std::cout << "Samples written to " << output_filename << "\n";
    outfile.close();
}

// Initialize the state given the initial partition
StateType initialize_state(std::vector<int>& partition_init, std::mt19937& rng){
    int q = *std::max_element(partition_init.begin(), partition_init.end());
    q += 1;
    int n = partition_init.size();

    std::normal_distribution<double> unit_normal_dist(0, 1);
    std::uniform_real_distribution<double> uniform_01_dist(0, 1);

    double p = uniform_01_dist(rng);
    double sigma_C_init = p;
    double sigma_I_init = p;

    std::vector<double> community_status_init;
    for(int r = 0; r < q; r++){
        double rand_community_status = sigma_C_init*unit_normal_dist(rng);
        community_status_init.push_back(rand_community_status);
    }

    std::vector<double> total_status_init;
    for(int i = 0; i < n; i++){
        double rand_individual_status = sigma_I_init*unit_normal_dist(rng);
        double rand_total_status = rand_individual_status + community_status_init[partition_init[i]];
        total_status_init.push_back(rand_total_status);
    }

    double nu_init = uniform_01_dist(rng);

    StateType state = std::make_tuple(partition_init, sigma_C_init, sigma_I_init, community_status_init, total_status_init, nu_init);
    
    return state;
}

// Implement the proposed swap. Also return the change in the log-likelihood, relevant for deciding whether to in fact accept this swap
// Goal is amortized (average) O(n/q) time complexity
double implement_proposal(std::string var_name, int var_index, double new_val, double old_val, int delta_q, StateType& state, StateBookkeepingType& state_bookkeeping, ConstInfoType& const_info, std::mt19937& rng){
    // Unpack the tuples
    auto& [partition, 
        sigma_C, 
        sigma_I,
        community_status, 
        total_status,
        nu] = state;

    auto& [q, 
        node_counts, 
        group_members,
        edge_counts, 
        edge_counts_sum,
        degree_counts_by_group, 
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_c_b_sigmaI,
        H_c_G_sigmaC,
        H_sigmaC,
        H_sigmaI,
        H_Abar_G_M_b, 
        H_k_G_eta, 
        H_eta_G_M_nb,
        H_M_G_q, 
        H_b_G_nb, 
        H_nb_G_q, 
        H_q, 
        energy] = state_bookkeeping;

    auto& [n,
        m,
        wins, 
        ties, 
        losses,
        neighbors,
        degree_list // std::vector<int> , \vec{k}
        ] = const_info;

    double deltaH = 0;
    // std::cout << var_name << " " << var_index << " " << old_val << " " << new_val << " " << delta_q << "\n";
    // std::cout << "partition["<<var_index<<"] = "<<old_val<<" -> "<<new_val<<"\n";
    // std::cout << "Initial ";
    // print(node_counts);
    // print(partition);
    // H_A_G_Abar_s_nu + H_nu + H_s_G_c_b_sigmaI + H_c_G_sigmaC + H_sigmaC + H_sigmaI
    if(var_name == "sigma_C"){
        // This will impact H_c_G_sigmaC and H_sigmaC, the H_c_G_sigmaC change needs to be accept/rejected
        sigma_C = new_val;
        // state_bookkeeping change
        double H_sigmaC_new = std::log(M_PI*MEAN_SIGMA) + std::pow(sigma_C/MEAN_SIGMA,2)/M_PI;
        double H_c_G_sigmaC_new = q*std::log(sigma_C) + q*0.5*std::log(2*M_PI) + std::pow(old_val/sigma_C,2)*(H_c_G_sigmaC - q*std::log(old_val) - q*0.5*std::log(2*M_PI));
        deltaH = H_c_G_sigmaC_new - H_c_G_sigmaC + H_sigmaC_new - H_sigmaC; // Change in likelihood
        H_c_G_sigmaC = H_c_G_sigmaC_new;
        H_sigmaC = H_sigmaC_new;
        energy = H_A_G_Abar_s_nu + H_nu + H_s_G_c_b_sigmaI + H_c_G_sigmaC + H_sigmaC + H_sigmaI + H_Abar_G_M_b + H_k_G_eta + H_eta_G_M_nb + H_M_G_q + H_b_G_nb + H_nb_G_q + H_q;
    }
    if(var_name == "sigma_I"){
        // This will impact H_s_G_c_b_sigmaI and H_sigmaI, the H_s_G_c_b_sigmaI change needs to be accept/rejected
        // state change
        sigma_I = new_val;
        // state_bookkeeping change
        double H_sigmaI_new = std::log(M_PI*MEAN_SIGMA) + std::pow(sigma_I/MEAN_SIGMA,2)/M_PI;
        double H_s_G_c_b_sigmaI_new = n*std::log(sigma_I) + n*0.5*std::log(2*M_PI) + std::pow(old_val/sigma_I,2)*(H_s_G_c_b_sigmaI - n*std::log(old_val) - n*0.5*std::log(2*M_PI));
        deltaH = H_s_G_c_b_sigmaI_new - H_s_G_c_b_sigmaI + H_sigmaI_new - H_sigmaI; // Change in likelihood
        H_s_G_c_b_sigmaI = H_s_G_c_b_sigmaI_new;
        H_sigmaI = H_sigmaI_new;
        energy = H_A_G_Abar_s_nu + H_nu + H_s_G_c_b_sigmaI + H_c_G_sigmaC + H_sigmaC + H_sigmaI + H_Abar_G_M_b + H_k_G_eta + H_eta_G_M_nb + H_M_G_q + H_b_G_nb + H_nb_G_q + H_q;
    }
    if(var_name == "community_status"){
        // This will impact H_s_G_c_b_sigmaI and H_c_G_sigmaC, the H_s_G_c_b_sigmaI change needs to be accept/rejected
        // state change
        int r = var_index;
        community_status[r] = new_val;
        // state_bookkeeping change
        double H_c_G_sigmaC_new = H_c_G_sigmaC;
        H_c_G_sigmaC_new -= 0.5*std::pow(old_val/sigma_C,2) + std::log(sigma_C) + 0.5*std::log(2*M_PI);
        H_c_G_sigmaC_new += 0.5*std::pow(community_status[r]/sigma_C,2) + std::log(sigma_C) + 0.5*std::log(2*M_PI);
        double H_s_G_c_b_sigmaI_new = H_s_G_c_b_sigmaI;
        for(int i : group_members[r]){
            H_s_G_c_b_sigmaI_new -= 0.5*std::pow((total_status[i]-old_val)/sigma_I,2) + std::log(sigma_I) + 0.5*std::log(2*M_PI);
            H_s_G_c_b_sigmaI_new += 0.5*std::pow((total_status[i]-community_status[partition[i]])/sigma_I,2) + std::log(sigma_I) + 0.5*std::log(2*M_PI);
        }
        deltaH = H_s_G_c_b_sigmaI_new - H_s_G_c_b_sigmaI + H_c_G_sigmaC_new - H_c_G_sigmaC;
        H_s_G_c_b_sigmaI = H_s_G_c_b_sigmaI_new;
        H_c_G_sigmaC = H_c_G_sigmaC_new;
        energy = H_A_G_Abar_s_nu + H_nu + H_s_G_c_b_sigmaI + H_c_G_sigmaC + H_sigmaC + H_sigmaI + H_Abar_G_M_b + H_k_G_eta + H_eta_G_M_nb + H_M_G_q + H_b_G_nb + H_nb_G_q + H_q;
    }
    if(var_name == "total_status"){
        // This will impact H_A_G_Abar_s_nu and H_s_G_c_b_sigmaI, the H_A_G_Abar_s_nu change needs to be accept/rejected
        // state change
        int i = var_index;
        total_status[i] = new_val;
        // state_bookkeeping change
        double H_s_G_c_b_sigmaI_new = H_s_G_c_b_sigmaI;
        H_s_G_c_b_sigmaI_new -= 0.5*std::pow((old_val-community_status[partition[i]])/sigma_I,2) + std::log(sigma_I) + 0.5*std::log(2*M_PI);
        H_s_G_c_b_sigmaI_new += 0.5*std::pow((total_status[i]-community_status[partition[i]])/sigma_I,2) + std::log(sigma_I) + 0.5*std::log(2*M_PI);
        double H_A_G_Abar_s_nu_new = H_A_G_Abar_s_nu;
        for(auto& [j, Aij] : wins[i]){
            double old_score_diff = old_val - total_status[j];
            double score_diff = total_status[i] - total_status[j];
            H_A_G_Abar_s_nu_new -= Aij*(log_denom(old_score_diff,nu) - old_score_diff/2); // -log P(i beat j)
            H_A_G_Abar_s_nu_new += Aij*(log_denom(score_diff,nu) - score_diff/2); // -log P(i beat j)
        }
        for(auto& [j, Aij] : ties[i]){
            double old_score_diff = old_val - total_status[j];
            double score_diff = total_status[i] - total_status[j];
            H_A_G_Abar_s_nu_new -= Aij*(log_denom(old_score_diff,nu) - std::log(2*nu)); // -log P(i ties j)
            H_A_G_Abar_s_nu_new += Aij*(log_denom(score_diff,nu) - std::log(2*nu)); // -log P(i ties j)
        }
        for(auto& [j, Aij] : losses[i]){
            double old_score_diff = total_status[j] - old_val; // Flip the score difference
            double score_diff =  total_status[j] - total_status[i];
            H_A_G_Abar_s_nu_new -= Aij*(log_denom(old_score_diff,nu) - old_score_diff/2); // -log P(j beat i)
            H_A_G_Abar_s_nu_new += Aij*(log_denom(score_diff,nu) - score_diff/2); // -log P(j beat i)
        }
        deltaH = H_A_G_Abar_s_nu_new - H_A_G_Abar_s_nu + H_s_G_c_b_sigmaI_new - H_s_G_c_b_sigmaI; // Change in likelihood
        H_A_G_Abar_s_nu = H_A_G_Abar_s_nu_new;
        H_s_G_c_b_sigmaI = H_s_G_c_b_sigmaI_new;
        energy = H_A_G_Abar_s_nu + H_nu + H_s_G_c_b_sigmaI + H_c_G_sigmaC + H_sigmaC + H_sigmaI + H_Abar_G_M_b + H_k_G_eta + H_eta_G_M_nb + H_M_G_q + H_b_G_nb + H_nb_G_q + H_q;
    }
    if(var_name == "nu"){
        // Pretty sure that we just need to fully recalculate H_A_G_Abar_s_nu
        // This will impact H_A_G_Abar_s_nu and H_nu, the H_A_G_Abar_s_nu change needs to be accept/rejected
        // state change
        nu = new_val;
        // state_bookkeeping change
        double H_nu_new = 2*std::log(1+nu);
        double H_A_G_Abar_s_nu_new = 0;
        for(int i = 0; i < n; i++){
            for(auto& [j, Aij] : wins[i]){
                double score_diff = total_status[i] - total_status[j];
                H_A_G_Abar_s_nu_new += Aij*(log_denom(score_diff,nu) - score_diff/2); // -log P(i beat j)
            }
            for(auto& [j, Aij] : ties[i]){
                if(i < j){ // Only count the ties once
                    double score_diff = total_status[i] - total_status[j];
                    H_A_G_Abar_s_nu_new += Aij*(log_denom(score_diff,nu) - std::log(2*nu)); // -log P(i ties j)
                }
            }
            // Note that we do not need to include the losses since they are captured by the wins of the opponent
        }
        deltaH = H_A_G_Abar_s_nu_new - H_A_G_Abar_s_nu + H_nu_new - H_nu; // Change in likelihood
        H_A_G_Abar_s_nu = H_A_G_Abar_s_nu_new;
        H_nu = H_nu_new;
        energy = H_A_G_Abar_s_nu + H_nu + H_s_G_c_b_sigmaI + H_c_G_sigmaC + H_sigmaC + H_sigmaI + H_Abar_G_M_b + H_k_G_eta + H_eta_G_M_nb + H_M_G_q + H_b_G_nb + H_nb_G_q + H_q;
    }
    if(var_name == "partition"){
        int i = var_index;
        int r_old = old_val;
        int r_new = new_val;
        partition[i] = r_new;
        double energy_old = energy; // For returning the change in energy
        // Check if the proposed move will change the number of communities
        if(delta_q == -1){ // This is the last member of a group, so we should shift the existing communities and entirely recompute.
            // std::cout << "Removing color "<<old_val<<", shifting label names\n";
            for(int i = 0; i < n; i++){
                if(partition[i] >= old_val){
                    partition[i] -= 1;
                }
            }
            for(int r = 0; r < q; r++){
                if(r > old_val){
                    community_status[r-1] = community_status[r];
                }
            }
            // Remove last element of community_status
            community_status.pop_back();
            // Recompute the bookkeeping from scratch (maybe this could be done more efficiently)
            state_bookkeeping = get_state_bookkeeping(state, const_info);
            double energy = std::get<energyStateBookkeepingTypeIndex>(state_bookkeeping);
            deltaH = energy - energy_old;
        }else if(delta_q == 1){ // This is a new color
            // std::cout << "Adding color "<<new_val<<", shifting label names\n";
            for(int i = 0; i < n; i++){ // Shift everything up except for what we just added.
                if(partition[i] >= new_val && i != var_index){
                    partition[i] += 1;
                }
            }
            community_status.push_back(0);
            for(int r = 0; r <= q; r++){
                if(r >= new_val){
                    community_status[r] = community_status[r - 1];
                }
            }
            // Randomly initialize the status of the new community
            std::normal_distribution<double> unit_normal_dist(0, 1);
            community_status[new_val] = sigma_C*unit_normal_dist(rng);
            // Recompute the bookkeeping from scratch (maybe this could be done more efficiently)
            state_bookkeeping = get_state_bookkeeping(state, const_info);
            double energy = std::get<energyStateBookkeepingTypeIndex>(state_bookkeeping);
            deltaH = energy - energy_old;
        }else{ // Compute the local changes
            // std::cout << "Implementing local change\n";
            // Removing old contributions to the energy
            // H_s_G_c_b_sigmaI
            H_s_G_c_b_sigmaI -= 0.5*std::pow((total_status[i]-community_status[r_old])/sigma_I,2) + std::log(sigma_I) + 0.5*std::log(2*M_PI); 
            // H_Abar_G_M_b 
            H_Abar_G_M_b -= std::lgamma(edge_counts_sum[r_old] + 1); // log(M[r]!)
            H_Abar_G_M_b += (edge_counts[r_old][r_old]/2*std::log(2) + std::lgamma(edge_counts[r_old][r_old]/2 + 1)); // log(M[r][r]!!) = log(2^(M[r][r]/2)M[r][r]!)
            H_Abar_G_M_b -= std::lgamma(edge_counts_sum[r_new] + 1); // log(M[r]!)
            H_Abar_G_M_b += (edge_counts[r_new][r_new]/2*std::log(2) + std::lgamma(edge_counts[r_new][r_new]/2 + 1)); // log(M[r][r]!!) = log(2^(M[r][r]/2)M[r][r]!)
            for(int s = 0; s < q; s++){ // log Omega, remove the 2*(q - 1) - 1 entries that will be impacted
                if(s != r_old){
                    H_Abar_G_M_b += std::lgamma(edge_counts[r_old][s] + 1); // log(M[r][s]!)
                    if(s != r_new){
                        H_Abar_G_M_b += std::lgamma(edge_counts[r_new][s] + 1); // log(M[r][s]!)
                    }
                }
            }
            // H_k_G_eta
            H_k_G_eta -= std::lgamma(node_counts[r_old] + 1); // log(nb[r]!)
            // H_k_G_eta += std::lgamma(degree_counts_by_group[r_old][degree_list[i]] + 1); // log(eta[r][k]!) [in terms of vector]
            for(auto& [k, count] : degree_counts_by_group[r_old]){
                H_k_G_eta += std::lgamma(count + 1); // log(eta[r][k]!)
            }
            H_k_G_eta -= std::lgamma(node_counts[r_new] + 1); // log(nb[r]!)
            // H_k_G_eta += std::lgamma(degree_counts_by_group[r_new][degree_list[i]] + 1); // log(eta[r][k]!) [in terms of vector]
            for(auto& [k, count] : degree_counts_by_group[r_new]){
                H_k_G_eta += std::lgamma(count + 1); // log(eta[r][k]!)
            }
            // H_eta_G_M_nb
            H_eta_G_M_nb -= log_q_vals[edge_counts_sum[r_old]][node_counts[r_old]];
            H_eta_G_M_nb -= log_q_vals[edge_counts_sum[r_new]][node_counts[r_new]];
            
            // H_b_G_nb
            H_b_G_nb += std::lgamma(node_counts[r_old] + 1); // log(nb[r]!)
            H_b_G_nb += std::lgamma(node_counts[r_new] + 1); // log(nb[r]!)


            // Updating the bookkeeping
            // node_counts
            node_counts[r_old] -= 1;
            node_counts[r_new] += 1;
            // group_members
            group_members[r_old].erase(i);
            group_members[r_new].insert(i);
            // edge_counts
            // edge_counts_sum
            for(auto& [j, Aij] : neighbors[i]){
                edge_counts[r_old][partition[j]] -= Aij;
                edge_counts[partition[j]][r_old] -= Aij;
                edge_counts[r_new][partition[j]] += Aij;
                edge_counts[partition[j]][r_new] += Aij;
                edge_counts_sum[r_old] -= Aij;
                edge_counts_sum[r_new] += Aij;
            }
            // degree_counts_by_group
            degree_counts_by_group[r_old][degree_list[i]] -= 1;
            if(degree_counts_by_group[r_old][degree_list[i]] == 0){
                degree_counts_by_group[r_old].erase(degree_list[i]);
            }
            if(degree_counts_by_group[r_new].find(degree_list[i]) == degree_counts_by_group[r_new].end()){
                degree_counts_by_group[r_new][degree_list[i]] = 1;
            }else{
                degree_counts_by_group[r_new][degree_list[i]] += 1;
            }

            // Adding terms back to the energy
            // Adding new contributions to the energy
            // H_s_G_c_b_sigmaI
            H_s_G_c_b_sigmaI += 0.5*std::pow((total_status[i]-community_status[r_new])/sigma_I,2) + std::log(sigma_I) + 0.5*std::log(2*M_PI); 
            // H_Abar_G_M_b
            H_Abar_G_M_b += std::lgamma(edge_counts_sum[r_old] + 1); // log(M[r]!)
            H_Abar_G_M_b -= (edge_counts[r_old][r_old]/2*std::log(2) + std::lgamma(edge_counts[r_old][r_old]/2 + 1)); // log(M[r][r]!!) = log(2^(M[r][r]/2)M[r][r]!)
            H_Abar_G_M_b += std::lgamma(edge_counts_sum[r_new] + 1); // log(M[r]!)
            H_Abar_G_M_b -= (edge_counts[r_new][r_new]/2*std::log(2) + std::lgamma(edge_counts[r_new][r_new]/2 + 1)); // log(M[r][r]!!) = log(2^(M[r][r]/2)M[r][r]!)
            
            for(int s = 0; s < q; s++){ // log Omega, reintroduce the 2*(q - 1) - 1 entries that will be impacted
                if(s != r_old){
                    H_Abar_G_M_b -= std::lgamma(edge_counts[r_old][s] + 1); // log(M[r][s]!)
                    if(s != r_new){
                        H_Abar_G_M_b -= std::lgamma(edge_counts[r_new][s] + 1); // log(M[r][s]!)
                    }
                }
            }
            // H_k_G_eta
            H_k_G_eta += std::lgamma(node_counts[r_old] + 1); // log(nb[r]!)
            // H_k_G_eta -= std::lgamma(degree_counts_by_group[r_old][degree_list[i]] + 1); // log(eta[r][k]!) [in terms of vector]
            for(auto& [k, count] : degree_counts_by_group[r_old]){
                H_k_G_eta -= std::lgamma(count + 1); // log(eta[r][k]!)
            }
            H_k_G_eta += std::lgamma(node_counts[r_new] + 1); // log(nb[r]!)
            // H_k_G_eta -= std::lgamma(degree_counts_by_group[r_new][degree_list[i]] + 1); // log(eta[r][k]!) [in terms of vector]
            for(auto& [k, count] : degree_counts_by_group[r_new]){
                H_k_G_eta -= std::lgamma(count + 1); // log(eta[r][k]!)
            }
            // H_eta_G_M_nb
            H_eta_G_M_nb += log_q_vals[edge_counts_sum[r_old]][node_counts[r_old]];
            H_eta_G_M_nb += log_q_vals[edge_counts_sum[r_new]][node_counts[r_new]];
            
            // H_b_G_nb
            H_b_G_nb -= std::lgamma(node_counts[r_old] + 1); // log(nb[r]!)
            H_b_G_nb -= std::lgamma(node_counts[r_new] + 1); // log(nb[r]!)
            // energy
            energy = 0;
            energy += H_A_G_Abar_s_nu + H_nu + H_s_G_c_b_sigmaI + H_c_G_sigmaC + H_sigmaC + H_sigmaI;
            energy += H_Abar_G_M_b;
            energy += H_k_G_eta + H_eta_G_M_nb + H_M_G_q;
            energy += H_b_G_nb + H_nb_G_q + H_q;
            deltaH = energy - energy_old;
        }
    }
    return deltaH;
}

// Check that the state and its bookkeeping are in alignment. If not print the error and exit.
void check_state_bookkeeping(StateType& state, StateBookkeepingType& state_bookkeeping, ConstInfoType& const_info){
    // Unpack the tuples
    auto& [partition, 
        sigma_C, 
        sigma_I,
        community_status, 
        total_status,
        nu] = state;

    auto& [q, 
        node_counts, 
        group_members,
        edge_counts, 
        edge_counts_sum,
        degree_counts_by_group, 
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_c_b_sigmaI,
        H_c_G_sigmaC,
        H_sigmaC,
        H_sigmaI,
        H_Abar_G_M_b, 
        H_k_G_eta, 
        H_eta_G_M_nb,
        H_M_G_q, 
        H_b_G_nb, 
        H_nb_G_q, 
        H_q, 
        energy] = state_bookkeeping;

    auto& [n,
        m,
        wins, 
        ties, 
        losses,
        neighbors, // std::vector<std::vector<int>>
        degree_list // std::vector<int> , \vec{k}
        ] = const_info;
    
    // Regenerate the state_bookkeeping from the state
    StateBookkeepingType state_bookkeeping_actual = get_state_bookkeeping(state, const_info);

    auto& [q_actual, 
        node_counts_actual, 
        group_members_actual,
        edge_counts_actual, 
        edge_counts_sum_actual,
        degree_counts_by_group_actual, 
        H_A_G_Abar_s_nu_actual,
        H_nu_actual,
        H_s_G_c_b_sigmaI_actual,
        H_c_G_sigmaC_actual,
        H_sigmaC_actual,
        H_sigmaI_actual,
        H_Abar_G_M_b_actual, 
        H_k_G_eta_actual, 
        H_eta_G_M_nb_actual,
        H_M_G_q_actual, 
        H_b_G_nb_actual, 
        H_nb_G_q_actual, 
        H_q_actual, 
        energy_actual] = state_bookkeeping_actual;

    // Check for equality
    if(q != q_actual){
        std::cout << "Error: bookkeeping value q = "<<q<<" actual value q = "<<q_actual<<"\n";
        exit(0);
    }
    for(int r = 0; r < q; r++){
        if(node_counts[r] != node_counts_actual[r]){
            std::cout << "Error: bookkeeping value node_counts["<<r<<"] = "<<node_counts[r]<<" actual value node_counts["<<r<<"] = "<<node_counts_actual[r]<<"\n";
            exit(0);
        }
    }
    for(int r = 0; r < q; r++){
        for(int i_actual : group_members_actual[r]){
            if(std::find(group_members[r].begin(), group_members[r].end(), i_actual) == group_members[r].end()){
                std::cout << "Error: student "<<i_actual<<" from actual group "<<r<<" not found in bookkeeping group_members["<<r<<"]\n";
                exit(0);
            }
        }
        for(int i : group_members[r]){
            if(std::find(group_members_actual[r].begin(), group_members_actual[r].end(), i) == group_members_actual[r].end()){
                std::cout << "Error: student "<<i<<" from bookkeeping group "<<r<<" not found in the actual group_members["<<r<<"]\n";
                exit(0);
            }
        }
    }
    for(int r = 0; r < q; r++){
        for(int s = 0; s < q; s++){
            if(edge_counts[r][s] != edge_counts_actual[r][s]){
                std::cout << "Error: bookkeeping value edge_counts["<<r<<"]["<<s<<"] = "<<edge_counts[r][s]<<" actual value edge_counts["<<r<<"]["<<s<<"] = "<<edge_counts_actual[r][s]<<"\n";
                exit(0);
            }
        }
    }
    for(int r = 0; r < q; r++){
        if(edge_counts_sum[r] != edge_counts_sum[r]){
            std::cout << "Error: bookkeeping value edge_counts_sum["<<r<<"] = "<<edge_counts_sum[r]<<" actual value edge_counts_sum["<<r<<"] = "<<edge_counts_sum[r]<<"\n";
            exit(0);
        }
    }
    for(int r = 0; r < q; r++){
        if(degree_counts_by_group[r].size() != degree_counts_by_group_actual[r].size()){
            std::cout << "Error: in group "<<r<<" bookkeeping number of distinct degrees = "<<degree_counts_by_group[r].size()<<" actual number of distinct degrees = "<<degree_counts_by_group_actual[r].size()<<"\n";
            exit(0);
        }
        for(auto& [k, count]: degree_counts_by_group[r]){
            if(count != degree_counts_by_group_actual[r][k]){
                std::cout << "Error: bookkeeping value degree_counts_by_group["<<r<<"]["<<k<<"] = "<<count<<" actual value degree_counts_by_group["<<r<<"]["<<k<<"] = "<<degree_counts_by_group_actual[r][k]<<"\n";
                exit(0);
            }
        }
    }
    // Direction
    if(std::abs(H_A_G_Abar_s_nu - H_A_G_Abar_s_nu_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_A_G_Abar_s_nu = "<<H_A_G_Abar_s_nu<<" actual value H_A_G_Abar_s_nu = "<<H_A_G_Abar_s_nu_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_nu - H_nu_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_nu = "<<H_nu<<" actual value H_nu = "<<H_nu_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_s_G_c_b_sigmaI - H_s_G_c_b_sigmaI_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_s_G_c_b_sigmaI = "<<H_s_G_c_b_sigmaI<<" actual value H_s_G_c_b_sigmaI = "<<H_s_G_c_b_sigmaI_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_c_G_sigmaC - H_c_G_sigmaC_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_c_G_sigmaC = "<<H_c_G_sigmaC<<" actual value H_c_G_sigmaC = "<<H_c_G_sigmaC_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_sigmaC - H_sigmaC_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_sigmaC = "<<H_sigmaC<<" actual value H_sigmaC = "<<H_sigmaC_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_sigmaI - H_sigmaI_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_sigmaI = "<<H_sigmaI<<" actual value H_sigmaI = "<<H_sigmaI_actual<<"\n";
        exit(0);
    }
    // Community
    if(std::abs(H_Abar_G_M_b - H_Abar_G_M_b_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_Abar_G_M_b = "<<H_Abar_G_M_b<<" actual value H_Abar_G_M_b = "<<H_Abar_G_M_b_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_k_G_eta - H_k_G_eta_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_k_G_eta = "<<H_k_G_eta<<" actual value H_k_G_eta = "<<H_k_G_eta_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_eta_G_M_nb - H_eta_G_M_nb_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_eta_G_M_nb = "<<H_eta_G_M_nb<<" actual value H_eta_G_M_nb = "<<H_eta_G_M_nb_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_M_G_q - H_M_G_q_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_M_G_q = "<<H_M_G_q<<" actual value H_M_G_q = "<<H_M_G_q_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_b_G_nb - H_b_G_nb_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_b_G_nb = "<<H_b_G_nb<<" actual value H_b_G_nb = "<<H_b_G_nb_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_nb_G_q - H_nb_G_q_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_nb_G_q = "<<H_nb_G_q<<" actual value H_nb_G_q = "<<H_nb_G_q_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_q - H_q_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_q = "<<H_q<<" actual value H_q = "<<H_q_actual<<"\n";
        exit(0);
    }
    if(std::abs(energy - energy_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value energy = "<<energy<<" actual value energy = "<<energy_actual<<"\n";
        exit(0);
    }
    // If checks are passed, update with the actual values (to stop floating point drift)
    H_A_G_Abar_s_nu = H_A_G_Abar_s_nu_actual;
    H_nu = H_nu_actual;
    H_s_G_c_b_sigmaI = H_s_G_c_b_sigmaI_actual;
    H_c_G_sigmaC = H_c_G_sigmaC_actual;
    H_sigmaC = H_sigmaC_actual;
    H_sigmaI = H_sigmaI_actual; 
    H_Abar_G_M_b = H_Abar_G_M_b_actual;
    H_k_G_eta = H_k_G_eta_actual;
    H_eta_G_M_nb = H_eta_G_M_nb_actual;
    H_M_G_q = H_M_G_q_actual;
    H_b_G_nb = H_b_G_nb_actual;
    H_nb_G_q = H_nb_G_q_actual;
    H_q = H_q_actual;
    energy = energy_actual;
}

// Generate proposed MCMC move: var_name, var_index, new_val, old_val
std::tuple<std::string, int, double, double, int> generate_proposal(StateType& state, StateBookkeepingType& state_bookkeeping, std::mt19937& rng){
    // Unpack the tuples
    auto& [partition, 
        sigma_C, 
        sigma_I,
        community_status, 
        total_status,
        nu] = state;

    auto& [q, 
        node_counts, 
        group_members,
        edge_counts, 
        edge_counts_sum,
        degree_counts_by_group, 
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_c_b_sigmaI,
        H_c_G_sigmaC,
        H_sigmaC,
        H_sigmaI,
        H_Abar_G_M_b, 
        H_k_G_eta, 
        H_eta_G_M_nb,
        H_M_G_q, 
        H_b_G_nb, 
        H_nb_G_q, 
        H_q, 
        energy] = state_bookkeeping;

    // Initializing the variables to be returned
    std::string var_name;
    int var_index;
    double new_val;
    double old_val;
    int delta_q; 
    
    // Which variable will we change?
    int total_variables = partition.size() + 1 + 1 + community_status.size() + total_status.size() + 1;
    std::uniform_int_distribution<int> rand_int(0, total_variables - 1);
    int rand_var_index = rand_int(rng);

    // By how much?
    std::normal_distribution<double> unit_normal_dist(0, 1);
    double z = unit_normal_dist(rng); // Element of randomness used to preturb continuous variables
    if(rand_var_index < partition.size()){ // Partition swap
        var_name = "partition";
        var_index = rand_var_index;
        old_val = partition[var_index];
        int q = *std::max_element(partition.begin(), partition.end());
        q += 1;
        if(FIX_Q){
            std::uniform_int_distribution<int> r_dist(0, q - 2); // Exclude possiblity of the same color
            int new_val = r_dist(rng);
            if(new_val >= old_val){ // Uniform over new_val != old_val
                new_val += 1; 
            }
            return std::make_tuple(var_name, var_index, new_val, old_val, delta_q);
        }else{
            std::uniform_int_distribution<int> r_dist(0, q - 1); // Exclude possiblity of the same color, include possibility of the new color q (in addition to 0,...,q-1)
            int new_val = r_dist(rng);
            if(new_val >= old_val){ // Uniform over new_val != old_val
                new_val += 1; 
            }
            // Compute how this will change the number of groups
            if(node_counts[old_val] == 1){ // This group is being removed
                if(new_val == q){ // A new group is being generated
                    new_val = old_val; // Just keep the old group (this would just be a relabeling)
                    delta_q = 0; // No change
                }else{
                    delta_q = -1; // A group is removed
                }
            }else{ // Old group is kept
                if(new_val == q){ // A new group is being generated
                    delta_q = 1; // A group is added
                }
            }
            return std::make_tuple(var_name, var_index, new_val, old_val, delta_q);
        }
    }else{
        rand_var_index -= partition.size();
    }
    if(rand_var_index < 1){ // Shifting sigma_C
        var_name = "sigma_C";
        var_index = 0;
        old_val = sigma_C;
        new_val = old_val + SIGMA_C_TILDE_STEP_SIZE*z; // Apply the change
        if(new_val < 0){ // sigma_C can't go negative
            new_val *= -1;
        }
        return std::make_tuple(var_name, var_index, new_val, old_val, 0);
    }else{
        rand_var_index -= 1;
    }
    if(rand_var_index < 1){ // Shifting sigma_I
        var_name = "sigma_I";
        var_index = 0;
        old_val = sigma_I;
        new_val = old_val + SIGMA_I_TILDE_STEP_SIZE*z; // Apply the change
        if(new_val < 0){ // sigma_I can't go negative
            new_val *= -1;
        }
        return std::make_tuple(var_name, var_index, new_val, old_val, 0);
    }else{
        rand_var_index -= 1;
    }
    if(rand_var_index < community_status.size()){ // Shifting community_status
        var_name = "community_status";
        var_index = rand_var_index;
        old_val = community_status[var_index];
        new_val = old_val + COMMUNITY_STATUS_TILDE_STEP_SIZE*z; // Apply the change
        return std::make_tuple(var_name, var_index, new_val, old_val, 0);
    }else{
        rand_var_index -= community_status.size();
    }
    if(rand_var_index < total_status.size()){ // Shifting total_status
        var_name = "total_status";
        var_index = rand_var_index;
        old_val = total_status[var_index];
        new_val = old_val + TOTAL_STATUS_TILDE_STEP_SIZE*z; // Apply the change
        
        return std::make_tuple(var_name, var_index, new_val, old_val, 0);
    }else{
        rand_var_index -= total_status.size();
    }
    if(rand_var_index < 1){ // Shifting nu
        var_name = "nu";
        var_index = 0;
        old_val = nu;
        new_val = old_val + NU_TILDE_STEP_SIZE*z; // Apply the change
        if(new_val < 0){ // nu can't go negative
            new_val *= -1;
        }
        return std::make_tuple(var_name, var_index, new_val, old_val, 0);
    }else{
        rand_var_index -= 1;
    }
    std::cout << "Error: No var_name sampled.\n";
    exit(0);
    return std::make_tuple(var_name, var_index, new_val, old_val, delta_q);
}

// Compute the bookkeeping information (including the energy) from the MCMC state. Goal is to compute this in O(n) time.
StateBookkeepingType get_state_bookkeeping(StateType& state, ConstInfoType& const_info){
    // Unpack the tuples
    auto& [partition, 
        sigma_C, 
        sigma_I,
        community_status, 
        total_status,
        nu] = state;
    
    auto& [n,
        m,
        wins, 
        ties, 
        losses,
        neighbors, // std::vector<std::vector<int>>
        degree_list // std::vector<int> , \vec{k}
        ] = const_info;
    
    // Basic information
    int q = *std::max_element(partition.begin(), partition.end());
    q += 1;
    std::vector<int> node_counts;
    for(int r = 0; r < q; r++){
        node_counts.push_back(0);
    }
    for(int i = 0; i < n; i++){
        node_counts[partition[i]]++;
    }
    // Check that none of the groups have size 0 (this is not allowed in the encoding and we should be relabeling to avoid this)
    for(int r = 0; r < q; r++){
        if(node_counts[r] == 0){
            std::cout << "Error: Group " << r << " has 0 elements.\n";
            exit(0); 
        }
    }

    std::vector<std::unordered_set<int>> group_members;
    for(int r = 0; r < q; r++){
        std::unordered_set<int> group_members_r;
        group_members.push_back(group_members_r);
    }
    for(int i = 0; i < n; i++){
        group_members[partition[i]].insert(i);
    }

    // Get the edge counts
    std::vector<std::vector<int>> edge_counts;
    std::vector<int> edge_counts_sum;
    for(int r = 0; r < q; r++){
        std::vector<int> edge_counts_r;
        for(int r = 0; r < q; r++){
            edge_counts_r.push_back(0);
        }
        edge_counts.push_back(edge_counts_r);
        edge_counts_sum.push_back(0);
    }
    for(int i = 0; i < n; i++){
        for(auto& [j, Aij] : neighbors[i]){
            edge_counts[partition[i]][partition[j]] += Aij;
            // Increment the sums
            edge_counts_sum[partition[i]] += Aij;
        }
    }

    // Get the degree counts by group
    std::vector<std::unordered_map<int,int>> degree_counts_by_group;
    for(int r = 0; r < q; r++){
        std::unordered_map<int,int> degree_counts_by_group_r;
        degree_counts_by_group.push_back(degree_counts_by_group_r);
    }
    for(int i = 0; i < n; i++){
        if(degree_counts_by_group[partition[i]].find(degree_list[i]) == degree_counts_by_group[partition[i]].end()){
            degree_counts_by_group[partition[i]][degree_list[i]] = 0;
        }
        degree_counts_by_group[partition[i]][degree_list[i]]++;
    }
    // Direction
    double H_A_G_Abar_s_nu = 0;
    for(int i = 0; i < n; i++){
        for(auto& [j, Aij] : wins[i]){
            double score_diff = total_status[i] - total_status[j];
            H_A_G_Abar_s_nu += Aij*(log_denom(score_diff,nu) - score_diff/2); // -log P(i beat j)
        }
        for(auto& [j, Aij] : ties[i]){
            if(i < j){ // Only count the ties once
                double score_diff = total_status[i] - total_status[j];
                H_A_G_Abar_s_nu += Aij*(log_denom(score_diff,nu) - std::log(2*nu)); // -log P(i ties j)
            }
        }
        // Note that we do not need to include the losses since they are captured by the wins of the opponent
    }
    double H_nu = 2*std::log(1+nu);
    double H_s_G_c_b_sigmaI = 0;
    for(int i = 0; i < n; i++){
        H_s_G_c_b_sigmaI += 0.5*std::pow((total_status[i]-community_status[partition[i]])/sigma_I,2) + std::log(sigma_I) + 0.5*std::log(2*M_PI);
    }
    double H_c_G_sigmaC = 0;
    for(int r = 0; r < q; r++){
        H_c_G_sigmaC += 0.5*std::pow(community_status[r]/sigma_C,2) + std::log(sigma_C) + 0.5*std::log(2*M_PI);
    }
    double H_sigmaC = std::log(M_PI*MEAN_SIGMA) + std::pow(sigma_C/MEAN_SIGMA,2)/M_PI;
    double H_sigmaI = std::log(M_PI*MEAN_SIGMA) + std::pow(sigma_I/MEAN_SIGMA,2)/M_PI;

    // Community
    double H_q = std::log(n);
    double H_nb_G_q = log_binom(n - 1, q - 1);
    double H_b_G_nb = std::lgamma(n + 1); // log(n!)
    for(int r = 0; r < q; r++){
        H_b_G_nb -= std::lgamma(node_counts[r] + 1); // log(nb[r]!)
    }
    double H_M_G_q = log_binom(m + q*(q+1)/2 - 1, q*(q+1)/2 - 1);
    double H_eta_G_M_nb = 0;
    for(int r = 0; r < q; r++){
        H_eta_G_M_nb += log_q_vals[edge_counts_sum[r]][node_counts[r]];
    }
    
    double H_k_G_eta = 0;
    for(int r = 0; r < q; r++){
        H_k_G_eta += std::lgamma(node_counts[r] + 1); // log(nb[r]!)
        for(auto& [k, count] : degree_counts_by_group[r]){
            H_k_G_eta -= std::lgamma(count + 1); // log(eta[r][k]!)
        }
    }
    double H_Abar_G_M_b = 0; 
    for(int r = 0; r < q; r++){ // log Omega
        H_Abar_G_M_b += std::lgamma(edge_counts_sum[r] + 1); // log(M[r]!)
        H_Abar_G_M_b -= (edge_counts[r][r]/2*std::log(2) + std::lgamma(edge_counts[r][r]/2 + 1)); // log(M[r][r]!!) = log(2^(M[r][r]/2)M[r][r]!)
        for(int s = r + 1; s < q; s++){
            H_Abar_G_M_b -= std::lgamma(edge_counts[r][s] + 1); // log(M[r][s]!)
        }
    }
    for(int i = 0; i < n; i++){ // log Xi
        H_Abar_G_M_b -= log_factorial(degree_list[i]); // log(k[i]!)
        // H_Abar_G_M_b += log_factorial2(A[i,i]); // Assume no self edges
        for(auto& [j, Aij]: neighbors[i]){
            if(i < j){
                H_Abar_G_M_b += log_factorial(Aij); 
            }
        }
    }
    double energy = 0;
    energy += H_A_G_Abar_s_nu + H_nu + H_s_G_c_b_sigmaI + H_c_G_sigmaC + H_sigmaC + H_sigmaI;
    energy += H_Abar_G_M_b; // Network
    energy += H_k_G_eta + H_eta_G_M_nb + H_M_G_q; // degrees and edge_count
    energy += H_b_G_nb + H_nb_G_q + H_q; // labeling

    StateBookkeepingType state_bookkeeping = std::make_tuple(q, node_counts, group_members, edge_counts, edge_counts_sum, degree_counts_by_group, 
    H_A_G_Abar_s_nu, H_nu, H_s_G_c_b_sigmaI, H_c_G_sigmaC, H_sigmaC, H_sigmaI,
    H_Abar_G_M_b, H_k_G_eta, H_eta_G_M_nb, H_M_G_q, H_b_G_nb, H_nb_G_q, H_q, energy);
    return state_bookkeeping;
}

double log_binom(double a, double b){
    return std::lgamma(a + 1) - std::lgamma(b + 1) - std::lgamma(a - b + 1);
}

// Overflow protected version of log(e^(sij/2) + 2 nu + e^(sji/2)) for evaluating H(A|Abar,s,nu)
double log_denom(double score_diff, double nu){
    // Symmetric in the score difference, so just take absolute value
    score_diff = std::abs(score_diff);
    // log(e^(ds/2) + 2 nu + e^(-ds/2)) = log(e^(ds/2)(1 + 2 nu e^(-ds/2) + e^(-ds))) = ds/2 + log(1 + 2 nu e^(-ds/2) + e^(-ds)))
    return score_diff/2 + std::log(1 + 2*nu*std::exp(-score_diff/2) + std::exp(-score_diff));
}

double log_factorial(int n){
    return std::lgamma(n + 1);
}

double log_factorial2(int n){
    // Check that n is an even integer
    if(n % 2 != 0){
        std::cout << "Error: log_factorial2 called with odd integer.\n";
        exit(0);
    }
    return n/2*std::log(2) + std::lgamma(n/2 + 1);
}

// Parse an edgelist and community partition from a GML file.
std::pair<std::vector<std::pair<int, int>>, std::vector<int>> parseGML(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;

    std::vector<std::pair<int, int>> edgelist;
    std::vector<int> partition;

    // Check that we can open the file
    if (!file.is_open()) {
        std::cerr << "Error: unable to open file " << filename << std::endl;
        exit(0);
    }

    int source = -1, target = -1;
    bool insideEdge = false; // Flag if the next tokens should be read to refer to a new edge
    int currentNodeIndex = 0;
    int n = 0;

    // Process one line at a time
    while (std::getline(file, line)) {
        std::istringstream iss(line);

        std::string token;
        // Process one (space separated) token at a time
        while(iss >> token){
            if(token == "["){ // Reset edge flag
                insideEdge = false;
            }
            else if (token == "edge") { // Start reading edge information
                insideEdge = true;
                edgelist.emplace_back(); // Create a new edge pair
            }
            else if (token == "source") {
                iss >> source;
                edgelist[edgelist.size() - 1].first = source;
            }
            else if (token == "target") {
                iss >> target;
                edgelist[edgelist.size() - 1].second = target;
            }
            else if (token == "community") { // Read next token as a community label
                int label;
                iss >> label;
                partition.push_back(label);
            }
            else if (token == "label") { // Count the number of nodes in case no initial communities are given.
                n++;
            }
        }
    }
    // For now just put everybody in one group. TODO: if there is no given community, we should produce a random initial assignment (pass rng)
    if(partition.size() == 0){
        for(int i = 0; i < n; i++){
            partition.push_back(0);
        }
    }

    file.close();
    return {edgelist, partition};
}

double log_sum_exp(double a, double b){
    double a_b_max = std::max(a,b);
    return a_b_max + std::log(std::exp(a - a_b_max) + std::exp(b - a_b_max));
}

// Write to file
std::string state_csv_string(StateType& state, StateBookkeepingType& state_bookkeeping, ConstInfoType& const_info){
    // Unpack the tuples
    auto& [partition, 
        sigma_C, 
        sigma_I,
        community_status, 
        total_status,
        nu] = state;

    auto& [q, 
        node_counts, 
        group_members,
        edge_counts, 
        edge_counts_sum,
        degree_counts_by_group, 
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_c_b_sigmaI,
        H_c_G_sigmaC,
        H_sigmaC,
        H_sigmaI,
        H_Abar_G_M_b, 
        H_k_G_eta, 
        H_eta_G_M_nb,
        H_M_G_q, 
        H_b_G_nb, 
        H_nb_G_q, 
        H_q, 
        energy] = state_bookkeeping;

    auto& [n,
        m,
        wins, 
        ties, 
        losses,
        neighbors, // std::vector<std::vector<int>>
        degree_list // std::vector<int> , \vec{k}
        ] = const_info;
    
    // Header line:
    // outfile <<"partition,sigma_C,sigma_I,community_status,total_status,nu,H_A_G_Abar_s_nu,H_nu,H_s_G_c_b_sigmaI,H_c_G_sigmaC,H_sigmaC,H_sigmaI,H_Abar_G_M_b,H_k_G_eta,H_eta_G_M_nb,H_M_G_q,H_b_G_nb,H_nb_G_q,H_q,energy\n";
    std::string output_string = "";
    std::string partition_string = "\"["; // List to represent the partition
    for(int i = 0; i < n; i++){
        partition_string += std::to_string(partition[i]);
        if(i != n - 1){
            partition_string += ",";
        }else{
            partition_string += "]\"";
        }
    }
    output_string += partition_string + ",";
    output_string += std::to_string(sigma_C) + ",";
    output_string += std::to_string(sigma_I) + ",";
    std::string community_status_string = "\"["; // List to represent community_status
    for(int r = 0; r < q; r++){
        community_status_string += std::to_string(community_status[r]);
        if(r != q - 1){
            community_status_string += ",";
        }else{
            community_status_string += "]\"";
        }
    }
    output_string += community_status_string + ",";
    std::string total_status_string = "\"["; // List to represent total_status
    for(int i = 0; i < n; i++){
        total_status_string += std::to_string(total_status[i]);
        if(i != n - 1){
            total_status_string += ",";
        }else{
            total_status_string += "]\"";
        }
    }
    output_string += total_status_string + ",";
    output_string += std::to_string(nu) + ","; 
    output_string += std::to_string(H_A_G_Abar_s_nu) + ","; 
    output_string += std::to_string(H_nu) + ","; 
    output_string += std::to_string(H_s_G_c_b_sigmaI) + ","; 
    output_string += std::to_string(H_c_G_sigmaC) + ","; 
    output_string += std::to_string(H_sigmaC) + ","; 
    output_string += std::to_string(H_sigmaI) + ","; 
    output_string += std::to_string(H_Abar_G_M_b) + ","; 
    output_string += std::to_string(H_k_G_eta) + ","; 
    output_string += std::to_string(H_eta_G_M_nb) + ","; 
    output_string += std::to_string(H_M_G_q) + ","; 
    output_string += std::to_string(H_b_G_nb) + ","; 
    output_string += std::to_string(H_nb_G_q) + ","; 
    output_string += std::to_string(H_q) + ","; 
    output_string += std::to_string(energy); // No comma at the end 
    output_string += "\n";
    return output_string;
}

// Printing functions (useful for debugging)
void print(ConstInfoType& const_info){
    auto& [n,
        m,
        wins, 
        ties, 
        losses,
        neighbors, // std::vector<std::vector<int>>
        degree_list // std::vector<int> , \vec{k}
        ] = const_info;
    std::cout << "n: " << n << ", m: " << m << "\n";
    std::cout << "wins:\n";
    for(int i = 0; i < n; i++){
        std::cout << "Node "<<i<<": ";
        for(auto& [j, Aij]: wins[i]){
            std::cout << j << " ";
        }
        std::cout << "\n";
    }
    std::cout << "ties:\n";
    for(int i = 0; i < n; i++){
        std::cout << "Node "<<i<<": ";
        for(auto& [j, Aij]: ties[i]){
            std::cout << j << " ";
        }
        std::cout << "\n";
    }
    std::cout << "losses:\n";
    for(int i = 0; i < n; i++){
        std::cout << "Node "<<i<<": ";
        for(auto& [j, Aij]: losses[i]){
            std::cout << j << " ";
        }
        std::cout << "\n";
    }
    std::cout << "neighbors:\n";
    for(int i = 0; i < n; i++){
        std::cout << "Node "<<i<<": ";
        for(auto& [j, Aij]: neighbors[i]){
            std::cout << j << " ";
        }
        std::cout << "\n";
    }
    std::cout << "degree_list: ";
    print(degree_list);
}
void print(StateType& state){
    auto& [partition, 
        sigma_C, 
        sigma_I,
        community_status, 
        total_status,
        nu] = state;
    std::cout << "partition: ";
    print(partition);
}
void print(StateBookkeepingType& state_bookkeeping){
    auto& [q, 
        node_counts, 
        group_members,
        edge_counts, 
        edge_counts_sum,
        degree_counts_by_group, 
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_c_b_sigmaI,
        H_c_G_sigmaC,
        H_sigmaC,
        H_sigmaI,
        H_Abar_G_M_b, 
        H_k_G_eta, 
        H_eta_G_M_nb,
        H_M_G_q, 
        H_b_G_nb, 
        H_nb_G_q, 
        H_q, 
        energy] = state_bookkeeping;
    
    std::cout << "q: " << q << ", H_A_G_Abar_s_nu: " << H_A_G_Abar_s_nu << ", H_nu: " << H_nu << ", H_s_G_c_b_sigmaI: " << H_s_G_c_b_sigmaI << 
    ", H_c_G_sigmaC: " << H_c_G_sigmaC << ", H_sigmaC: " << H_sigmaC << ", H_sigmaI: " << H_sigmaI << 
    ", H_Abar_G_M_b: " << H_Abar_G_M_b << ", H_k_G_eta: " << H_k_G_eta << ", H_eta_G_M_nb: " << H_eta_G_M_nb << ", H_M_G_q: " << H_M_G_q << 
    ", H_b_G_nb: " << H_b_G_nb << ", H_nb_G_q: " << H_nb_G_q << ", H_q: " << H_q << ", energy: " << energy << "\n";

    std::cout << "node_counts: ";
    print(node_counts);
    std::cout << "group_members:\n";
    for(int r = 0; r < q; r++){
        std::cout << "Group "<<r<<": ";
        for(int i : group_members[r]){
            std::cout <<i<<" ";
        }
        std::cout << "\n";
    }
    std::cout << "edge_counts:\n";
    print(edge_counts);
    std::cout << "edge_counts_sum:\n";
    print(edge_counts_sum);
    std::cout << "degree_counts_by_group:\n";
    for(int r = 0; r < q; r++){
        std::cout << "Group "<<r<<": ";
        for(auto& [k, count]: degree_counts_by_group[r]){
            std::cout << count << "x " << k << ", ";
        }
        std::cout << "\n";
    }
}
void print(std::vector<std::vector<int>> table) {
  for(int i = 0; i < table.size(); i++){
    for(int j = 0; j < table[0].size(); j++){
        std::cout << table[i][j] << ' ';
    }
    std::cout << "\n";
  }
}
void print(std::vector<std::vector<double>> table) {
  for(int i = 0; i < table.size(); i++){
    for(int j = 0; j < table[0].size(); j++){
        std::cout << table[i][j] << ' ';
    }
    std::cout << "\n";
  }
}
void print(std::vector<int> vec) {
  for (int n=0; n<vec.size(); ++n)
    std::cout << vec[n] << ' ';
  std::cout << '\n';
}
void print(std::vector<double> vec) {
  for (int n=0; n<vec.size(); ++n)
    std::cout << vec[n] << ' ';
  std::cout << '\n';
}
void print(int val){
  std::cout<<val<<std::endl;
}
void print(double val){
  std::cout<<val<<std::endl;
}
