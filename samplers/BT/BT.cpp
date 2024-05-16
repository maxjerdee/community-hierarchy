/* This script uses parallel tempering on single site moves to sample from the posterior distribution
 * of the Bradley-Terry model including ties (Davidson model)
 * 
 * 
 * Written by Max Jerdee  16 MAY 2024
 */

// After making the file (run make in the directory), the executable may be run for example as:
// ./BT.exe -i ../../data/gml_files/AddHealth/highschool_1.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_1.csv
// ./BT.exe -i ../../data/gml_files/AddHealth/highschool_49.gml -o out.csv
// ./BT.exe -i ../../data/gml_files/misc/test.gml -o ../../data/samples/misc/BT/test.csv
// ./BT.exe -i ../../data/gml_files/villages/anyinteractions1.gml -o out.csv

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
const double MEAN_SIGMA = 3; // A priori mean value of sigma_T, the total status spread

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

const double NUMERICAL_CHECK_EPSILON = 0.001; // Tolerance to use when checking that the bookkeeping is in sync.

const double SIGMA_T_TILDE_STEP_SIZE = 0.1; // Typical amount to change sigma_T in each proposal
const double NU_TILDE_STEP_SIZE = 0.1;
const double TOTAL_STATUS_TILDE_STEP_SIZE = 0.5;

const double SIGMA_MIN = 0.1; // There is a logarithmic singularity when the sigma -> 0 which it is pretty easy for the Monte Carlo to get stuck in. 
// Apply the constraint that sigma_T must be above this threshold. (Really should be 0)
// Note that this changes the form of the hyperprior on the sigma

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
double, // sigma_T, spread of total statuses
std::vector<double>, // total_status, \vec{s}, total statuses
double // nu, the parameter that controls the likelihood of ties
> StateType;

const int sigma_T_StateTypeIndex = 0;

// MCMC State bookkeeping (information that can be inferred from the MCMC state but is useful to have on hand)
typedef std::tuple<
    double, // H_A_G_Abar_s_nu
    double, // H_nu
    double, // H_s_G_sigmaT
    double, // H_sigmaT
    double // energy (H = -log P)
    > StateBookkeepingType;

// Indices in the tuple at which to extract important information
const int energyStateBookkeepingTypeIndex = 4;

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

// Overflow protected version of log(e^a + e^b)
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
    outfile <<"sigma_T,total_status,nu,H_A_G_Abar_s_nu,H_nu,H_s_G_sigmaT,H_sigmaT,energy\n";

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
        if(i != j && ties[i].find(j) == ties[i].end()) {
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
    // print(const_info);

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
                if(var_name=="sigma_T" && (new_val < SIGMA_MIN)){ // Apply the constraint that sigma_T must be above SIGMA_MIN (this should always be the case anyway)
                    deltaH = 10000; // Prohibitively high potential barrier
                    std::cout << "Warning: artifical energy barrier\n";
                }
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
                    std::cout << " energy: " << std::get<energyStateBookkeepingTypeIndex>(state_bookkeepings[state_i]) 
                     << " sigma_T: " << std::get<sigma_T_StateTypeIndex>(states[state_i]) << " ";
                    
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
                        std::cout << " energy: " << std::get<energyStateBookkeepingTypeIndex>(state_bookkeepings[state_i]) 
                     << " sigma_T: " << std::get<sigma_T_StateTypeIndex>(states[state_i]) << " ";
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
    double sigma_T_init = p + 1; 

    std::vector<double> total_status_init;
    for(int i = 0; i < n; i++){
        double rand_total_status = sigma_T_init*unit_normal_dist(rng);
        total_status_init.push_back(rand_total_status);
    }

    double nu_init = uniform_01_dist(rng);

    StateType state = std::make_tuple(sigma_T_init, total_status_init, nu_init);
    
    return state;
}

// Implement the proposed swap. Also return the change in the log-likelihood, relevant for deciding whether to in fact accept this swap
// Goal is amortized (average) O(n/q) time complexity
double implement_proposal(std::string var_name, int var_index, double new_val, double old_val, int delta_q, StateType& state, StateBookkeepingType& state_bookkeeping, ConstInfoType& const_info, std::mt19937& rng){
    // Unpack the tuples
    auto& [sigma_T,
        total_status,
        nu] = state;

    auto& [
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_sigmaT,
        H_sigmaT,
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
    if(var_name == "sigma_T"){
        // This will impact H_s_G_sigmaT and H_sigmaT
        sigma_T = new_val;
        // state_bookkeeping change
        double H_sigmaT_new = std::log(M_PI*MEAN_SIGMA) + std::pow(sigma_T/MEAN_SIGMA,2)/M_PI;
        double H_s_G_sigmaT_new = n*std::log(sigma_T) + n*0.5*std::log(2*M_PI) + std::pow(old_val/sigma_T,2)*(H_s_G_sigmaT - n*std::log(old_val) - n*0.5*std::log(2*M_PI));
        deltaH = H_s_G_sigmaT_new - H_s_G_sigmaT + H_sigmaT_new - H_sigmaT; // Change in likelihood
        H_s_G_sigmaT = H_s_G_sigmaT_new;
        H_sigmaT = H_sigmaT_new;
        energy = H_A_G_Abar_s_nu + H_nu + H_s_G_sigmaT + H_sigmaT;
    }
    if(var_name == "total_status"){
        // This will impact H_A_G_Abar_s_nu and H_s_G_sigmaT
        // state change
        int i = var_index;
        total_status[i] = new_val;
        // state_bookkeeping change
        double H_s_G_sigmaT_new = H_s_G_sigmaT;
        H_s_G_sigmaT_new -= 0.5*std::pow(old_val/sigma_T,2) + std::log(sigma_T) + 0.5*std::log(2*M_PI);
        H_s_G_sigmaT_new += 0.5*std::pow(new_val/sigma_T,2) + std::log(sigma_T) + 0.5*std::log(2*M_PI);
        double H_A_G_Abar_s_nu_new = H_A_G_Abar_s_nu;

        for(auto& [j, Aij] : wins[i]){
            // std::cout <<"wins: " << j << " " << Aij << "\n";
            double old_score_diff = old_val - total_status[j];
            double score_diff = total_status[i] - total_status[j];
            H_A_G_Abar_s_nu_new -= Aij*(log_denom(old_score_diff,nu) - old_score_diff/2); // -log P(i beat j)
            H_A_G_Abar_s_nu_new += Aij*(log_denom(score_diff,nu) - score_diff/2); // -log P(i beat j)
        }
        for(auto& [j, Aij] : ties[i]){
            // std::cout <<"ties: " << j << " " << Aij << "\n";
            double old_score_diff = old_val - total_status[j];
            double score_diff = total_status[i] - total_status[j];
            H_A_G_Abar_s_nu_new -= Aij*(log_denom(old_score_diff,nu) - std::log(2*nu)); // -log P(i ties j)
            H_A_G_Abar_s_nu_new += Aij*(log_denom(score_diff,nu) - std::log(2*nu)); // -log P(i ties j)
        }
        for(auto& [j, Aij] : losses[i]){
            // std::cout <<"losses: " << j << " " << Aij << "\n";
            double old_score_diff = total_status[j] - old_val; // Flip the score difference
            double score_diff =  total_status[j] - total_status[i];
            H_A_G_Abar_s_nu_new -= Aij*(log_denom(old_score_diff,nu) - old_score_diff/2); // -log P(j beat i)
            H_A_G_Abar_s_nu_new += Aij*(log_denom(score_diff,nu) - score_diff/2); // -log P(j beat i)
        }
        deltaH = H_A_G_Abar_s_nu_new - H_A_G_Abar_s_nu + H_s_G_sigmaT_new - H_s_G_sigmaT; // Change in likelihood
        H_A_G_Abar_s_nu = H_A_G_Abar_s_nu_new;
        H_s_G_sigmaT = H_s_G_sigmaT_new;
        energy = H_A_G_Abar_s_nu + H_nu + H_s_G_sigmaT + H_sigmaT;
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
        energy = H_A_G_Abar_s_nu + H_nu + H_s_G_sigmaT + H_sigmaT;
    }
    return deltaH;
}

// Check that the state and its bookkeeping are in alignment. If not print the error and exit.
void check_state_bookkeeping(StateType& state, StateBookkeepingType& state_bookkeeping, ConstInfoType& const_info){
    // Unpack the tuples
    auto& [sigma_T,
        total_status,
        nu] = state;

    auto& [
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_sigmaT,
        H_sigmaT,
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

    auto& [
        H_A_G_Abar_s_nu_actual,
        H_nu_actual,
        H_s_G_sigmaT_actual,
        H_sigmaT_actual,
        energy_actual] = state_bookkeeping_actual;

    // Check for equality
    if(std::abs(H_A_G_Abar_s_nu - H_A_G_Abar_s_nu_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_A_G_Abar_s_nu = "<<H_A_G_Abar_s_nu<<" actual value H_A_G_Abar_s_nu = "<<H_A_G_Abar_s_nu_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_nu - H_nu_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_nu = "<<H_nu<<" actual value H_nu = "<<H_nu_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_s_G_sigmaT - H_s_G_sigmaT_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_s_G_sigmaT = "<<H_s_G_sigmaT<<" actual value H_s_G_sigmaT = "<<H_s_G_sigmaT_actual<<"\n";
        exit(0);
    }
    if(std::abs(H_sigmaT - H_sigmaT_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value H_sigmaT = "<<H_sigmaT<<" actual value H_sigmaT = "<<H_sigmaT_actual<<"\n";
        exit(0);
    }
    if(std::abs(energy - energy_actual) > NUMERICAL_CHECK_EPSILON){
        std::cout << "Error: bookkeeping value energy = "<<energy<<" actual value energy = "<<energy_actual<<"\n";
        exit(0);
    }
    // If checks are passed, update with the actual values (to stop floating point drift)
    H_A_G_Abar_s_nu = H_A_G_Abar_s_nu_actual;
    H_nu = H_nu_actual;
    H_s_G_sigmaT = H_s_G_sigmaT_actual;
    H_sigmaT = H_sigmaT_actual;
    energy = energy_actual;
}

// Generate proposed MCMC move: var_name, var_index, new_val, old_val
std::tuple<std::string, int, double, double, int> generate_proposal(StateType& state, StateBookkeepingType& state_bookkeeping, std::mt19937& rng){
    // Unpack the tuples
    auto& [sigma_T,
        total_status,
        nu] = state;

    auto& [
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_sigmaT,
        H_sigmaT,
        energy] = state_bookkeeping;

    // Initializing the variables to be returned
    std::string var_name;
    int var_index;
    double new_val;
    double old_val;
    int delta_q; 
    
    // Which variable will we change?
    int total_variables = 1 + total_status.size() + 1; // Total number of state variables
    std::uniform_int_distribution<int> rand_int(0, total_variables - 1);
    int rand_var_index = rand_int(rng);

    // By how much?
    std::normal_distribution<double> unit_normal_dist(0, 1);
    double z = unit_normal_dist(rng); // Element of randomness used to preturb continuous variables
    if(rand_var_index < 1){ // Shifting sigma_T
        var_name = "sigma_T";
        var_index = 0;
        old_val = sigma_T;
        new_val = old_val + SIGMA_T_TILDE_STEP_SIZE*z; // Apply the change
        if(new_val < SIGMA_MIN){ // sigma_T can't go below SIGMA_MIN
            std::cout << "Reflected sigma_T < SIGMA_MIN\n";
            new_val = SIGMA_MIN + (SIGMA_MIN - new_val); // Reflective boundary condition
        }
        return std::make_tuple(var_name, var_index, new_val, old_val, 0);
    }else{
        rand_var_index -= 1;
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
    auto& [sigma_T,
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
    double H_s_G_sigmaT = 0;
    for(int i = 0; i < n; i++){
        H_s_G_sigmaT += 0.5*std::pow(total_status[i]/sigma_T,2) + std::log(sigma_T) + 0.5*std::log(2*M_PI);
    }
    double H_sigmaT = std::log(M_PI*MEAN_SIGMA) + std::pow(sigma_T/MEAN_SIGMA,2)/M_PI;

    double energy = 0;
    energy += H_A_G_Abar_s_nu + H_nu + H_s_G_sigmaT + H_sigmaT;

    StateBookkeepingType state_bookkeeping = std::make_tuple(H_A_G_Abar_s_nu, H_nu, H_s_G_sigmaT, H_sigmaT, energy);
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
    auto& [sigma_T,
        total_status,
        nu] = state;

    auto& [
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_sigmaT,
        H_sigmaT,
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
    // outfile <<"sigma_T,total_status,nu,H_A_G_Abar_s_nu,H_nu,H_s_G_sigmaT,H_sigmaT,energy\n";
    std::string output_string = "";
    output_string += std::to_string(sigma_T) + ",";
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
    output_string += std::to_string(H_s_G_sigmaT) + ","; 
    output_string += std::to_string(H_sigmaT) + ","; 
    output_string += std::to_string(energy); // No comma at the end 
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
    auto& [sigma_T,
        total_status,
        nu] = state;
    std::cout << "nu: " << nu << ", sigma_T: " << sigma_T; 
}
void print(StateBookkeepingType& state_bookkeeping){
    
    auto& [
        H_A_G_Abar_s_nu,
        H_nu,
        H_s_G_sigmaT,
        H_sigmaT,
        energy] = state_bookkeeping;
    
    std::cout << "H_A_G_Abar_s_nu: " << H_A_G_Abar_s_nu << ", H_nu: " << H_nu << 
    ", H_s_G_sigmaT: " << H_s_G_sigmaT << ", H_sigmaT: " << H_sigmaT << ", energy: " << energy << "\n";
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
