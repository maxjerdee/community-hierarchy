#ifndef CONSENSUS_H
#define CONSENSUS_H

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

class ConsensusClustering {
public:
    /**
     * Compute the consensus clustering from a set of clusterings.
     * 
     * Finds the clustering whose coincidence matrix is closest to the average 
     * coincidence matrix under the specified metric.
     * 
     * @param clusterings: 2D vector of clusterings (each row is a clustering)
     * @param metric: "L1" or "L2" distance metric
     * @return The consensus clustering (vector of group assignments)
     */
    static std::vector<int> consensus_clustering(
        const std::vector<std::vector<int>>& clusterings,
        const std::string& metric = "L1"
    );
};

#endif // CONSENSUS_H
