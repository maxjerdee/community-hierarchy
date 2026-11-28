// Model presets, shared by every binding: from a model choice (the named presets plus
// any explicit values) to the parameters the sampler takes.
//
// The bindings check the choice (names, ranges, conflicts with the graph) and raise
// their own errors; resolve_model assumes a valid choice and only guards against an
// unknown name.

#ifndef MODEL_PRESETS_H
#define MODEL_PRESETS_H

#include <map>
#include <optional>
#include <string>

struct ModelChoice {
    std::string groups_model = "traditional_SBM";
    std::string hierarchy_model;  // "" = unset: chosen from the graph
    std::string interaction = "coupled";
    std::string mixing_variation;  // "" = unset: from groups_model
    int assortative = -1;          // -1 = unset (from groups_model), 0 or 1
    bool degree_correction = true;
    // Explicit settings of variation_in, variation_out, individual_depth,
    // group_depth, ties_parameter and num_groups.  A missing key is unset (the
    // preset decides); a key with no value means "infer".
    std::map<std::string, std::optional<double>> given;
};

struct ResolvedModel {
    std::string hierarchy_model;
    bool assortative = true;
    std::optional<double> variation_in, variation_out;  // nullopt = inferred
    double degree_correction = 0.5;                     // variation of the degree prior
    std::optional<double> individual_depth, group_depth, ties_parameter;
    std::optional<int> num_groups;
};

// has_neutral: whether the graph has neutral edges (draws), which decides an unset
// hierarchy_model ("bradley_terry_ties" if so, else "bradley_terry").
//
//   groups_model         mixing_variation   assortative
//   traditional_SBM      simple             false   (the default)
//   general_ASBM         general            true
//   simple_ASBM          simple             true
//   hybrid_ASBM          internal           true
//   planted_partition    none               true
//
//   mixing_variation     variation_in / variation_out
//   general              inferred / inferred
//   simple               0.5 / 0.5
//   none                 0 / 0
//   internal             0.5 / 0
//   external             0 / 0.5
//
// Explicit values take precedence over both tables, except that hierarchy_model
// "bradley_terry" fixes ties_parameter at 0 and interaction "independent" fixes
// group_depth at 0.  degree_correction true is the variation 0.5 (Dirichlet
// concentration 1: uniform over degree sequences).  Throws std::invalid_argument
// for an unknown preset name.
ResolvedModel resolve_model(const ModelChoice& m, bool has_neutral);

#endif  // MODEL_PRESETS_H
