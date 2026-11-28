#include "model_presets.h"

#include <stdexcept>

namespace {

void unknown(const char* what, const std::string& name) {
    throw std::invalid_argument(std::string("resolve_model: unknown ") + what + " \"" + name + "\"");
}

std::optional<std::optional<double>> setting(const ModelChoice& m, const char* name) {
    auto it = m.given.find(name);
    if (it == m.given.end()) return std::nullopt;
    return it->second;
}

}  // namespace

ResolvedModel resolve_model(const ModelChoice& m, bool has_neutral) {
    ResolvedModel r;
    r.hierarchy_model = !m.hierarchy_model.empty() ? m.hierarchy_model
                        : has_neutral              ? "bradley_terry_ties"
                                                   : "bradley_terry";
    if (r.hierarchy_model != "bradley_terry" && r.hierarchy_model != "bradley_terry_ties")
        unknown("hierarchy_model", r.hierarchy_model);
    if (m.interaction != "coupled" && m.interaction != "independent")
        unknown("interaction", m.interaction);

    std::string mixing = m.mixing_variation;
    if (mixing.empty()) {
        if (m.groups_model == "general_ASBM") mixing = "general";
        else if (m.groups_model == "simple_ASBM" || m.groups_model == "traditional_SBM") mixing = "simple";
        else if (m.groups_model == "hybrid_ASBM") mixing = "internal";
        else if (m.groups_model == "planted_partition") mixing = "none";
        else unknown("groups_model", m.groups_model);
    }
    if (mixing == "simple") { r.variation_in = 0.5; r.variation_out = 0.5; }
    else if (mixing == "none") { r.variation_in = 0.0; r.variation_out = 0.0; }
    else if (mixing == "internal") { r.variation_in = 0.5; r.variation_out = 0.0; }
    else if (mixing == "external") { r.variation_in = 0.0; r.variation_out = 0.5; }
    else if (mixing != "general") unknown("mixing_variation", mixing);
    if (auto s = setting(m, "variation_in")) r.variation_in = *s;
    if (auto s = setting(m, "variation_out")) r.variation_out = *s;

    r.assortative = m.assortative == -1 ? m.groups_model != "traditional_SBM" : m.assortative == 1;
    r.degree_correction = m.degree_correction ? 0.5 : 0.0;
    if (auto s = setting(m, "individual_depth")) r.individual_depth = *s;
    if (auto s = setting(m, "group_depth")) r.group_depth = *s;
    if (auto s = setting(m, "ties_parameter")) r.ties_parameter = *s;
    if (m.interaction == "independent") r.group_depth = 0.0;
    if (r.hierarchy_model == "bradley_terry") r.ties_parameter = 0.0;
    if (auto s = setting(m, "num_groups"); s && *s) r.num_groups = static_cast<int>(**s);
    return r;
}
