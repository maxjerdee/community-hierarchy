#' Model configuration
#'
#' Every argument has a default, so `ch_config()` is the full coupled model with
#' everything inferred.  Arguments left out take the value of the `groups_model`,
#' `hierarchy_model` and `interaction` presets.  For the optional numeric
#' parameters, leaving the argument out means "as the preset says", and `NULL`
#' means "infer it".
#'
#' @param groups_model Community-structure preset.  It sets `mixing_variation`
#'   (and, for `"traditional_SBM"`, `assortative = FALSE`) unless those are given:
#'
#'   | preset | `mixing_variation` | `assortative` |
#'   |---|---|---|
#'   | `"traditional_SBM"` (default) | `"simple"` | `FALSE` |
#'   | `"general_ASBM"` | `"general"` | `TRUE` |
#'   | `"simple_ASBM"` | `"simple"` | `TRUE` |
#'   | `"hybrid_ASBM"` | `"internal"` | `TRUE` |
#'   | `"planted_partition"` | `"none"` | `TRUE` |
#'
#'   The default is the degree-corrected SBM, one density for all group pairs;
#'   the assortative presets give edges within and between groups separate
#'   densities.
#' @param hierarchy_model `"bradley_terry"` (no draws: `ties_parameter` is fixed
#'   at 0, and a graph with neutral edges is an error) or `"bradley_terry_ties"`
#'   (draws as in Davidson 1970).  Left out: `"bradley_terry_ties"` if the graph
#'   has a neutral edge, else `"bradley_terry"`.
#' @param interaction `"coupled"` (default): a score is its group's score plus an
#'   individual deviation, and `group_depth` is inferred.  `"independent"`:
#'   `group_depth` is fixed at 0, so the scores do not depend on the groups.
#' @param assortative `TRUE`: separate densities for edges within and between
#'   groups.  `FALSE`: one density.  Left out: from `groups_model`.
#' @param mixing_variation Preset for `variation_in` / `variation_out`:
#'   `"general"` infers both, `"simple"` fixes both at 0.5, `"none"` both at 0,
#'   `"internal"` 0.5 / 0, `"external"` 0 / 0.5.  Left out: from `groups_model`.
#' @param variation_in,variation_out Fixed variation in \[0, 1\] of the
#'   within-group / between-group block edge counts; `NULL` infers it.  Left out:
#'   from `mixing_variation`.
#' @param degree_correction `TRUE` (default): degree correction, with a uniform
#'   prior over each group's degree sequence.  `FALSE`: none.
#' @param individual_depth Fixed spread of the individual scores about their
#'   group's score (sigma_I, > 0); `NULL` or left out infers it.
#' @param group_depth Fixed spread of the group scores (sigma_G, >= 0); `NULL` or
#'   left out infers it (it is fixed at 0 by `interaction = "independent"`).
#' @param ties_parameter Fixed draw propensity nu (>= 0); `NULL` or left out
#'   infers it (it is fixed at 0 by `hierarchy_model = "bradley_terry"`).
#' @param num_groups Fixed number of groups; `NULL` or left out infers it.
#' @param group_attribute Name of a node attribute holding known groups.  The
#'   partition is then fixed and only the hierarchy is inferred.
#' @param hierarchy_attribute Name of a numeric node attribute holding known
#'   scores, which are then fixed.
#' @param weighted_SBM `TRUE`: edge weights are real-valued strengths rather than
#'   counts.  The binary network follows the SBM, the weights between two groups
#'   are uniform on the simplex of their total (Peixoto 2018), and the totals of
#'   the group pairs joined by an edge are uniform on the simplex of the overall
#'   weight.  Default `FALSE`.
#' @param weighted_proposals `weighted_SBM` only: `TRUE` makes the block-informed
#'   partition proposals follow the weights rather than the binary edge counts
#'   (experimental).  The posterior is the same either way.
#'
#' @return An object of class `ch_config`.
#' @seealso [ch_fit()]
#' @examples
#' ch_config()
#' ch_config(interaction = "independent")
#' ch_config(assortative = FALSE, variation_in = 0.5, variation_out = 0.5)
#' @export
ch_config <- function(groups_model = "traditional_SBM", hierarchy_model, interaction = "coupled",
                      assortative, mixing_variation, variation_in, variation_out,
                      degree_correction = TRUE, individual_depth, group_depth,
                      ties_parameter, num_groups, group_attribute = NULL,
                      hierarchy_attribute = NULL, weighted_SBM = FALSE,
                      weighted_proposals = FALSE) {
  cfg <- list(groups_model = groups_model, interaction = interaction,
              degree_correction = degree_correction, group_attribute = group_attribute,
              hierarchy_attribute = hierarchy_attribute, weighted_SBM = weighted_SBM,
              weighted_proposals = weighted_proposals)
  # Optional arguments are stored only when given; list(x = NULL) keeps an explicit NULL.
  if (!missing(hierarchy_model)) cfg["hierarchy_model"] <- list(hierarchy_model)
  if (!missing(assortative)) cfg["assortative"] <- list(assortative)
  if (!missing(mixing_variation)) cfg["mixing_variation"] <- list(mixing_variation)
  if (!missing(variation_in)) cfg["variation_in"] <- list(variation_in)
  if (!missing(variation_out)) cfg["variation_out"] <- list(variation_out)
  if (!missing(individual_depth)) cfg["individual_depth"] <- list(individual_depth)
  if (!missing(group_depth)) cfg["group_depth"] <- list(group_depth)
  if (!missing(ties_parameter)) cfg["ties_parameter"] <- list(ties_parameter)
  if (!missing(num_groups)) cfg["num_groups"] <- list(num_groups)
  validate_config(cfg)
  structure(cfg, class = "ch_config")
}

is_flag <- function(x) is.logical(x) && length(x) == 1L && !is.na(x)
is_number <- function(x) is.numeric(x) && length(x) == 1L && is.finite(x)
is_string <- function(x) is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x)
given <- function(cfg, name) name %in% names(cfg)

.model_settings <- c("variation_in", "variation_out", "individual_depth", "group_depth",
                     "ties_parameter", "num_groups")

# The configuration as the C++ core's ModelChoice (src/model_presets.h).
model_choice <- function(cfg) {
  list(groups_model = cfg$groups_model,
       hierarchy_model = if (given(cfg, "hierarchy_model")) cfg$hierarchy_model else "",
       interaction = cfg$interaction,
       mixing_variation = if (given(cfg, "mixing_variation")) cfg$mixing_variation else "",
       assortative = if (given(cfg, "assortative")) as.integer(cfg$assortative) else -1L,
       degree_correction = cfg$degree_correction,
       given = cfg[intersect(.model_settings, names(cfg))])
}

.groups_models <- c("traditional_SBM", "general_ASBM", "simple_ASBM", "hybrid_ASBM",
                    "planted_partition")
.hierarchy_models <- c("bradley_terry", "bradley_terry_ties")
.interactions <- c("coupled", "independent")
.mixing_variations <- c("general", "simple", "none", "internal", "external")

one_of <- function(x, valid, name, hint = "") {
  if (!x %in% valid) {
    stop(sprintf("Unknown %s \"%s\". Valid options: %s.%s", name, x,
                 paste0("\"", valid, "\"", collapse = ", "), hint), call. = FALSE)
  }
}

# The checks that need no graph: types, names, ranges, and explicit values that a
# preset would override.  (The presets themselves are applied in the C++ core.)
validate_config <- function(cfg) {
  for (name in c("groups_model", "interaction")) {
    if (!is_string(cfg[[name]])) stop(name, " must be a string.", call. = FALSE)
  }
  for (name in c("hierarchy_model", "mixing_variation")) {
    if (given(cfg, name) && !is_string(cfg[[name]])) stop(name, " must be a string.", call. = FALSE)
  }
  for (name in c("degree_correction", "weighted_SBM", "weighted_proposals")) {
    if (!is_flag(cfg[[name]])) stop(name, " must be TRUE or FALSE.", call. = FALSE)
  }
  if (given(cfg, "assortative") && !is_flag(cfg$assortative)) {
    stop("assortative must be TRUE or FALSE.", call. = FALSE)
  }
  for (name in c("group_attribute", "hierarchy_attribute")) {
    if (!is.null(cfg[[name]]) && !is_string(cfg[[name]])) {
      stop(name, " must be a non-empty string or NULL.", call. = FALSE)
    }
  }

  if (cfg$groups_model == "general_canonical") {
    stop("groups_model \"general_canonical\" was renamed \"general_ASBM\" (the same model).",
         call. = FALSE)
  }
  one_of(cfg$groups_model, .groups_models, "groups_model",
         " Degree correction is not part of the model name; set it with degree_correction.")
  if (given(cfg, "hierarchy_model")) one_of(cfg$hierarchy_model, .hierarchy_models, "hierarchy_model")
  one_of(cfg$interaction, .interactions, "interaction")
  if (given(cfg, "mixing_variation")) one_of(cfg$mixing_variation, .mixing_variations, "mixing_variation")

  for (name in intersect(.model_settings, names(cfg))) {
    x <- cfg[[name]]
    if (is.null(x)) next
    if (!(is.numeric(x) && length(x) == 1L && is.finite(x))) {
      stop(name, " must be a finite number or NULL (inferred).", call. = FALSE)
    }
    if (name %in% c("variation_in", "variation_out") && !(x >= 0 && x <= 1)) {
      stop(name, " must be in [0, 1] or NULL (inferred), got ", x, ".", call. = FALSE)
    }
    if (name == "individual_depth" && !(x > 0)) {
      stop("individual_depth must be positive, got ", x, ".", call. = FALSE)
    }
    if (name %in% c("group_depth", "ties_parameter") && !(x >= 0)) {
      stop(name, " must be non-negative, got ", x, ".", call. = FALSE)
    }
    if (name == "num_groups" && !(x >= 1 && x == round(x))) {
      stop("num_groups must be a positive whole number or NULL (inferred), got ", x, ".",
           call. = FALSE)
    }
  }

  # Given, and not 0 (NULL, "infer", counts as nonzero).
  nonzero <- function(name) given(cfg, name) && !identical(cfg[[name]], 0) && !identical(cfg[[name]], 0L)
  if (given(cfg, "hierarchy_model") && cfg$hierarchy_model == "bradley_terry" && nonzero("ties_parameter")) {
    stop("hierarchy_model \"bradley_terry\" fixes ties_parameter at 0, but ties_parameter = ",
         format(cfg$ties_parameter), " was given. Use hierarchy_model \"bradley_terry_ties\" ",
         "to model draws.", call. = FALSE)
  }
  if (cfg$interaction == "independent" && nonzero("group_depth")) {
    stop("interaction \"independent\" fixes group_depth at 0, but group_depth = ",
         format(cfg$group_depth), " was given. Use interaction \"coupled\" to fit group_depth.",
         call. = FALSE)
  }
  invisible(cfg)
}

# The checks that need the graph, then the sampler's parameters from the presets
# and the explicit values (C++ core).
resolve_config <- function(cfg, has_neutral, num_nodes) {
  if (!given(cfg, "hierarchy_model") && !has_neutral && given(cfg, "ties_parameter") &&
      !identical(as.numeric(cfg$ties_parameter), 0)) {
    stop("ties_parameter = ", format(cfg$ties_parameter), " was given, but G has no neutral ",
         "edges, so hierarchy_model resolves to \"bradley_terry\", which fixes ",
         "ties_parameter at 0. Pass hierarchy_model \"bradley_terry_ties\" to fit a draw ",
         "rate anyway.", call. = FALSE)
  }
  if (given(cfg, "hierarchy_model") && cfg$hierarchy_model == "bradley_terry" && has_neutral) {
    stop("hierarchy_model \"bradley_terry\" has no draws, but G has neutral edges. Use ",
         "hierarchy_model \"bradley_terry_ties\" (the default when G has neutral edges), ",
         "or remove the neutral edges.", call. = FALSE)
  }
  if (!is.null(cfg$num_groups) && cfg$num_groups > num_nodes) {
    stop("num_groups = ", cfg$num_groups, " is larger than the number of nodes (", num_nodes,
         ").", call. = FALSE)
  }
  do.call(.ch_resolve_model, c(model_choice(cfg), list(has_neutral = has_neutral)))
}

#' @export
print.ch_config <- function(x, ...) {
  shown <- vapply(names(x), function(n) {
    v <- x[[n]]
    if (is.null(v)) {
      if (n %in% c("group_attribute", "hierarchy_attribute")) return(NA_character_)
      return(paste0(n, " = NULL"))
    }
    paste0(n, " = ", deparse(v))
  }, character(1))
  cat("<ch_config>\n", paste0("  ", stats::na.omit(shown), collapse = "\n"), "\n", sep = "")
  invisible(x)
}
