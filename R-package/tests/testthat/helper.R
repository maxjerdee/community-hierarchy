games <- function() {
  data.frame(winner = c("a", "a", "b", "c", "d", "a", "b", "e"),
             loser = c("b", "c", "c", "d", "e", "e", "d", "f"),
             type = c(rep("dominant", 3), "neutral", rep("dominant", 4)),
             weight = c(1, 1, 1, 2, 1, 1, 1, 1))
}
quick <- function(G, config = ch_config(), ...) {
  ch_fit(G, config, num_chains = 1, samples_per_chain = 30, burn_sweeps = 5, seed = 1,
         show_progress = FALSE, ...)
}
