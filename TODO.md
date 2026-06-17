1/21/26:
Goal: Create a python/c++ package that allows us to quickly run the community-hierarchy model (built distribution hosted on pypi). 
Creating a new repo, but note that we will then need to port changes made in the old case to this new SBM-hierarchy model. 
Steps to make this work:
1. Adjust the python wrapper to the new API and to accept (in fact insist on) directed graphs instead of undirected ones.
2. Adjust the c++ implementation to include score parameters for each of the nodes, as well as individual_depth
3. Add the community_depth parameter, ad the coupling (along with the 2nd moment of the scores within each group, which must be updated as bookkeeping).
Q: Does the fact that the meaning of the scores, stored in the [0,1] interval, changes as the depth changes matter? Should we instead prefer that when the depth changes the true scores should be invariant? I don't think so, the scores will stay the same as this inversion only occurs on proposal and depends on the current depth.
- Add a H_total (perhaps with a different name) to track as well, be sure to update it according to proposals
- We appear to have a singularity problem when sigma_I goes to 0, even when we fix sigma_C = 0 (so that this is just the usual depth). I recall this was an issue before too where there is a logarithmic divergence, how did we fix it?
- Perhaps if we scale the score proposals by sigmaI (although should be sure that this preserves the MCMC) .

1/29/16:
- Changed proposals to act on sigma_I * scores to avoid the earlier ergodicity issues that were leading to the sigma_I sinkhole. 
- Still need to check against Mathematica for the simple test case.

2/6/26:
- Implemented the community-hierarchy interaction, and treatment of ties (neutral interactions)
- For some reason the direct c++ runs 3x faster than the same function when called from python. Having trouble figuring out what the difference is here. 
- Should implement the rest of the API, host the function on pypi, and get the documentation live.
- Add some basic functionality tests (checks against exact Mathematica results)
- Include versions of the model where the community and hierarchy can each (or both) be fixed, in case we are interested in how it might change due to some externally observed variable.
- Add the consensus clustering code (in c++ since it can be pretty intensive)

2/8/26:
- DONE: Hosted on pypi with documentation
- Add ability for the downstream functions like node_properties to take samples_df as an argument

2/11/26:
- Add a wrapper as an R package

2/16/26:
- Make sure that we raise an error if there are no edges, also convert multigraphs
- DONE: Handle the case where nothing is recorded as dominant, default to this. 

2/18/26:
- DONE: _get_node_data() function, and take the group and hierarchy information, raise errors or return None
- Make function to plot the status decomposition of the graph
- Test against Mathematica for a multigraph
- Change type -> interaction_type

3/15/26:
- STAN implementation of the fixed groups model gives different and more sensible results
- Re-implement this STAN model using the form we have derived integrated over the community scores.
- Debug the directedstructure implementation until the outcomes match
- Test out a way to just use the g++ compiler directly to configure the sampling code
- Send the new version of the code to Elizabeth along with requests of what should be installed
- Note that make test on the current multigraph example results in a seg fault that we need to track down (should run this with valgrind and sort out the error)

  Node 17 with weight 6.000000
  Node 0 with weight 0.000000
make: *** [Makefile:63: test] Segmentation fault (core dumped)

3/10/26:
- Found the gap between the STAN and c++ implementations for the ties
- Want to implement the ability to pass fixed groups or scores
- Set up the code such that it is possible to take an input and

3/11/26:
- Need to bring the stripped and original implementations into alignment. 
- For some reason I'm now seeing that the c++ testing script is not exploring possible community structures at all while the python wrapped version seemingly is. 

3/12/26:
- Throw error when the input graph has no edges
- Consider renaming interaction -> type

3/20/26:
- Run the package on the synthetic network examples
- Debug why snr150_sigC5_rep09.gml, even in independent mode, is generating way too many communities. 