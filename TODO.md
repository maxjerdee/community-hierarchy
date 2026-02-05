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