# API reference

The public API is three objects: {class}`~community_hierarchy.Config` describes the
model, {func}`~community_hierarchy.fit` runs the sampler, and
{class}`~community_hierarchy.Result` holds the samples and their summaries.  Plotting and
export helpers work on a `Result`.

## Model and inference

```{eval-rst}
.. autoclass:: community_hierarchy.Config
   :members:

.. autofunction:: community_hierarchy.fit
```

## Results

```{eval-rst}
.. autoclass:: community_hierarchy.Result
   :members:
   :special-members: __len__
```

## Plotting and export

```{eval-rst}
.. autofunction:: community_hierarchy.plot_node_properties

.. autofunction:: community_hierarchy.plot_adjacency_matrix

.. autofunction:: community_hierarchy.plot_MCMC_entropy

.. autofunction:: community_hierarchy.write_gml_with_inferences
```
