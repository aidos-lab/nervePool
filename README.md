# nervePool
### A pooling layer for simplicial complexes
> For deep learning problems on graph-structured data, pooling layers are important for down sampling, reducing computational cost, and to minimize overfitting.
> We define a pooling layer, NervePool, for data structured as simplicial complexes, which are generalizations of graphs that include higher-dimensional simplices beyond vertices and edges; this structure allows for greater flexibility in modeling higher-order relationships. 
> The proposed simplicial coarsening scheme is built upon partitions of vertices, which allow us to generate hierarchical representations of simplicial complexes, collapsing information in a learned fashion. 
> NervePool builds on the learned vertex cluster assignments and extends to coarsening of higher dimensional simplices in a deterministic fashion. 
> While in practice, the pooling operations are computed via a series of matrix operations, the topological motivation is a set-theoretic construction based on unions of stars of simplices and the nerve complex.
<br/>
Paper: [`arXiv:####.#####`][https://www.sarah-mcguire.com]


  ### Code
  The notebook `examples_notebook.ipynb` walks through a series of examples, using NervePool to pool various simplicicial compelxes, for different choices of input vertex cover.
  Each simplicial complex is an object defined by `complex.py` and the collection of functions that perform the pooling on a complex are found in `pooling_functions.py`.
 
  
  

## License & citation

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
S. McGuire, E. Munch, and M. Hirn. NervePool: A Simplicial Pooling Layer, 2023. 

```
@article{mcguire2023nervePool,
  title = {NervePool: A Simplicial Pooling Layer},
  author = {McGuire, Sarah and Munch, Elizabeth and Hirn, Matthew},
  year = {2023},
  archiveprefix = {arXiv},
  eprint = {2222:2222},
  url = {https://arxiv.org/abs/},
}
```
