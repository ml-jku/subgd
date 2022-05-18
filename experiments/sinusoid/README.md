# Sinusoid experiments

- The main notebook for the sinusoid experiments is `sinusoid-results.ipynb`. It contains the baselines comparison, the effective rank plots, and the comparison of different pretraining strategies.
- `sinusoid-results-fixedsteps.ipynb` contains the analysis of results after a single update step.
- `sinusoid-ablations.ipynb` contains the comparisons of different strategies to calculate a preconditioning matrix and restricting finetuning to different degrees of freedom.
- `sinusoid-chunking-analysis.ipynb` contains the analysis of using epoch-wise vs. global differences to calculate the preconditioning matrix.

For the MT-Net and Meta-Curvature experiments, we used the authors' implementations:
- MT-Net: https://github.com/yoonholee/MT-net
- Meta-Curvature: https://github.com/silverbottlep/meta_curvature
