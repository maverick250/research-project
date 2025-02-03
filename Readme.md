# Bayesian Nonparametric Framework for Clustering and Generative Modeling of Multiple Time Series

**Abstract:**  
This project proposes a Bayesian nonparametric framework for clustering and generative modeling of multiple time series, integrating a Dirichlet Process Mixture Model (DPMM) for clustering with an Indian Buffet Process (IBP) for latent feature allocation within a state-space model. By avoiding predefined limits on the number of clusters or features, the approach can adapt its complexity to the data and discover shared temporal structures across series.

Evaluation on synthetic datasets demonstrates that the model successfully captures broad clustering patterns and is capable of generating realistic synthetic time series, albeit with some limitations: threshold-based feature allocations often lead to oversimplified “all-on” or “all-off” usage, and certain parameters (e.g., process noise) are overestimated, reducing clustering accuracy (Adjusted Rand Index ≈ 0.35). Posterior predictive checks show moderate performance for most time series but underestimate high-variance behaviors. Future work includes refining prior specifications, exploring more robust inference methods, and testing on complex real-world data to fully exploit the model’s flexibility and improve its feature-allocation fidelity.

