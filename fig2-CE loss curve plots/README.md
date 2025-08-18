### Replicating Figure 2 Analysis

To reproduce the model testing analysis shown in Figure 2:

1.  Run `01-testing_AbLMs.ipynb`: This notebook evaluates each pre-trained Antibody Language Model (AbLM) on the test datasets available [here](https://zenodo.org/records/15447079), calculating the cross-entropy loss for each combination.
2.  Run `02-plots.ipynb`: This notebook uses the results from the first step to generate the loss curves per model/dataset size and the projected plot, as seen in Figure 2.

Ensure you run the notebooks in numerical order.