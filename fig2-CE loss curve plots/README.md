### Replicating Figure 2 Analysis

To reproduce the model testing analysis shown in Figure 2:

1.  Run `01-testing_AbLMs.ipynb`, once per model. It evaluates one pre-trained Antibody Language
    Model (AbLM) on the ten donor test datasets available
    [here](https://doi.org/10.5281/zenodo.15447078) (`Fixed_data_profiles.zip`, unzipped into
    `data/`), and prints the cross-entropy loss on each donor and the mean over the ten.
2.  Collect the fifteen means in `isodata.csv`, next to these notebooks, with one row per model:

    | model_size | size_label | model_name | dataset | loss |
    |---|---|---|---|---|
    | 150000000 | 150M | 150M-F | full | 0.23943 |

    These are the numbers of the S3 Table.
3.  Run `02-plots.ipynb`: it reads `isodata.csv` and draws the loss curves per model/dataset size
    and the projected plot, as seen in Figure 2. Following the Methods, the quadratics are fit to
    the points around the minimum of each curve; the four points that are left out are listed at
    the top of the notebook.

Ensure you run the notebooks in numerical order.
