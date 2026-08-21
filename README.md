# eTFAST: An Explainable Deep Learning Framework for Inferring Spatially Resolved Transcription Factor Activity from Cell Signaling Networks



**eTFAST** is an explainable deep learning framework based on physical modeling and attention mechanisms, designed to infer spatially resolved transcription factor (TF) activity from spatial transcriptomics (ST) data. By integrating intercellular communication (CCC) signals, TF expression, and prior regulatory knowledge, it maps extracellular ligand–receptor signals to intracellular TF activity and downstream target gene expression, thereby revealing how the spatial microenvironment regulates transcriptional programs. The TF activities inferred by eTFAST serve as biologically interpretable, low‑dimensional representations of cellular states, directly applicable to downstream tasks such as spatial clustering, pseudotime inference, multi‑layer CCC analysis, and functional enrichment analysis.

## Key Features
- **Physics‑inspired spatial signaling modeling**: For diffusion‑dependent ligands, a diffusion‑degradation equation is used to simulate spatial concentration fields; for contact‑dependent signaling, interactions are restricted to directly adjacent cells, enabling spatially realistic estimation of CCC signals.

- **Context‑aware TF activity inference**: An attention mechanism dynamically weighs the contributions of upstream LR signals from the spatial neighborhood against the cell's own TF expression to predict microenvironment‑specific TF activity.

- **End‑to‑end interpretable architecture**: Jointly optimizes the "spatial LR signal → TF activity → target gene expression" cascade, incorporating contrastive learning, Elastic Weight Consolidation (EWC), and multi‑task learning to output high‑resolution spatial maps of TF activity.

- **Unified downstream analysis platform**: Provides low‑dimensional embeddings of TF activity that support spatial clustering, pseudotime trajectory reconstruction, key regulon identification, functional enrichment analysis, and visualization of multi‑layer signaling networks.

## Installation
We recommend using `conda` to create a separate environment and install dependencies:

```bash
conda create -n etfast python=3.9
conda activate etfast
pip install -r requirements.txt
```
The eTFAST Python package and its dependencies are available from the GitHub repository: https://github.com/SunXQlab/eTFAST

## Data Preparation

Before running eTFAST, prepare your ST data in a standardized format, including:
- Gene expression matrix (cells × genes)
- Spatial coordinates of cells
- Cell type annotations (optional but recommended)

The data format is recommended to be `.h5ad` (AnnData) or CSV files. The preprocessing scripts (`0_run_eTFAST_prepare_*.ipynb`) in the repository can convert raw data into the required input format and automatically select candidate ligands, receptors, TFs, and target genes.

Input files should be placed under the `data/` directory, with the following structure:

```text
data/
├── raw_expression.h5ad        # Raw expression data
├── spatial_coordinates.csv     # Spatial coordinates
├── cell_metadata.csv          # Cell annotations
└── prior_database/            # Prior databases (diffusion‑dependent LigRecDB, contact‑dependent LigRecDB, TFTGDB)
```
## Running eTFAST

We provide complete runnable examples for three datasets: mouse brain, human tonsil, and human melanoma.

### 1. Preprocess the data
Use the Jupyter notebook scripts to preprocess raw data and generate model inputs:
- `0_run_eTFAST_prepare_mouse_brain.ipynb`
- `0_run_eTFAST_prepare_human_tonsil.ipynb`
- `0_run_eTFAST_prepare_human_melanoma.ipynb`

After running these, standardized input files will be saved under `data/processed/`.

### 2. Run the main eTFAST program
Select the corresponding main script according to your dataset:
- Mouse brain: `main_mouse_brain.py`
- Human tonsil: `main_human_tonsil.py`
- Human melanoma: `main_human_melamela.py`

Example execution:
```bash
python main_mouse_brain.py
```
During execution, the model will automatically train, infer TF activities, and predict target gene expression. All results will be saved in the `results_<dataset_name>/` directory.

### 3. Output results
The model outputs include:
- `results.pkl`: Summary of output results (containing TF activity matrix, LR signaling strengths, regulatory importance scores, etc.)
- Additional detailed result files, such as clustering labels, pseudotime values, and network visualizations.

## Examples and Reproducibility

We provide complete analysis pipelines for the three datasets, with code and results available in the following repositories:
- eTFAST main repository: https://github.com/SunXQlab/eTFAST
- Analysis code repository (including preprocessing, running, and visualization): https://github.com/SunXQlab/eTFAST-analysis

## Application Guide

To apply eTFAST to your own ST dataset, follow these steps:

1. **Organize your data**

Prepare your expression matrix, spatial coordinates, and cell annotations according to the Data Preparation requirements above.

2. **Modify the main script**

Update the file paths and adjust key parameters (e.g., cell type thresholds, diffusion coefficient, learning rate) in the corresponding `main_*.py` script.

3. **Run preprocessing and training**

Execute the preprocessing notebook and then the main script. Tune hyperparameters (e.g., batch size, number of epochs, regularization weights) to fit your data scale and complexity.

4. **Perform downstream analyses**

Use the output TF activity matrix (`results.pkl` or the exported `.h5ad` file) for clustering, pseudotime trajectory reconstruction, CCC network inference, and functional enrichment analysis.

## License
This project is licensed under the MIT License.

## Contact
For any questions, bug reports, or feature requests, please open a GitHub Issue or contact us via email: sunxq6@mail.sysu.edu.cn.
