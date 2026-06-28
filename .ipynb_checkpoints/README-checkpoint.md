# Explainable AI Mini Project

## R-GCN Node Classification with GNNExplainer on the AIFB RDF Knowledge Graph

This project explains predictions of a machine learning model on an RDF dataset. It uses the AIFB RDF Knowledge Graph for node classification and applies GNNExplainer to identify which graph relationships influenced an individual model prediction.

## Authors

- Steevan Gonsalves
- Ashton Prince Rodrigues
- Aman Sami Ur Rahman Mohammed

## Course

Explainable AI (XAI), Paderborn University

## Project Goal

The goal of this mini-project is to:

1. Select an RDF dataset.
2. Analyze the dataset.
3. Train a machine learning model on the data.
4. Evaluate the predictive performance of the model.
5. Generate explanations of the model.
6. Evaluate and interpret the explanations.

## Dataset

The project uses the **AIFB RDF Knowledge Graph**.

Files used:

- `aifbfixed_complete.n3`
- `completeDataset.tsv`

Dataset statistics:

- RDF triples: **29,226**
- Graph nodes: **8,285**
- Relation types: **47**
- Labeled person nodes: **176**
- Unlabeled nodes: **8,109**
- Target classes: **4**

The RDF graph contains entities such as people, research groups, publications, projects, and research areas. RDF triples are represented as:

```text
Subject -> Predicate -> Object
```

## Methodology

### RDF Loading and Graph Conversion

The RDF dataset is loaded using RDFLib. The RDF triples are converted into a PyTorch Geometric graph representation containing:

- `edge_index`
- `edge_type`
- node indices
- relation indices

The graph is then stored in a PyTorch Geometric `Data` object.

### Node Features

Node features are generated using `OneHotDegree`. This creates structural node features based on graph degree.

Final feature matrix:

```text
8285 nodes x 72 features
```

### Labels

The classification labels are loaded from `completeDataset.tsv`. Labeled person nodes receive class labels, while unlabeled nodes receive `-1`.

Only labeled nodes are used for supervised training and evaluation. Unlabeled nodes are kept in the graph structure but are not included in the training loss.

### Train-Test Split

The 176 labeled nodes are split using an 80/20 stratified split:

- Training nodes: **140**
- Test nodes: **36**

The split is deterministic and uses a fixed random seed.

### Model

The model is a two-layer Relational Graph Convolutional Network using PyTorch Geometric:

```text
FastRGCNConv(72 -> 16)
ReLU
FastRGCNConv(16 -> 4)
```

Training setup:

- Model: **FastRGCNConv R-GCN**
- Optimizer: **Adam**
- Learning rate: **0.01**
- Loss: **Cross-entropy**
- Epochs: **100**

## Results

Final classification performance:

- Training accuracy: **97.86%**
- Test accuracy: **94.44%**

The model correctly classified **34 out of 36** test nodes.

The final training trace was:

```text
Epoch: 010, Loss: 0.5551, Train Acc: 0.9214, Test Acc: 0.9167
Epoch: 020, Loss: 0.2255, Train Acc: 0.9429, Test Acc: 0.9167
Epoch: 030, Loss: 0.1363, Train Acc: 0.9571, Test Acc: 0.9722
Epoch: 040, Loss: 0.1030, Train Acc: 0.9786, Test Acc: 0.9722
Epoch: 050, Loss: 0.0884, Train Acc: 0.9786, Test Acc: 0.9444
Epoch: 060, Loss: 0.0809, Train Acc: 0.9786, Test Acc: 0.9444
Epoch: 070, Loss: 0.0760, Train Acc: 0.9786, Test Acc: 0.9444
Epoch: 080, Loss: 0.0729, Train Acc: 0.9786, Test Acc: 0.9444
Epoch: 090, Loss: 0.0710, Train Acc: 0.9786, Test Acc: 0.9444
Epoch: 100, Loss: 0.0696, Train Acc: 0.9786, Test Acc: 0.9444
```

## Explanation with GNNExplainer

GNNExplainer is used to explain one correctly classified test node. The selected node is the correctly classified test node with the highest prediction confidence.

Selected node:

- Node index: **5710**
- Predicted class: **2**
- True class: **2**
- Confidence: **1.0**

Top edge importance scores:

```text
[0.8011, 0.7937, 0.7818, 0.7707, 0.7613, 0.7464, 0.1365, 0.1365, 0.1362, 0.1362]
```

Top edge indices:

```text
[2095, 9069, 21478, 1480, 2574, 1455, 5928, 25052, 27301, 15586]
```

## Human-Readable Explanation

The top-ranked edge indices are converted back into RDF triples. The most influential relationships include:

- Research Group -> carriesOut -> Project
- Project -> member -> Person
- Publication -> editor -> Person
- Research Area -> isWorkedOnBy -> Person
- Research Group -> publishes -> Publication
- Research Area -> dealtWithIn -> Project

These relationships are meaningful for the AIFB node classification task because the model predicts a person's affiliation. Information about research groups, projects, publications, and research areas can provide useful evidence about which affiliation class a person belongs to.

## Evaluation of the Explanation

The explanation is reasonable because the highest-scoring edges correspond to meaningful academic relationships in the AIFB knowledge graph. The important relationships are not random isolated connections; they describe project participation, publication information, research group activity, and research-area involvement.

The explanation is also human-readable because the numerical edge indices are converted back into RDF triples. This allows the explanation to be interpreted in the original RDF domain instead of only as numerical importance scores.

## Reproducibility

The experiment was made reproducible by fixing random seeds for:

- Python
- NumPy
- PyTorch
- model initialization
- train-test splitting
- GNNExplainer initialization

In addition, RDF nodes, relations, and triples are sorted before assigning numerical IDs. This is important because unordered Python sets can assign different node indices after a kernel restart. Without deterministic ordering, the train-test split, selected explanation node, and explanation scores may change.

After these changes, restarting the kernel and running all notebook cells produces the same final accuracy and selected explanation node in the same environment.

Small differences may still occur across different hardware, PyTorch versions, PyTorch Geometric versions, or CUDA backends.

## Environment

This project was developed using Miniconda with the environment name:

```text
xai26-mini
```

## How to Run the Project

### 1. Open a terminal

Open Anaconda Prompt, Miniconda Prompt, PowerShell, or a terminal inside VS Code.

### 2. Go to the project folder

```bash
cd "D:\UPB note\xai-mini-project"
```

If the project is placed in another location, replace the path above with the location of the project folder.

### 3. Create or activate the Conda environment

If the `xai26-mini` environment already exists, activate it:

```bash
conda activate xai26-mini
```

If the environment does not exist yet, create it first:

```bash
conda create -n xai26-mini python=3.12
conda activate xai26-mini
```

### 4. Install dependencies

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required libraries include:

- PyTorch
- PyTorch Geometric
- RDFLib
- NumPy
- Pandas
- Scikit-Learn
- Jupyter Notebook

### 5. Check the project structure

The project should contain the following important files and folders:

```text
xai-mini-project/
|-- aifb_rgcn.ipynb
|-- README.md
|-- requirements.txt
|-- data/
|   `-- Entities/
|       `-- aifb/
|           `-- raw/
|               |-- aifbfixed_complete.n3
|               `-- completeDataset.tsv
|-- figures/
|-- notebooks/
|-- report/
|-- results/
`-- src/
```

The notebook expects the dataset files to be available at:

```text
data/Entities/aifb/raw/aifbfixed_complete.n3
data/Entities/aifb/raw/completeDataset.tsv
```

### 6. Start Jupyter Notebook

```bash
jupyter notebook
```

### 7. Open the notebook

```text
aifb_rgcn.ipynb
```

### 8. Run all cells

In Jupyter Notebook, run:

```text
Kernel -> Restart Kernel and Run All Cells
```

This is recommended because the notebook contains reproducibility settings that should be applied from a clean kernel.

### 9. Expected output

After running all cells, the final training result should be:

```text
Epoch: 100, Loss: 0.0696, Train Acc: 0.9786, Test Acc: 0.9444
```

This corresponds to:

- Training accuracy: **97.86%**
- Test accuracy: **94.44%**
- Correctly classified test nodes: **34 out of 36**

The selected node for GNNExplainer should be:

```text
Selected node: 5710
Prediction: 2
True label: 2
Confidence: 1.0
```

The expected top edge indices are:

```text
[2095, 9069, 21478, 1480, 2574, 1455, 5928, 25052, 27301, 15586]
```

The expected top edge scores are:

```text
[0.8011, 0.7937, 0.7818, 0.7707, 0.7613, 0.7464, 0.1365, 0.1365, 0.1362, 0.1362]
```

### 10. What the notebook does

The notebook performs the following steps:

1. Imports libraries and sets random seeds.
2. Loads the AIFB RDF graph using RDFLib.
3. Computes RDF dataset statistics.
4. Converts RDF triples into a PyTorch Geometric graph.
5. Loads node labels from `completeDataset.tsv`.
6. Encodes labels and marks unlabeled nodes as `-1`.
7. Creates node features using `OneHotDegree`.
8. Splits labeled nodes into deterministic train/test masks.
9. Trains a two-layer FastRGCNConv model.
10. Evaluates train and test accuracy.
11. Selects a correctly classified test node.
12. Applies GNNExplainer.
13. Extracts top important edges.
14. Converts important edges back into human-readable RDF triples.
15. Interprets and evaluates the explanation.

### 11. Notes on reproducibility

To reproduce the reported results, run the full notebook from a restarted kernel. Running only selected cells can change the random state and may slightly affect GNNExplainer scores.

The experiment is deterministic in the tested environment because:

- Python, NumPy, and PyTorch seeds are fixed.
- Model initialization is seeded.
- The train/test split uses a fixed random seed.
- GNNExplainer initialization is seeded.
- RDF nodes, relations, and triples are sorted before numerical IDs are assigned.

Small differences may still occur on a different machine, PyTorch version, PyTorch Geometric version, or CUDA backend.

## Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- RDFLib
- NumPy
- Pandas
- Scikit-Learn
- Jupyter Notebook

## Conclusion

This project demonstrates how a Relational Graph Convolutional Network can be applied to an RDF knowledge graph for node classification and how GNNExplainer can be used to explain individual model predictions.

The trained R-GCN achieved a final test accuracy of **94.44%** on the AIFB dataset. The explanation results identify meaningful RDF relationships that influenced the selected prediction, making the model output more transparent and interpretable.
