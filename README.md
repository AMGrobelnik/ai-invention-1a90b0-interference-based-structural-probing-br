# AI Invention Research Repository

This repository contains artifacts from an AI-generated research project.

## Research Paper

[![Download PDF](https://img.shields.io/badge/Download-PDF-red)](https://github.com/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/blob/main/paper/paper.pdf) [![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)](https://github.com/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/tree/main/paper) [![Figures](https://img.shields.io/badge/Figures-3-blue)](https://github.com/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/tree/main/figures)

## Quick Start - Interactive Demos

Click the badges below to open notebooks directly in Google Colab:

### Jupyter Notebooks

| Folder | Description | Open in Colab |
|--------|-------------|---------------|
| `dataset_iter1_tudatasets_grap` | TUDatasets Graph Classification Benchmarks for ISP... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/blob/main/dataset_iter1_tudatasets_grap/demo/data_code_demo.ipynb) |
| `dataset_iter1_graph_expressiv` | Graph Expressiveness Benchmark for ISP Testing... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/blob/main/dataset_iter1_graph_expressiv/demo/data_code_demo.ipynb) |
| `experiment_iter2_isp_gnn_complex` | ISP-GNN: Complex-Valued Message Passing for Breaki... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/blob/main/experiment_iter2_isp_gnn_complex/demo/method_code_demo.ipynb) |
| `experiment_iter3_isp_gin_real_va` | ISP-GIN Real-Valued Ablation & Interference Isolat... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/blob/main/experiment_iter3_isp_gin_real_va/demo/method_code_demo.ipynb) |
| `evaluation_iter4_isp_gin_compreh` | ISP-GIN Comprehensive Final Evaluation: Hypothesis... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/blob/main/evaluation_iter4_isp_gin_compreh/demo/eval_code_demo.ipynb) |
| `experiment_iter4_isp_gin_vs_real` | ISP-GIN vs RealGIN-Aug vs GIN: Graph Classificatio... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/blob/main/experiment_iter4_isp_gin_vs_real/demo/method_code_demo.ipynb) |

### Research & Documentation

| Folder | Description | View Research |
|--------|-------------|---------------|
| `research_iter1_isp_lit_survey` | ISP Lit Survey... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/blob/main/research_iter1_isp_lit_survey/demo/research_demo.md) |
| `research_iter4_rwse_mpnn_vs_1` | RWSE+MPNN vs 1-WL... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br/blob/main/research_iter4_rwse_mpnn_vs_1/demo/research_demo.md) |

## Repository Structure

Each artifact has its own folder with source code and demos:

```
.
├── <artifact_id>/
│   ├── src/                     # Full workspace from execution
│   │   ├── method.py            # Main implementation
│   │   ├── method_out.json      # Full output data
│   │   ├── mini_method_out.json # Mini version (3 examples)
│   │   └── ...                  # All execution artifacts
│   └── demo/                    # Self-contained demos
│       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
├── <another_artifact>/
│   ├── src/
│   └── demo/
├── paper/                       # LaTeX paper and PDF
├── figures/                     # Visualizations
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/AMGrobelnik/ai-invention-1a90b0-interference-based-structural-probing-br.git
cd ai-invention-1a90b0-interference-based-structural-probing-br

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook exp_001/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
