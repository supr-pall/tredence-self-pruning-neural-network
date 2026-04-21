# Self-Pruning Neural Network with Straight-Through Estimator

**Tredence Analytics — AI Engineering Intern Case Study | Supriya Pallisetty**

## Unique Approach
Uses a Straight-Through Estimator (STE) for truly binary (0/1) gates — 
unlike soft sigmoid gates that never reach exactly zero.

## Files
- `self_pruning_network.py` — Full implementation (PrunableLinear, STE, training loop)
- `Tredence_CaseStudy_Supriya.docx` — Written report with theory, results & analysis

## Run
pip install torch torchvision matplotlib numpy
python self_pruning_network.py

## Tech Stack
Python · PyTorch · CIFAR-10 · Custom Autograd · L1 Sparsity Regularization
