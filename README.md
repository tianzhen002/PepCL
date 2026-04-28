# PepCL

PepCL is a two-stage contrastive learning framework for antimicrobial peptide activity prediction. The framework first uses a contrastive encoder to refine residue-level peptide representations and then applies a downstream neural classifier for binary peptide activity prediction.

This repository contains the downstream prediction code of PepCL, including feature preprocessing, model definition, training, validation, testing, and metric calculation.

## Overview

PepCL is designed for peptide activity prediction tasks such as antimicrobial peptide prediction and related functional peptide classification tasks. The downstream prediction module takes residue-level pretrained protein language model features and contrastively refined representations as input, and outputs the probability that a peptide belongs to the positive class.

The downstream classifier includes:

- preprocessing of peptide sequences and pretrained residue-level features
- contrastive feature generation from a pretrained contrastive encoder checkpoint
- a neural prediction module based on convolutional embedding, rotary positional encoding, Transformer encoding, residual connection, adaptive average pooling, and binary classification
- evaluation with common binary classification metrics

## Repository structure

```text
PepCL/
├── model.py
├── preprocess.py
├── train.py
├── valid_metrices.py
├── contrastive_petrain/
│   ├── dataloader.py
│   ├── main.py
│   └── model.py
└── README.md
