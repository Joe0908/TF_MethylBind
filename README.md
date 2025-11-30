# TF_MethylBind pipeline_0

## goal
Predict methylation-driven changes in transcription factor binding using paired PBM 8-mer Z-scores.

## pipeline steps
For each TF in the dataset (CUX1, CUX2, MAX, DLX3, NKX2.5, NFATC2, POU5F1, LHX9), the script:
1. Loads paired files
Finds one unmethylated file (*Zscore_Cytosine.txt) and one methylated file (*Zscore_5mCG.txt).
Merges them on the shared 8-mer.
Computes Δbinding = Z_5mCG − Z_Cytosine (supervised learning target).
2. Encodes sequences
Converts each 8-bp DNA probe into a 32-dim vector using one-hot encoding [A, C, G, T] × 8 positions.
3. splits data
70% train, 15% validation, 15% test (shuffled, seeded for reproducibility).
4. Trains a regression network
Architecture: 32 → 64 → 64 → 1, with ReLU activations.
Loss: MSE, optimizer: Adam.
Early stopping on validation loss (patience = 5 epochs).
5. Evaluates performance
Computes correlation + regression quality on the test set: Pearson r, Spearman ρ, R² (true Δbinding vs predicted Δbinding).
Generates scatter plots (y = x reference line) for quick visual inspection.

#check results in yuqitao/outputs
