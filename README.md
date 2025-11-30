**TF_MethylBind pipeline_0
project goal**
Train regression models to predict methylation-sensitive TF binding using PBM 8-mer Z-scores.

**what this pipeline does**
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


# TF_MethylBind

## goal
Predict methylation-driven changes in transcription factor binding using paired PBM 8-mer Z-scores.

## pipeline steps
1) Loads *unmethylated* and *5mCG-methylated* PBM summary files from each TF.  
2) Merges on 8-mer sequence and computes **Δbinding = Z(5mCG) − Z(Cytosine)** as the learning target.  
3) One-hot encodes 8-mers into 32 features (8 bases × 4 channels).  
4) Splits data to 70% train, 15% validation, 15% test (seed-controlled).  
5) Trains a **fully connected regression model (32 → 64 → 64 → 1, ReLU)** using MSE + Adam with early stopping.  
6) Evaluates on test set using **Pearson r, Spearman ρ, R²**.  
7) Saves model weights (`.pt`), splits (`.npz`), metrics (`.json`), and scatter plots (`.png`) in a tidy `outputs/TFNAME/` hierarchy.  
8) Summarises all TF metrics in `metrics_summary.json`.

## current metrics
:contentReference[oaicite:1]{index=1},  
:contentReference[oaicite:2]{index=2},  
:contentReference[oaicite:3]{index=3} for 8 TFs are stored in `metrics_summary.json`.

High performance is seen for CUX1 (r=0.742, R²=0.505), CUX2 (r=0.702, R²=0.491), MAX (r=0.842, R²=0.694), NFATC2 (r=0.753, R²=0.553), suggesting a strong methylation signal captured by a sequence-only FC model. NKX2.5 and POU5F1 show near-zero R², indicating weak or noisy Δbinding patterns not well explained by 8-mer sequence alone.

## usage
```bash
python TF_methylbind_pipeline.py \
  --input_dir /path/to/PBM_65536 \
  --output_dir ./outputs \
  --seed 42 \
  --n_jobs 4
