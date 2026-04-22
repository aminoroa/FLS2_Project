# 1. Data structure
FLS2.fasta, flg22.fasta, and coreceptor.fasta files contain sequences of FLS2s, flg22, and coreceptors, respectively.

Input files related to FLS2-flg22-coreceptor complexes for AlphaFold3 calculations are provided in ~/AF3_inputs

# 2. Connecting to MSU HPCC

```
ssh <usernanme>@hpcc.msu.edu
<password>
sbatch af3_run.sb
```

# 1. AlphaFold3 Output
To get the AlphaFold3 predicted structure of complexes, a batch job script was submitted to HPCC

```
sbatch run_af3_array.sb
```
Note: This job script has a constraint for intel21. When I submitted the job script, it was on the queue with a BadConstraint flag but it worked. It seems the Scheduler "Magic" Trick Look closely at the PARTITION column for the running jobs versus the pending jobs: Running Tasks were running in general-s (General Short) while Pending Tasks were still sitting in general-l (General Long) with a BadConstraint flag.

To maintain the reporducibility, three runs per input files were done by sprcifying three (1, 2, and 3) modelSeeds in each input .json files.

# 3. Understanding the Output Structure

You’ll notice that for each of your 3 seeds, there are actually 5 samples (numbered 0 through 4), totaling 15 individual structure predictions.

Here is the breakdown of the most important files:

ranking_scores.csv (The Shortcut): This is often the best place to start. It provides a simple table that ranks all 15 models by their Aggregate Score. It tells you exactly which seed and which sample produced the "best" fold.

atfls2_102_atbak1_summary_confidences.json (The Detail): This is indeed the main summary. It contains the high-level metrics (pLDDT, iPTM, etc.) for the top-ranked model.

atfls2_102_atbak1_model.cif: This is the 3D coordinate file for your #1 ranked prediction. This is the file you would open in ChimeraX or PyMOL.

seed-X_sample-Y Folders: Inside these, you will find the .cif files and full confidence data for every single one of the 15 attempts. If the top model looks weird, you can dig into these to see the variations.

# 4. summary of scores
These are the global metrics used to judge if the model is a "success" or a "hallucination."

iptm (0.85): Interface Predicted Template Modeling score. This is your most important number. It measures the confidence of the interaction between all chains.

Interpretation: >0.7 is high confidence. At 0.85, this is a very reliable interface prediction.

ptm (0.86): Predicted Template Modeling score. This measures the confidence in the overall global fold/topology of the entire assembly.

ranking_score (0.86): This is the weighted average AF3 uses to rank its 15 samples. For complexes, it is calculated as:

ranking_score=(0.8×iptm)+(0.2×ptm)
2. The Chain Matrices (Interface Details)

Since you have three chains (likely FLS2, the peptide, and BAK1), AF3 provides a 3x3 matrix to show how each pair interacts.

chain_pair_iptm (Confidence Matrix)

This tells you which specific interaction AF3 is most certain about.

[!NOTE]
Assuming your chains are A (FLS2), B (Peptide), and C (BAK1):

Chain A	Chain B	Chain C
Chain A	0.86	0.44	0.90
Chain B	0.44	0.06	0.27
Chain C	0.90	0.27	0.89
Confidence in A-C (0.90): This is extremely high. AF3 is very sure about the direct interaction between FLS2 and BAK1.

Confidence in A-B (0.44): This is moderate/low. It suggests the peptide's binding to FLS2 might be more flexible or have a smaller interface area.

chain_pair_pae_min (Distance Error)

This is the Predicted Aligned Error (PAE) in Ångströms ( 
A
˚
 ). Lower is better.

A-C (1.89 / 1.96 Å): Very low error. The relative positions of FLS2 and BAK1 are predicted with high precision.

A-B (10.89 Å): Higher error. There is about 10 Å of uncertainty regarding exactly where the peptide sits relative to the main receptor.

3. Physical Health Metrics

fraction_disordered (0.02): Only 2% of your complex is considered "floppy" or unstructured. This is great; it means the complex is mostly one solid, well-folded unit.

has_clash (0.0): No atoms are physically overlapping. The model is physically "legal."

# 5. Extracting data

You can generate af3 data or download our data from ...
run the scrip below to extract data ...

```
python extract_af3_data.py
```
data will be provided to af3_output_data.xlsx file. Transfer data to Sequence.xlsx file manually

files below where generated as just cleaner version of Sequences.xlsx file
    Sequences_cleaned_original.xlsx: it is directly the cleaned version with no manipulation
    Sequences_cleaned_1.xlsx: in this one perception is partial or complete immune response vs No perception is no immune response
    Sequences_cleaned_2.xlsx: in this one perception is activation or antogonism of receptor vs No perception is true evasion

# 6. scores visualizaiton 
first visualizing score range

```
python perception_scores.py Name directory_1 directory_2
```
Note: directory_1 should be the actual path/name of your input Excel file (e.g., input_data.xlsx).

```
python perception_scores.py original /Users/aminoroa/FLS2_Project/input_data/Sequences_cleaned_original.xlsx /Users/aminoroa/FLS2_Project/output_results
python plot_box.py /Users/aminoroa/FLS2_Project/output_results /Users/aminoroa/FLS2_Project/output_results
```

```
python classification_scores.py original /Users/aminoroa/FLS2_Project/input_data/Sequences_cleaned_original.xlsx /Users/aminoroa/FLS2_Project/output_results

python classification_box_plot.py /Users/aminoroa/FLS2_Project/output_results/original_classification_iptm.csv /Users/aminoroa/FLS2_Project/output_plots
```


# 5. ML Model

dataset is very small and has imbalanced biological dataset (n = 116, only 11 antagonists). This immediately constrains what will work reliably. Deep learning is not appropriate here—we need simple, interpretable, and regularized models with careful validation.

a) Data Preparation: Load your data, drop the 14 complexes with "No Labeling (-)", and isolate your features (scores) from your targets (classifications).

b) Stratified Split: Split the data into the 85% Training Set and the 15% Test Set. Lock the 15% Test Set away. Do not touch it again until the very end.

c) Feature Ranking: Run your feature ranking methods (Tree-based, RFE, Permutation, ANOVA, RFECV) strictly on the 85% Training Set to identify the most predictive scores.

d) Model Training & Validation: Train your models (Random Forest, SVM, Logistic Regression, Gradient Boosting) on the 85% Training Set. Use K-Fold Cross-Validation within this 85% to tune your settings and evaluate which model is learning the best.

e) Final Testing: Once you have selected your absolute best model, run it exactly once on the unseen 15% Test Set to see its true real-world accuracy.

```
python 01_data_prep_split.py ML_run /Users/aminoroa/FLS2_Project/input_data/Sequences_cleaned_original.xlsx /Users/aminoroa/FLS2_Project/ml_data

python 02_feature_ranking.py ML_run /Users/aminoroa/FLS2_Project/ml_data

python 03_model_evaluation.py ML_run /Users/aminoroa/FLS2_Project/ml_data

python 04_final_testing.py ML_run /Users/aminoroa/FLS2_Project/ml_data

```


# 7. New predictions
How to use it:

Open the script in your text editor.

Update the six numbers in the new_scores dictionary with the data from your new AlphaFold3 run.

Save the file and run it from your terminal:

```
python predict_new_complex.py
```

