# 1. Data structure
FLS2.fasta, flg22.fasta, and coreceptor.fasta files contain sequences of FLS2s, flg22, and coreceptors, respectively.

Input files related to FLS2-flg22-coreceptor complexes for AlphaFold3 calculations are provided in ~/AF3_inputs

# 2. Connecting to MSU HPCC
**Add explanations here**

# 1. AlphaFold3 Output
To get the AlphaFold3 predicted structure of complexes, a batch job script was submitted to HPCC

```
cd ~
sbatch af3_rub.sb
```

