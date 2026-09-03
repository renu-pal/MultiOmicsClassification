Multi Omics Classification 
===========================

A Machine Learning Framework for Benchmarking, Feature Selection, and Hyperparameter Optimization of Machine Learning Models for Multiomics Data.

Input
======
The tool requires following inputs :

- **X: Feature matrix file(s) (required)**
    - Tab-separated file.
    - Samples are represented as rows and features as columns.
    - Multiple Omics datasets can be added as input.
    - Each file includes a header row containing the feature names.
    - The first column must contain unique sample IDs.
    - Sample IDs must be consistent across the feature matrix/matrices and the target file.
    - For fair comparison between different omics datasets, the feature matrices should contain the same number of samples, ideally with the same sample IDs. If the datasets contain different numbers of samples, performance comparisons may be affected by differences in sample size.

- **Omics Type**
    - Select the omics type corresponding to each feature matrix (for example, RNA-seq, DNA methylation, proteomics, or other omics data).
    - Each feature matrix must have a corresponding omics name. 
    - When multiple omics datasets are provided, the omics type names are used to identify the different omics layers in the output and diagnostic plots.
    
- **Combine omics (optional)**
    - Enable this option to combines multiple omics feature matrices into a single feature matrix.
    - Samples are aligned using their sample IDs.
    - Only samples common to all provided omics datasets are retained.
    - Features from the different omics datasets are concatenated column-wise.

    
- **Y: Target file (required)**
    - Tab-separated file.
    - The first column must contain the sample IDs.
    - The sample IDs must match those in the feature matrix.
    - Must contain the target column specified by the user (e.g. "target" or "Sample_Condition").
    - The target column contains the class labels used for
      multiclass classification.
      
     Note:
      The feature matrix and target file do not need to contain
      exactly the same samples. Only samples present in both the
      feature matrix and target file are used.

      When multiple omics datasets are combined, only samples
      present in all omics datasets and in the target file are used.

- **Target Column**
    - Specify the column in the target file Y that contains the class labels to be predicted.

- **Feature counts (k) (required)**
    - Comma-separated list of numbers specifying how many top features to evaluate (e.g., 10,50,100).

- **Number of seeds (n) (default = 2)**
    - Number of random seeds used to repeat the classification.

- **Model type (required)**
    - rf (RandomForest)
    - xgb (XGBoost)
    - etc (ExtraTreesClassifier)
    - lgbm (LightGBM)
    - tabpfn (TabPFN model)

- **Sampling strategy (required)**
    - No Sampling
    - Random OverSampling
    - SMOTE
    - Random UnderSampling
    - NearMiss (v1)
    - NearMiss (v2)
    - NearMiss (v3)

- **Grid Search (optional)**
    - Default : No
    - Enables hyperparameter optimization
    - Can substantially increase runtime

Output
======
The tool creates following output files :

- **MultiClass Metric score**
    - This is the main output file depicting different classification metric scores.
    - For each seed, feature count (k), class (or class pair), and evaluation type (OvR/OvO), it reports:
        - ROC AUC – class separation ability
        - PR AUC – precision-recall performance
        - Precision
        - Recall
        - F1 Score
        - MCC (balanced classification metric)
    - It also includes Macro averages across all classes.

- **Diagnostic Plots**
    - A PNG file showing pairwise class comparisons.
    - For every class pair, it contains:
        - ROC curve
        - Precision–Recall curve
        - Predicted probability histogram or ECDF plot (multiple input datasets).
    - ROC Curve (Receiver Operating Characteristic): Shows how well the model distinguishes between classes by plotting the True Positive Rate against the False Positive Rate at different classification thresholds. A curve closer to the top-left corner indicates better class distinction.
    - Precision–Recall Curve: Shows the trade-off between Precision and Recall at different classification thresholds. It is particularly useful for evaluating class distinction when the classes are imbalanced.
    - Predicted Probability Histogram shows how many samples fall within each predicted probability range for each class in omic dataset. A bar at probability 0.8–0.9 with a height of 15 means that 15 samples have predicted probabilities between 0.8 and 0.9. If the probability distributions for the two classes are well separated, the model distinguishes the classes well.
    - ECDF(Empirical Cumulative Distribution Function) shows the cumulative proportion of samples at or below each predicted probability value for each class in omics dataset. Example, a cumulative proportion value of 0.6 at probability 0.7 means that 60% of samples have predicted probabilities ≤ 0.7. The greater the separation between the ECDF curves of the two classes, the better the model distinguishes the classes.


- **performance per feature plots**
    - This plot answers : "How does model performance change as I change the number of selected features?"
    - A PNG file showing model performance across different numbers of selected features (k).
    - It plots:
        - ROC AUC
        - PR AUC
    - for:
        - One-vs-Rest (OvR)
        - One-vs-One (OvO)
    - This helps identify the optimal number of features and compare performance across classes.


