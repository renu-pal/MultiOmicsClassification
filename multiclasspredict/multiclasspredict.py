import argparse
import os
import copy
import itertools
import warnings


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from lightgbm import LGBMClassifier
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    f_classif,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold
)
from sklearn.metrics import (
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, LabelBinarizer, LabelEncoder, StandardScaler

from itertools import combinations
from sklearn.svm import SVC

# from tabpfn import TabPFNClassifier
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

from xgboost import XGBClassifier

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


def split_classes(X, y):
    return {
        (c1, c2): (X[(y == c1) | (y == c2)], y[(y == c1) | (y == c2)])
        for c1, c2 in itertools.combinations(np.unique(y), 2)
    }


def ovo_and_ova_multiclass_auc(X, y, base_clf, p_grid, random_state, model_name):
    results = {}
    plot_data = []
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    # Stratified K-Folds
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    ####################
    # One-vs-Rest Classification
    ####################
    print("Performing One vs Rest classification")

    # checking grid search enabled or not
    if p_grid is not None:
        ovr_clf = GridSearchCV(
            estimator=OneVsRestClassifier(base_clf),
            param_grid=p_grid,
            cv=inner_cv,
            scoring="roc_auc_ovr",
        )
    else:
        ovr_clf = OneVsRestClassifier(base_clf)

    y_score = cross_val_predict(ovr_clf, X, y_encoded, cv=outer_cv, method="predict_proba")
    y_pred = np.argmax(y_score, axis=1)

    # Per-class metrics for OvR
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_mcc = []

    for idx, cls in enumerate(class_names):
        y_bin = (y_encoded == idx).astype(int)
        cls_score = y_score[:, idx]

        # Ensure minority class is positive
        if np.sum(y_bin) > np.sum(1 - y_bin):
            y_bin = 1 - y_bin
            cls_score = 1 - cls_score

        y_pred_bin = (y_pred == idx).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_bin, y_pred_bin, average="binary")
        mcc = matthews_corrcoef(y_bin, y_pred_bin)
        prec_curve, rec_curve, _ = precision_recall_curve(y_bin, cls_score)
        pr_auc_val = auc(rec_curve, prec_curve)
        roc_auc_val = roc_auc_score(y_bin, cls_score)

        results[f"{cls} vs Rest - Precision"] = precision
        results[f"{cls} vs Rest - Recall"] = recall
        results[f"{cls} vs Rest - F1"] = f1
        results[f"{cls} vs Rest - MCC"] = mcc
        results[f"{cls} vs Rest - PR AUC"] = pr_auc_val
        results[f"{cls} vs Rest - ROC AUC"] = roc_auc_val

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
        per_class_mcc.append(mcc)

    # Macro metrics OvR

    macro_ovr_auc = np.mean([results[f"{cls} vs Rest - ROC AUC"] for cls in class_names])
    macro_ovr_precision = np.mean(per_class_precision)
    macro_ovr_recall = np.mean(per_class_recall)
    macro_ovr_f1 = np.mean(per_class_f1)
    macro_ovr_mcc = np.mean(per_class_mcc)
    macro_ovr_pr_auc = np.mean([results[f"{cls} vs Rest - PR AUC"] for cls in class_names])

    results["OvR Macro ROC AUC"] = macro_ovr_auc
    results["OvR Macro Precision"] = macro_ovr_precision
    results["OvR Macro Recall"] = macro_ovr_recall
    results["OvR Macro F1"] = macro_ovr_f1
    results["OvR Macro MCC"] =  macro_ovr_mcc
    results["OvR Macro PR AUC"] = macro_ovr_pr_auc

    '''
    print(f"Macro ROC AUC (OvR): {macro_ovr_auc:.4f}")
    print(f"Macro Precision (OvR): {macro_ovr_precision:.4f}")
    print(f"Macro Recall (OvR): {macro_ovr_recall:.4f}")
    print(f"Macro F1 (OvR): {macro_ovr_f1:.4f}")
    print(f"Macro MCC (OvR): {macro_ovr_mcc:.4f}")
    print(f"Macro PR AUC (OvR): {macro_ovr_pr_auc:.4f}")  '''

    # avoiding  meaningless computation as OvO metrics won’t make sense with TabPFN
    if model_name == "tabpfn":
        print("Skipping One-vs-One metrics for TabPFN")
    else:
        
        ####################
        # One-vs-One Classification
        ####################
        print("Performing One vs One classification")

        ovo_auc = {}
        ovo_precision = {}
        ovo_recall = {}
        ovo_f1 = {}
        ovo_mcc = {}

        for c1, c2 in combinations(range(len(class_names)), 2): 
            mask = np.isin(y_encoded, [c1, c2]) 
            X_pair, y_pair = X[mask], y_encoded[mask] 
    
            # checking grid search enabled or not
            if p_grid is not None:
                ovo_clf = GridSearchCV(
                    estimator=base_clf,
                    param_grid={k.replace("estimator__", ""): v for k, v in p_grid.items()},
                    cv=inner_cv,
                    scoring="roc_auc"
                )
            else:
                ovo_clf = base_clf
                
            y_score_pair = cross_val_predict(ovo_clf, X_pair, y_pair, cv=outer_cv, method="predict_proba") 
            
            # Identify minority 
            
            vals, counts = np.unique(y_pair, return_counts=True) 
            minority = vals[np.argmin(counts)] 
            minority_idx = np.where([c1, c2] == minority)[0][0] 
            
            y_bin = (y_pair == minority).astype(int) 
            y_score_cls = y_score_pair[:, minority_idx] 
            
            
            
            # Ensure minority positive 
            
            if np.sum(y_bin) > np.sum(1 - y_bin): 
                y_bin = 1 - y_bin 
                y_score_cls = 1 - y_score_cls 
                
            y_pred_bin = (np.argmax(y_score_pair, axis=1) == minority_idx).astype(int)
                                                                                 
            precision, recall, f1, _ = precision_recall_fscore_support(y_bin, y_pred_bin, average="binary") 
            mcc = matthews_corrcoef(y_bin, y_pred_bin) 
            prec_curve, rec_curve, _ = precision_recall_curve(y_bin, y_score_cls) 
            pr_auc_val = auc(rec_curve, prec_curve) 
            roc_auc_val = roc_auc_score(y_bin, y_score_cls)
            
            pair_name = f"{le.inverse_transform([c1])[0]} vs {le.inverse_transform([c2])[0]}" 
            
            results[f"{pair_name} - Precision"] = precision 
            results[f"{pair_name} - Recall"] = recall 
            results[f"{pair_name} - F1"] = f1 
            results[f"{pair_name} - MCC"] = mcc 
            results[f"{pair_name} - PR AUC"] = pr_auc_val 
            results[f"{pair_name} - ROC AUC"] = roc_auc_val 
            
            ovo_auc[(c1, c2)] = roc_auc_val 
            ovo_precision[(c1, c2)] = precision 
            ovo_recall[(c1, c2)] = recall 
            ovo_f1[(c1, c2)] = f1 
            ovo_mcc[(c1, c2)] = mcc 
            
            # for plotting 
            plot_data.append({
                "class_a": le.inverse_transform([c1])[0],
                "class_b": le.inverse_transform([c2])[0],
                "pair_name": pair_name,
                "y_true": y_bin.copy(),
                "y_prob": y_score_cls.copy(),
                "roc_auc": roc_auc_val,
                "pr_auc": pr_auc_val
            })
            
        # Macro metrics OvO 
        macro_ovo_auc = np.mean(list(ovo_auc.values()))
        macro_ovo_precision = np.mean(list(ovo_precision.values()))
        macro_ovo_recall = np.mean(list(ovo_recall.values())) 
        macro_ovo_f1 = np.mean(list(ovo_f1.values())) 
        macro_ovo_mcc = np.mean(list(ovo_mcc.values())) 
        macro_ovo_pr_auc = np.mean([results[k] for k in results if "vs" in k and "PR AUC" in k]) 
    
        results["OvO Macro ROC AUC"] =  macro_ovo_auc
        results["OvO Macro Precision"] = macro_ovo_precision
        results["OvO Macro Recall"] = macro_ovo_recall
        results["OvO Macro F1"] = macro_ovo_f1
        results["OvO Macro MCC"] = macro_ovo_mcc
        results["OvO Macro PR AUC"] =  macro_ovo_pr_auc
    
        ''' 
        print(f"Macro ROC AUC (OvO): {macro_ovo_auc:.4f}")
        print(f"Macro Precision (OvO): {macro_ovo_precision:.4f}")
        print(f"Macro Recall (OvO): {macro_ovo_recall:.4f}")
        print(f"Macro F1 (OvO): {macro_ovo_f1:.4f}")
        print(f"Macro MCC (OvO): {macro_ovo_mcc:.4f}")
        print(f"Macro PR AUC (OvO): {macro_ovo_pr_auc:.4f}") '''
    
    return results, plot_data


def repeat_clf(n_seeds, ks, X, y, label, model, sampling_strategy, use_grid=False):

    print("features(ks): ", ks)
    print("seeds: ", n_seeds)

    # Define sampling strategies
    sampling_strategies = {
        "No Sampling": None,
        "Random OverSampling": RandomOverSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42),
        "Random UnderSampling": RandomUnderSampler(random_state=42),
        "NearMiss (v1)": NearMiss(version=1),
        "NearMiss (v2)": NearMiss(version=2),
        "NearMiss (v3)": NearMiss(version=3),
    }

    # If the selected strategy is not in the dictionary, use "No Sampling"
    sampler = sampling_strategies.get(sampling_strategy, None)

    seed_results = {}

    for seed in range(n_seeds):

        ks_results = {}
        for k in ks:

            print(f"CV for seed {seed} and {k} features")

            # Create a Random Forest Classifier
            rf = RandomForestClassifier(random_state=seed)

            # Create a SelectFromModel using the Random Forest Classifier
            selector = SelectFromModel(rf, max_features=k)

            if model == "rf":
                ml_model = rf
                ml_model_grid = {
                    "estimator__classification__n_estimators": [100, 300, 500],  # Number of trees in the forest
                    "estimator__classification__max_depth": [None, 10, 20, 30],  # tree depth
                    "estimator__classification__max_features": ["sqrt", "log2"],  # Feature selection strategy
                    "estimator__classification__criterion": ["entropy"],  # Split criterion
                    "estimator__classification__min_samples_leaf": [1, 2, 4],  # Minimum samples per leaf
                }
            elif model == "xgb":
                ml_model = XGBClassifier(
                    use_label_encoder=False, eval_metric="logloss", random_state=seed
                )
                ml_model_grid = {
                    "estimator__classification__n_estimators": [100, 300, 500], 
                    "estimator__classification__gamma": [0, 0.1, 0.3],  # min loss reduction
                    "estimator__classification__max_depth": [3, 5, 7], 
                    "estimator__classification__learning_rate": [0.01, 0.05, 0.1], #  step size
                }
            elif model == "etc":
                ml_model = ExtraTreesClassifier(random_state=seed)
                ml_model_grid = {
                    "estimator__classification__n_estimators": [100, 300, 500],
                    "estimator__classification__max_depth": [None, 10, 20],       #  tree depth
                    "estimator__classification__max_features": ["sqrt", "log2"],  #  features per split
                    "estimator__classification__min_samples_leaf": [1, 2, 4],     #  min leaf samples
                    
                }
            elif model == "lgbm":
                ml_model = LGBMClassifier(random_state=seed, verbose=-1)
                ml_model_grid = {
                    "estimator__classification__n_estimators": [100, 300, 500],  
                    "estimator__classification__learning_rate": [0.01, 0.05, 0.1],
                    "estimator__classification__num_leaves": [31, 63, 127],      # leaves per tree
                            
                }
            elif model == "tabpfn":
                from tabpfn import TabPFNClassifier
                ml_model = TabPFNClassifier(
                    device="cpu",  
                    n_estimators=32  # default
                )
                ml_model_grid = None  # TabPFN does not use GridSearch

            # If there is a sampler, include it in the pipeline
            steps = []
            if sampler:
                steps.append(("sampling", sampler))
            steps.append(("feature_selection", selector))
            steps.append(("classification", ml_model))

            # Create a pipeline with feature selection, sampling, and classification
            pipeline = Pipeline(steps=steps)

            ###########################

            # Run the classification with the sampling strategy
            if use_grid:
                results, plot_data = ovo_and_ova_multiclass_auc(
                    X, y, pipeline, ml_model_grid, random_state=seed, model_name=model
                )
            else:
                results, plot_data = ovo_and_ova_multiclass_auc(
                    X, y, pipeline, None, random_state=seed, model_name=model
                )
                       
            # print(results)

            ks_results[k] = {
                "results": results,
                "plot_data": plot_data,
                "Label": label,
                "Model": model,
                "Sampling_Strategy": sampling_strategy,
                "Grid_Search": use_grid,
            }

        seed_results[seed] = copy.copy(ks_results)

    return seed_results


def store_results(seed_results, output):

    # Flatten the nested dictionary into a DataFrame
    '''df = pd.DataFrame(
        {
            (outer_key, inner_key): values
            for outer_key, inner_dict in seed_results.items()
            for inner_key, values in inner_dict.items()
        }
    ).T

    # '''
    
    final_results = []
    metrics = ["ROC AUC", "Precision", "Recall", "F1", "MCC", "PR AUC"]
    
    for seed, ks_results in seed_results.items():
        for k, result_info in ks_results.items():
            result = result_info["results"]
            model = result_info["Model"]
            sampling_strategy = result_info["Sampling_Strategy"]
            label = result_info["Label"]
            grid_search = result_info.get("Grid_Search", False)
                        
            # Determine Class and Type
           
            groups = set()
            for key in result.keys():
                if "Macro" in key:
                    groups.add(("Macro", "OvR" if "OvR" in key else "OvO"))
                elif "vs Rest" in key:
                    groups.add((key.split(" vs Rest")[0], "OvR"))
                else:
                    groups.add((key.split(" - ")[0], "OvO"))

            # assign metric values according to class and type
            for class_name, type_name in groups:

                metric_values = {}
            
                for metric in metrics:
                    metric_key = None
            
                    for key in result.keys():
                        if class_name == "Macro":
                            if metric in key and "Macro" in key and type_name in key:
                                metric_key = key
                                break
                        elif type_name == "OvR":
                            if metric in key and f"{class_name} vs Rest" in key:
                                metric_key = key
                                break
                        else:  # OvO
                            if metric in key and "vs" in key and class_name in key:
                                metric_key = key
                                break
            
                    metric_values[metric] = result[metric_key] if metric_key else np.nan

                final_results.append({
                    "Seed": seed,
                    "Features (k)": k,
                    "Label": label,
                    "Model": model,
                    "Hyper Parameter tuning" : grid_search,
                    "Sampling_Strategy": sampling_strategy,
                    "Class": class_name,
                    "Type": type_name,
                    **metric_values
                })
                
    df = pd.DataFrame(final_results)

    df.to_csv(output, sep="\t", mode='a', header=not os.path.exists(output) or os.path.getsize(output) == 0, index=False)


def run_classification(X, y, ks, n_seeds,output, label, model, sampling_strategy, use_grid=False):

    '''# Ensure ks does not exceed the number of columns in X
    max_features = len(X.columns)
    ks = [k for k in ks if k <= max_features]
    if max_features not in ks:
        ks.append(max_features)'''

    seed_results = repeat_clf(n_seeds, ks, X, y, label, model, sampling_strategy, use_grid=use_grid)
    store_results(seed_results, output)
    
    return seed_results

        
def plot_pairwise_diagnostics(seed_results, diagnostic_plot):
    
    # store combined predictions
    combined_plot_data = {}
    
    for seed in seed_results:

        for k in seed_results[seed]:
            
            pair_plot_data = seed_results[seed][k]["plot_data"]
    
            for item in pair_plot_data:
    
                pair = (item["class_a"], item["class_b"])
    
                if pair not in combined_plot_data:
                    combined_plot_data[pair] = {
                        "class_a": item["class_a"],
                        "class_b": item["class_b"],
                        "y_true": [],
                        "y_prob": [],
                        "roc_aucs": [],
                        "pr_aucs": []
                    }
    
                combined_plot_data[pair]["y_true"].extend(item["y_true"])
                combined_plot_data[pair]["y_prob"].extend(item["y_prob"])
                # roc auc
                roc_auc = roc_auc_score(
                    item["y_true"],
                    item["y_prob"]
                )

                combined_plot_data[pair]["roc_aucs"].append(roc_auc)

                # pr auc
                precision, recall, _ = precision_recall_curve(
                    item["y_true"],
                    item["y_prob"]
                )

                pr_auc = auc(recall, precision)

                combined_plot_data[pair]["pr_aucs"].append(pr_auc)

    
    
    final_plot_data = []
    
    for pair, item in combined_plot_data.items():

        item["y_true"] = np.asarray(item["y_true"])
        item["y_prob"] = np.asarray(item["y_prob"])

        item["mean_auc"] = np.mean(item["roc_aucs"])
        item["std_auc"] = np.std(item["roc_aucs"], ddof=1)

        item["mean_pr_auc"] = np.mean(item["pr_aucs"])
        item["std_pr_auc"] = np.std(item["pr_aucs"], ddof=1)

        final_plot_data.append(item)

    n_pairs = len(final_plot_data)

    fig, axes = plt.subplots(n_pairs, 3, figsize = (18, 4*n_pairs))

    if n_pairs == 1:
        axes = np.array([axes])

    for ax_row, item in zip(axes, final_plot_data):
        class_a = item["class_a"]
        class_b = item["class_b"]
        y_true = item["y_true"]
        y_prob = item["y_prob"]
        #pair_name = item["pair_name"]

        ax_roc, ax_pr, ax_hist = ax_row

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_roc.plot(
            fpr,
            tpr,
            lw=2,
            label=f"ROC-AUC = {item['mean_auc']:.2f} ± {item['std_auc']:.2f}"
        )

        ax_roc.plot(
            [0, 1],
            [0, 1],
            "k--",
            alpha=0.5
        )

        ax_roc.set_title(f"{class_a} vs {class_b}\nROC Curve")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()


        # PR curve

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_prob
        )

        ax_pr.plot(
            recall,
            precision,
            lw=2,
            label=f"PR-AUC = {item['mean_pr_auc']:.2f} ± {item['std_pr_auc']:.2f}"
        )

        baseline = np.mean(y_true)

        ax_pr.plot(
            [0, 1],
            [baseline, baseline],
            "k--",
            alpha=0.5
        )

        ax_pr.set_title(f"{class_a} vs {class_b}\nPrecision-Recall Curve")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend()

        # Histogram
        ax_hist.hist(
            [
                y_prob[y_true == 0],
                y_prob[y_true == 1]
            ],
            bins=20,
            alpha=0.7,
            label=[
                f"Negative ({class_a})",
                f"Positive ({class_b})"
            ]
        )

        ax_hist.set_title(f"{class_a} vs {class_b}\nPredicted Probabilities")
        ax_hist.set_xlabel("Predicted Probability")
        ax_hist.set_ylabel("Count")
        ax_hist.legend()

    fig.tight_layout()
    fig.savefig(diagnostic_plot, format="png", dpi=300, bbox_inches="tight")
    

# diagnostic plot for multiple omics datasets
def plot_pairwise_diagnostics_multiple(all_seed_results, diagnostic_plot):
    
    # combined predictions
    combined_plot_data = {}
    
    for assay, seed_results in all_seed_results.items():

        for seed in seed_results:

            for k in seed_results[seed]:
                
                pair_plot_data = seed_results[seed][k]["plot_data"]
        
                for item in pair_plot_data:
        
                    pair = (item["class_a"], item["class_b"])
        
                    if pair not in combined_plot_data:
                        combined_plot_data[pair] = {}
                    
                    if assay not in combined_plot_data[pair]:
                        combined_plot_data[pair][assay] = {
                            "class_a": item["class_a"],
                            "class_b": item["class_b"],
                            "y_true": [],
                            "y_prob": [],
                            "roc_aucs": [],
                            "pr_aucs": []
                        }
        
                    combined_plot_data[pair][assay]["y_true"].extend(item["y_true"])
                    combined_plot_data[pair][assay]["y_prob"].extend(item["y_prob"])

                    # roc auc
                    roc_auc = roc_auc_score(
                        item["y_true"],
                        item["y_prob"]
                    )

                    combined_plot_data[pair][assay]["roc_aucs"].append(roc_auc)

                    # pr auc
                    precision, recall, _ = precision_recall_curve(
                        item["y_true"],
                        item["y_prob"]
                    )

                    pr_auc = auc(recall, precision)

                    combined_plot_data[pair][assay]["pr_aucs"].append(pr_auc)


    n_pairs = len(combined_plot_data)

    fig, axes = plt.subplots(
        n_pairs, 3,
        figsize=(18, 4*n_pairs)
    )

    if n_pairs == 1:
        axes = np.array([axes])
    
    for ax_row, (pair, assay_data) in zip(axes, combined_plot_data.items()):

        ax_roc, ax_pr, ax_ecdf = ax_row

        class_a, class_b = pair
    
        for (assay, item) in assay_data.items():

            y_true = np.asarray(item["y_true"])
            y_prob = np.asarray(item["y_prob"])

            mean_auc = np.mean(item["roc_aucs"])
            std_auc = np.std(item["roc_aucs"], ddof=1)

            mean_pr_auc = np.mean(item["pr_aucs"])
            std_pr_auc = np.std(item["pr_aucs"], ddof=1)

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)

            ax_roc.plot(
                fpr,
                tpr,
                lw=2,
                label=f"{assay}: ROC-AUC = {mean_auc:.2f} ± {std_auc:.2f}"
            )

            # PR curve
            precision, recall, _ = precision_recall_curve(
                y_true,
                y_prob
            )

            ax_pr.plot(
                recall,
                precision,
                lw=2,
                label=f"{assay}: PR-AUC = {mean_pr_auc:.2f} ± {std_pr_auc:.2f}"
            )
            
           
           
            # ECDF
            # An ECDF (Empirical Cumulative Distribution Function) plot shows you the distribution of predicted probabilities.
            # its answers : For a given probability threshold, what fraction of samples have predicted probabilities below that threshold?
            # Cumulative proportion means: what fraction of the samples have a predicted probability less than or equal to a given probability.
            for class_value, class_name, class_label, linestyle in [ (0, class_a, "Negative", "-"),(1, class_b, "Positive", "--") ]:


                probs = np.sort(y_prob[y_true == class_value])

                ecdf = np.arange(1, len(probs) + 1) / len(probs)

                ax_ecdf.plot(
                    probs,
                    ecdf,
                    linestyle=linestyle,
                    lw=2,
                   label=f"{assay}: {class_label} ({class_name})"
                )

        # ROC baseline
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)

        ax_roc.set_title(f"{class_a} vs {class_b}\nROC Curve")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()

        # PR baseline
        baseline = np.mean(
            next(iter(assay_data.values()))["y_true"]
        )

        ax_pr.plot([0, 1], [baseline, baseline], "k--", alpha=0.5)

        ax_pr.set_title(f"{class_a} vs {class_b}\nPrecision-Recall Curve")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend()

        # ECDF
        ax_ecdf.set_title(f"{class_a} vs {class_b}\nPredicted Probability ECDF")
        ax_ecdf.set_xlabel("Predicted Probability")
        ax_ecdf.set_ylabel("Cumulative Proportion")
        ax_ecdf.set_xlim(0, 1)
        ax_ecdf.set_ylim(0, 1)
        ax_ecdf.legend()

    fig.tight_layout()
    fig.savefig(
        diagnostic_plot,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

def plot_model_performance_by_features(result_file, plot_per_feature):
    
    df = pd.read_csv(result_file, sep="\t")
    df.columns = df.columns.str.strip()

    # Keep ONLY non-macro classes
    if "Class" in df.columns:
        df = df[df["Class"] != "Macro"]

    df["Features (k)"] = pd.to_numeric(
        df["Features (k)"],
        errors="coerce"
    )

    metrics = ["ROC AUC", "PR AUC"]
    feature_sets = sorted(df["Features (k)"].unique())

    multiple_datasets = df["Label"].nunique() > 1

   
    # MULTIPLE DATASETS
    
    if multiple_datasets:

        fig, axes = plt.subplots(
            len(feature_sets) * 2,
            2,
            figsize=(16, 6 * len(feature_sets)),
            sharey=True
        )

        for row, feature in enumerate(feature_sets):

            for col, t in enumerate(["OvR", "OvO"]):

                df_plot = df[
                    (df["Features (k)"] == feature) &
                    (df["Type"] == t)
                ].copy()

                for metric_row, metric in enumerate(metrics):

                    ax = axes[row * 2 + metric_row, col]

                    sns.barplot(
                        data=df_plot,
                        x="Class",
                        y=metric,
                        hue="Label",
                        edgecolor="black",
                        errorbar="sd",
                        ax=ax
                    )

                    ax.set_title(
                        f"{t} - {feature} Features\n{metric}",
                        fontsize=14
                    )

                    ax.set_xlabel("Class")
                    ax.set_ylabel("AUC")
                    ax.set_ylim(0, 1)
                    ax.tick_params(axis="x", rotation=30)

                    if row == 0 and col == 0 and metric_row == 0:
                        ax.legend(
                            title="Dataset",
                            bbox_to_anchor=(1.02, 1),
                            loc="upper left"
                        )
                    else:
                        ax.get_legend().remove()

    
    # SINGLE DATASET
   
    else:

        fig, axes = plt.subplots(
            len(feature_sets),
            2,
            figsize=(16, 6 * len(feature_sets)),
            sharey=True
        )

        if len(feature_sets) == 1:
            axes = np.array([axes])

        for row, feature in enumerate(feature_sets):

            for col, t in enumerate(["OvR", "OvO"]):

                ax = axes[row, col]

                df_plot = df[
                    (df["Features (k)"] == feature) &
                    (df["Type"] == t)
                ].copy()

                df_melt = df_plot.melt(
                    id_vars=["Class", "Label"],
                    value_vars=metrics,
                    var_name="Metric",
                    value_name="Score"
                )

                sns.barplot(
                    data=df_melt,
                    x="Class",
                    y="Score",
                    hue="Metric",
                    edgecolor="black",
                    errorbar="sd",
                    ax=ax
                )

                ax.set_title(
                    f"{t} - {feature} Features",
                    fontsize=14
                )

                ax.set_xlabel("Class")
                ax.set_ylabel("AUC")
                ax.set_ylim(0, 1)
                ax.tick_params(axis="x", rotation=30)

                ax.legend(
                    title="Metric",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left"
                )

                # Value labels
                for container in ax.containers:
                    for bar in container:

                        height = bar.get_height()

                        if not np.isnan(height):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                height / 2,
                                f"{height:.2f}",
                                ha="center",
                                va="center",
                                fontsize=9
                            )

    fig.suptitle(
        "Model Performance Across Feature Sets",
        fontsize=18
    )

    fig.tight_layout()

    fig.savefig(
        plot_per_feature,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    
    
def main():
    parser = argparse.ArgumentParser(description="Run Classification Model")
    parser.add_argument("--X", action="append", required=True, help="Path to an omics feature matrix. Can be provided multiple times.")
    parser.add_argument("--omics_type", action="append", required=True, help="Name of the corresponding omics layer.")
    parser.add_argument("--combine_omics", action="store_true", help="Combine all provided omics feature matrices.")    
    parser.add_argument("--y", type=str, required=True, help="path to y")
    parser.add_argument("--target_column", type=str, required=True, help="target column in the target file y.")
    parser.add_argument("--ks", type=str, required=True, help="list of values of k")
    parser.add_argument("--n_seeds", type=int, default=2, help="number of seeds")
    parser.add_argument("--model", type=str, required=True, help="choose model :['rf', 'XGB', 'ETC', 'lgbm', 'TabPFN']")
    parser.add_argument("--sampling_strategy", type=str, required=True, help="choose sampling strategy: ['No Sampling','Random OverSampling','SMOTE','Random UnderSampling','NearMiss (v1)','NearMiss (v2)','NearMiss (v3)']")
    parser.add_argument("--grid_search", action="store_true", help="grid search")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--diagnostic_plot", type=str, required=True)
    parser.add_argument("--plot_per_feature", type=str, required=True)
    args = parser.parse_args()
    
    # reading str file paths
    omics_data = []

    for X_file, assay in zip(args.X, args.omics_type):
        X = pd.read_csv(X_file, sep="\t", index_col=0)   
        # keep only numeric columns for sampling strategies
        X = X.select_dtypes(include=[np.number])
        omics_data.append((assay, X))
        
    #X = pd.read_csv(args.X, sep="\t")
    # reading target file 
    y = pd.read_csv(args.y, sep="\t", index_col=0 )
 
    # target column from the target file
    y = y[args.target_column]
    
     # ks str value to int list 
    ks = [int(x.strip()) for x in args.ks.split(",")]
    
    
  
    # CASE 1: Multiple omics inputs
    if len(omics_data) > 1:

        # Multiple inputs + combine
        if args.combine_omics:

            # making sure only the common sample information is concatenated
            X = pd.concat(
                [X for assay, X in omics_data],
                axis=1,
                join="inner"
            )

            # keeping only the sample present in X
            y = y.loc[X.index]

            # flattening y into 1D array
            y = y.values.ravel()

            # Label = combined
            label = "combined"

            seed_results = run_classification(
                X,
                y,
                ks,
                args.n_seeds,
                args.output,
                label,
                args.model,
                args.sampling_strategy,
                args.grid_search
            )

            plot_pairwise_diagnostics(
                seed_results,
                args.diagnostic_plot
            )


        # Multiple inputs + do NOT combine
        else:

            all_seed_results = {}

            for assay, X in omics_data:

                common_samples = X.index.intersection(y.index)

                X_aligned = X.loc[common_samples]
                y_aligned = y.loc[common_samples]

                # Label = assay name
                label = assay

                # flattening y into 1D array
                y_aligned = y_aligned.values.ravel()

                seed_results = run_classification(
                    X_aligned,
                    y_aligned,
                    ks,
                    args.n_seeds,
                    args.output,
                    label,
                    args.model,
                    args.sampling_strategy,
                    args.grid_search
                )

                all_seed_results[assay] = seed_results

            plot_pairwise_diagnostics_multiple(
                all_seed_results,
                args.diagnostic_plot
            )


    # CASE 2: Single omics input
    else:

        assay, X = omics_data[0]

        common_samples = X.index.intersection(y.index)

        X_aligned = X.loc[common_samples]
        y_aligned = y.loc[common_samples]

        # Label = assay name
        label = assay

        # flattening y into 1D array
        y_aligned = y_aligned.values.ravel()

        seed_results = run_classification(
            X_aligned,
            y_aligned,
            ks,
            args.n_seeds,
            args.output,
            label,
            args.model,
            args.sampling_strategy,
            args.grid_search
        )

        plot_pairwise_diagnostics(
            seed_results,
            args.diagnostic_plot
        )


    # Plot model performance for all cases
    plot_model_performance_by_features(
        args.output,
        args.plot_per_feature
    )
    

        

if __name__ == "__main__":
    main()
