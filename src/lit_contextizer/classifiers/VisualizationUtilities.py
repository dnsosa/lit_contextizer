"""Collection of functions for visualizing results."""

# -*- coding: utf-8 -*-

import itertools
import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt

from matplotlib_venn import venn3

import networkx as nx

import numpy as np

import pandas as pd

from scipy.special import softmax

import seaborn as sns

from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils import resample

from .TrainTransformers import train_transformers


def generate_analysis_figs(in_df, data_id, tf_model_root_dir,
                           grouped_analysis=False,
                           downsample_maj=True,
                           upsample_min=False,
                           weight_imbalance=False,
                           plot_roc_curve=True,
                           plot_confusion_matrices=True,
                           plot_feature_analysis=True,
                           in_clf_list=None,
                           fit_transformers=True,
                           recall_only=False,
                           in_loc='lower right',
                           SEED=44,
                           out_dir=None,
                           filename="",
                           dpi=300):
    """
    Generate main analysis figures.

    :param in_df: Input DF
    :param data_id: ID for the dataset and condition for which the figures are being created
    :param tf_model_root_dir: root directory for saving TF models
    :param grouped_analysis: If True, contexts considered as a group (concept-level) not individual mentions
    :param downsample_maj: Downsample majority class?
    :param upsample_min: Upsample minority class?
    :param weight_imbalance: Incorporate weighting for class imbalance?
    :param plot_roc_curve: Plot ROC curve?
    :param plot_confusion_matrices: Plot confusion matrices?
    :param plot_feature_analysis: Plot feature importance anlayses?
    :param in_clf_list: Input list of pre-trained classifiers
    :param recall_only: Only plot recall results?
    :param in_loc: Location for legend
    :param SEED: random seed
    :param out_dir: Output directory
    :param filename: Output filename
    :param dpi: Image resolution
    """
    # Pre-process the data
    if not grouped_analysis:
        model_contra_df = in_df[
            ['rel', 'con_sent', 'con', 'sent_dist', 'sec_dist', 'norm_rel_sec', 'norm_con_sec',
             'num_con_mentions', 'con_mention_frac', 'con_mention_50', 'is_con_mention_max',
             'is_con_fp', 'is_closest_cont_by_sent', 'con_in_mesh_headings', 'annotation']].drop_duplicates().dropna()
    else:
        in_df_grp = in_df.groupby(['rel', 'con'])
        in_df_grp_features = in_df.assign(min_sent_dist=in_df_grp['sent_dist'].transform(min),
                                          min_sec_dist=in_df_grp['sec_dist'].transform(min),
                                          num_con_mentions=in_df_grp['num_con_mentions'].transform(
                                              lambda x: x.drop_duplicates().sum()),  # Here in case of plurals
                                          is_closest_cont_by_sent=in_df_grp['is_closest_cont_by_sent'].transform(max),
                                          # Here in case of plurals issue
                                          any_con_fp=in_df_grp['is_con_fp'].transform(max),
                                          any_con_title=in_df_grp['norm_con_sec'].transform(
                                              lambda x: x.eq('title').any()),
                                          any_con_abstract=in_df_grp['norm_con_sec'].transform(
                                              lambda x: x.eq('abstract').any()),
                                          any_con_background=in_df_grp['norm_con_sec'].transform(
                                              lambda x: x.eq('background').any()),
                                          any_con_methods=in_df_grp['norm_con_sec'].transform(
                                              lambda x: x.eq('methods').any()),
                                          any_con_results=in_df_grp['norm_con_sec'].transform(
                                              lambda x: x.eq('results').any()),
                                          any_con_disc_conc=in_df_grp['norm_con_sec'].transform(
                                              lambda x: x.eq('disicussion and conclusion').any()),
                                          )

        in_df_grp_features = in_df_grp_features[
            ['rel', 'con', 'min_sent_dist', 'min_sec_dist',
             'num_con_mentions', 'con_mention_frac', 'con_mention_50', 'is_con_mention_max',
             'any_con_fp', 'is_closest_cont_by_sent',
             'con_in_mesh_headings', 'norm_rel_sec', 'any_con_title', 'any_con_abstract', 'any_con_background',
             'any_con_methods', 'any_con_results', 'any_con_disc_conc', 'annotation']].drop_duplicates().dropna()

        # Need to recalculate the num_mentions_frac features
        in_df_grp_features_mentions_grp = in_df_grp_features.groupby(['rel'])['num_con_mentions']
        t1, t2 = in_df_grp_features['num_con_mentions'], in_df_grp_features_mentions_grp.transform('sum')
        in_df_grp_features['con_mention_frac'] = t1 / t2
        in_df_grp_features = in_df_grp_features. \
            assign(num_con_mentions_max=in_df_grp_features_mentions_grp.transform(max))
        t1, t2 = in_df_grp_features['num_con_mentions'], in_df_grp_features['num_con_mentions_max']
        in_df_grp_features["is_con_mention_max"] = (t1 == t2)
        in_df_grp_features['con_mention_50'] = in_df_grp_features['con_mention_frac'] >= 0.5

        model_contra_df = in_df_grp_features[
            ['rel', 'con', 'min_sent_dist', 'min_sec_dist',
             'num_con_mentions', 'con_mention_frac', 'con_mention_50', 'is_con_mention_max',
             'any_con_fp', 'is_closest_cont_by_sent',
             'con_in_mesh_headings', 'norm_rel_sec', 'any_con_title', 'any_con_abstract', 'any_con_background',
             'any_con_methods', 'any_con_results', 'any_con_disc_conc', 'annotation']].drop_duplicates().dropna()

    # Find positives and negatives in case need to up/down sample a class
    mc_df_neg = model_contra_df[~model_contra_df.annotation]
    mc_df_pos = model_contra_df[model_contra_df.annotation]
    print(f"N negatives: {len(model_contra_df[~model_contra_df.annotation].drop_duplicates())}")
    print(f"N positives: {len(model_contra_df[model_contra_df.annotation].drop_duplicates())}")
    print(f"Total len of resulting DF: {len(model_contra_df)}")
    print("\n")

    if len(mc_df_neg) > len(mc_df_pos):
        mc_df_maj, mc_df_min = mc_df_neg, mc_df_pos
    else:
        mc_df_maj, mc_df_min = mc_df_pos, mc_df_neg

    if upsample_min:
        mc_df_min_up = resample(mc_df_min, replace=True, n_samples=len(mc_df_maj), random_state=SEED)
        model_contra_df = pd.concat([mc_df_maj, mc_df_min_up])

    elif downsample_maj:
        mc_df_maj_down = resample(mc_df_maj, replace=False, n_samples=len(mc_df_min), random_state=SEED)
        model_contra_df = pd.concat([mc_df_maj_down, mc_df_min])

    model_contra_df.con_in_mesh_headings = model_contra_df.con_in_mesh_headings.astype(int)
    n_negs = len(model_contra_df[~model_contra_df.annotation].drop_duplicates())
    print(f"N negatives - after up/down-weighting: {n_negs}")
    n_pos = len(model_contra_df[model_contra_df.annotation].drop_duplicates())
    print(f"N positives - after up/down-weighting: {n_pos}")
    print(f"Total len of resulting DF: {len(model_contra_df)}")

    if not grouped_analysis:
        df = pd.get_dummies(model_contra_df[['sent_dist', 'sec_dist', 'norm_rel_sec', 'norm_con_sec',
                                             'num_con_mentions', 'con_mention_frac', 'con_mention_50',
                                             'is_con_mention_max', 'is_con_fp', 'is_closest_cont_by_sent',
                                             'con_in_mesh_headings', 'annotation']])
    else:
        df = pd.get_dummies(model_contra_df[['min_sent_dist', 'min_sec_dist',
                                             'num_con_mentions', 'con_mention_frac', 'con_mention_50',
                                             'is_con_mention_max', 'any_con_fp',
                                             'is_closest_cont_by_sent', 'con_in_mesh_headings', 'norm_rel_sec',
                                             'any_con_title', 'any_con_abstract', 'any_con_background',
                                             'any_con_methods', 'any_con_results', 'any_con_disc_conc', 'annotation']])

        # Check that all the columns are present since the classifiers will expect the full number of columns
        r_section_names = ["title", "abstract", "background", "methods", "results", "discussion and conclusion"]
        c_section_names = ["title", "abstract", "background", "methods", "results", "disc_conc"]
        for _, (r_sec, c_sec) in enumerate(zip(r_section_names, c_section_names)):
            rel_col_name = f"norm_rel_sec_{r_sec}"
            if rel_col_name not in df.columns:
                print(f"{rel_col_name} not found in column list. Assigning it a column of falses")
                df[rel_col_name] = False
            con_col_name = f"any_con_{c_sec}"
            if con_col_name not in df.columns:
                print(f"{con_col_name} not found in column list. Assigning it a column of falses")
                df[con_col_name] = 0

    if in_clf_list is None:
        X_train, X_test, y_train, y_test = train_test_split(df.drop(["annotation"], axis=1), df["annotation"],
                                                            test_size=1.0 / 3, random_state=SEED)
    else:
        X_test = df.drop(["annotation"], axis=1).values
        y_test = df["annotation"].values

    # Train the classifiers
    if in_clf_list is None:
        print("Initializing list of untrained models...")
        if not weight_imbalance:
            clf_list = [LogisticRegression(penalty='l2', random_state=44, solver='liblinear'),
                        # LinearSVC(random_state=44),
                        SVC(kernel='linear', random_state=44, probability=True),
                        # SVC(kernel='rbf', random_state=44, probability=True),
                        RandomForestClassifier(random_state=44),
                        MLPClassifier(random_state=44),
                        GradientBoostingClassifier(random_state=44)]
        else:
            clf_list = [LogisticRegression(penalty='l2', random_state=44, solver='liblinear', class_weight='balanced'),
                        # LinearSVC(random_state=44, class_weight='balanced'),
                        SVC(kernel='linear', random_state=44, probability=True, class_weight='balanced'),
                        # SVC(kernel='rbf', random_state=44, probability=True, class_weight='balanced'),
                        RandomForestClassifier(random_state=44, class_weight='balanced'),
                        MLPClassifier(random_state=44),
                        GradientBoostingClassifier(random_state=44)]
    else:
        print(f"List of {len(in_clf_list)} pre-trained models provided. No need to initialize new models.")
        clf_list = in_clf_list

    clf_label_map = {"LogisticRegression": "Logistic Reg",
                     "SVC_linear": "SVC - Linear",
                     "SVC_rbf": "SVC - Gaussian",
                     "RandomForestClassifier": "Random Forest",
                     "MLPClassifier": "Feedforward Neural Net",
                     "GradientBoostingClassifier": "Gradient Boosted Trees",
                     "biobert": "BioBERT",
                     "pubmedbert": "PubMedBERT",
                     "roberta": "RoBERTa"}

    colors = ["#E69F00", "#67C8FF", "#00A77F", "#F0E442", "#0072B2", "#CE4646"]
    clf_color_map = dict(zip(clf_label_map.values(), colors))

    tf_color_map = {"biobert": "#21B930", "pubmedbert": "#A230A7", "roberta": "#DAB973"}

    def get_clf_name(clf, mapper=clf_label_map):
        raw_clf_name = str(clf).split('(')[0]
        if raw_clf_name == "SVC":
            raw_clf_name += f"_{clf.kernel}"
        clf_name = mapper[raw_clf_name]
        return clf_name

    if in_clf_list is None:
        for clf in clf_list:
            print(f"Fitting model: {get_clf_name(clf)}")
            clf.fit(X_train, y_train)

    if fit_transformers:
        trainer_list, test_dataset_list, tf_names = train_transformers(model_contra_df, data_id, tf_model_root_dir,
                                                                       test_frac=1/3 if in_clf_list is None else 1,
                                                                       truncation=True, epochs=1, batch_size=2,
                                                                       learning_rate=1e-6,
                                                                       SEED=42)

    # ROC Curve
    if plot_roc_curve:
        _ = plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 16})

        precs = []
        recs = []
        fs = []
        y_pred_list = []
        clf_names = []
        for clf in clf_list:
            clf_name = get_clf_name(clf)
            clf_names.append(clf_name)
            print(f"Looking at clf: {clf_name}")
            y_pred = clf.predict(X_test)
            y_pred_list.append(y_pred)
            print("Accuracy", metrics.accuracy_score(y_test, y_pred))
            prec, rec, f, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='binary')
            print("PRINTING METRICS!")
            print(metrics.precision_recall_fscore_support(y_test, y_pred, average='binary'))
            precs.append(prec)
            recs.append(rec)
            fs.append(f)
            y_pred_proba = clf.predict_proba(X_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

            # Only calculate AUC if both classes represented
            if len(set(y_test)) >= 2:
                auc = metrics.roc_auc_score(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f"{get_clf_name(clf)}, AUC={round(auc, 3)}",
                         color=clf_color_map[get_clf_name(clf)])
                plt.legend(loc=4, prop={'size': 16})

        if fit_transformers:
            for tf_idx, trainer in enumerate(trainer_list):
                raw_y_pred, _, _ = trainer.predict(test_dataset_list[tf_idx])
                y_pred = np.argmax(raw_y_pred, axis=1)
                y_pred_list.append(y_pred)
                y_test = test_dataset_list[tf_idx]['labels']
                print("Accuracy", metrics.accuracy_score(y_test, y_pred))
                prec, rec, f, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='binary')
                print("PRINTING METRICS!")
                print(metrics.precision_recall_fscore_support(y_test, y_pred, average='binary'))
                precs.append(prec)
                recs.append(rec)
                fs.append(f)
                y_pred_proba = softmax(raw_y_pred, axis=1)[:, 1]
                fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

                if len(set(y_test)) >= 2:
                    auc = metrics.roc_auc_score(y_test, y_pred_proba)
                    plt.plot(fpr, tpr, label=f"{clf_label_map[tf_names[tf_idx]]}, AUC={auc:.3f}",
                             color=tf_color_map[tf_names[tf_idx]])
                    plt.legend(loc=4, prop={'size': 16})

        # Only plot if both classes represented
        if len(set(y_test)) >= 2:
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            plt.ylabel("TPR")
            plt.xlabel("FPR")

        if out_dir is not None:
            out_file = os.path.join(out_dir, f"{filename}_ROC.png")
            plt.savefig(out_file, dpi=dpi, bbox_inches="tight")

        # Make an output dataframe of the labels and the predictions
        annots_df = pd.DataFrame({"Label": y_test,
                                  f"Pred: {clf_names[0]}": y_pred_list[0],
                                  f"Pred: {clf_names[1]}": y_pred_list[1],
                                  f"Pred: {clf_names[2]}": y_pred_list[2],
                                  f"Pred: {clf_names[3]}": y_pred_list[3],
                                  f"Pred: {clf_names[4]}": y_pred_list[4],
                                  f"Pred: {tf_names[0]}": y_pred_list[5],
                                  f"Pred: {tf_names[1]}": y_pred_list[6]})
        if in_clf_list is not None:
                X_test = df.drop(["annotation"], axis=1)  # return back to a dataframe
        X_test_rel_con_df = pd.merge(model_contra_df[['rel', 'con']], X_test, left_index=True, right_index=True)
        predictions_df = pd.merge(X_test_rel_con_df, annots_df, left_index=True, right_index=True)

        plt.show()

        # Overall metric values
        # _ = plt.figure(figsize=(12, 10))
        # width = 0.2 if not recall_only else 0.7
        _ = plt.figure(figsize=(20, 10))
        width = .3
        print("Here are the fs")
        print(fs)
        print("Here are the recs")
        print(recs)
        v_space = 0.03
        x = np.arange(len(recs))
        # Recall
        plt.bar(x, recs, width, color='green', label='Recall')
        for i, v in enumerate(recs):
            fontsize = 20 if recall_only else 14
            t2 = plt.text(i, v + v_space, f"{round(v, 3):.3f}", color="black", ha='center', fontsize=fontsize)
            t2.set_bbox({'facecolor': 'white',
                         'alpha': 1,
                         'boxstyle': 'square,pad=0',
                         'edgecolor': 'white'})
        if not recall_only:
            # Precision
            plt.bar(x - width, precs, width, color='orange', label='Precision')
            for i, v in enumerate(precs):
                t1 = plt.text(i - width * .65, v + v_space, f"{round(v, 3):.3f}", color="black", ha='right',
                              fontsize=fontsize)
                t1.set_bbox({'facecolor': 'white',
                             'alpha': 1,
                             'boxstyle': 'square,pad=0',
                             'edgecolor': 'white'})
            # F1
            plt.bar(x + width, fs, width, color='blue', label='F1')
            print("Enumerating fs")
            print(enumerate(fs))
            for i, v in enumerate(fs):
                t3 = plt.text(i + width * .65, v + v_space, f"{round(v, 3):.3f}", color="black", ha='left',
                              fontsize=fontsize)
                t3.set_bbox({'facecolor': 'white',
                             'alpha': 1,
                             'boxstyle': 'square,pad=0',
                             'edgecolor': 'white'})
        clf_labels = [get_clf_name(clf) for clf in clf_list]
        if fit_transformers:
            clf_labels += [clf_label_map[tf] for tf in tf_names]
        plt.xticks(x, clf_labels, rotation=45, ha='right')
        if recall_only:
            plt.legend(["Recall"], loc=in_loc, framealpha=1)
        else:
            # Reorder the legend labels so recall is in the middle
            plt.legend(*([x[i] for i in [1, 0, 2]] for x in plt.gca().get_legend_handles_labels()),
                       loc=in_loc, framealpha=1)
        plt.ylim([0, 1.1])
        plt.ylabel("Metric")

        if out_dir is not None:
            out_file = os.path.join(out_dir, f"{filename}_Metrics.png")
            plt.savefig(out_file, dpi=dpi, bbox_inches="tight")

        plt.show()
    else:
        predictions_df = None

    # Confusion Matrices
    # TODO: Clean up formatting, remove color bar
    if plot_confusion_matrices:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 15))

        for _, (clf, ax) in enumerate(zip(clf_list, axes.flatten())):
            ConfusionMatrixDisplay.from_estimator(clf,
                                                  X_test,
                                                  y_test,
                                                  ax=ax,
                                                  display_labels=["Negative", "Positive"],
                                                  normalize='all',
                                                  cmap=plt.cm.Blues)
            ax.title.set_text(get_clf_name(clf))
        plt.tight_layout()

        if out_dir is not None:
            out_file = os.path.join(out_dir, f"{filename}_ConfusionMatrices.png")
            plt.savefig(out_file, dpi=dpi, bbox_inches="tight")

        plt.show()

    # Feature Analysis
    if plot_feature_analysis:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 14), sharey=True)

        if not grouped_analysis:
            col_mapper = {"sent_dist": "Sent Dist",
                          "sec_dist": "Sec Dist",
                          "num_con_mentions": "Num. Con Mentions",
                          'con_mention_frac': "Con Mention Fraction",
                          'con_mention_50': "Con Frac > 50%",
                          'is_con_mention_max': "Con is Most Frequent",
                          "is_con_fp": "Con is 1P",
                          "is_closest_cont_by_sent": "Con closest to Rel",
                          "con_in_mesh_headings": "Con in MeSH",
                          "norm_rel_sec_title": "Rel Sec: Title",
                          "norm_rel_sec_abstract": "Rel Sec: Abstract",
                          "norm_rel_sec_background": "Rel Sec: Bkgd",
                          "norm_rel_sec_methods": "Rel Sec: Methods",
                          "norm_rel_sec_results": "Rel Sec: Results",
                          "norm_rel_sec_discussion and conclusion": "Rel Sec: Disc/Conc",
                          "norm_con_sec_title": "Con Sec: Title",
                          "norm_con_sec_abstract": "Con Sec: Abstract",
                          "norm_con_sec_background": "Con Sec: Bkgd",
                          "norm_con_sec_methods": "Con Sec: Methods",
                          "norm_con_sec_results": "Con Sec: Results",
                          "norm_con_sec_discussion and conclusion": "Con Sec: Disc/Conc"}
        else:
            col_mapper = {"min_sent_dist": "Min Sent Dist",
                          "min_sec_dist": "Min Sec Dist",
                          "num_con_mentions": "Num. Con Mentions",
                          'con_mention_frac': "Con Mention Fraction",
                          'con_mention_50': "Con Frac > 50%",
                          'is_con_mention_max': "Con is Most Frequent",
                          "any_con_fp": "Any Con in 1P",
                          "is_closest_cont_by_sent": "Con closest to Rel",
                          "con_in_mesh_headings": "Con in MeSH",
                          "norm_rel_sec_title": "Rel Sec: Title",
                          "norm_rel_sec_abstract": "Rel Sec: Abstract",
                          "norm_rel_sec_background": "Rel Sec: Bkgd",
                          "norm_rel_sec_methods": "Rel Sec: Methods",
                          "norm_rel_sec_results": "Rel Sec: Results",
                          "norm_rel_sec_discussion and conclusion": "Rel Sec: Disc/Conc",
                          "any_con_title": "Any Con in Title",
                          "any_con_abstract": "Any Con in Abstract",
                          "any_con_background": "Any Con in Bkgd",
                          "any_con_methods": "Any Con in Methods",
                          "any_con_results": "Any Con in Results",
                          "any_con_disc_conc": "Any Con in Disc/Conc"}

        # Log Reg
        log_reg = [clf for clf in clf_list if get_clf_name(clf) == clf_label_map["LogisticRegression"]][0]
        importance = log_reg.coef_.flatten()

        axes[0].barh(np.arange(len(importance)), importance, color="blue")
        axes[0].set_yticks(np.arange(len(importance)))
        axes[0].set_yticklabels(list(map(col_mapper.get, X_train.columns)), rotation=0)
        for i, v in enumerate(importance):
            if v > 0:
                axes[0].text(v + .025, i - .16, f"{round(v, 3):.3f}", ha="left", color="black", fontsize=20)
            else:
                axes[0].text(v - .025, i - .16, f"{round(v, 3):.3f}", ha="right", color="black", fontsize=20)

        axes[0].set_xlim(-max(abs(importance)) * 1.5, max(abs(importance)) * 1.5)
        axes[0].set_xlabel("Log. Reg. Coeff, $\\beta_i$", fontsize=26)
        axes[0].set_title("Logistic Regression\nFeat. Importance")

        # Random Forest
        rand_forest = [clf for clf in clf_list if get_clf_name(clf) == clf_label_map["RandomForestClassifier"]][0]
        importance = rand_forest.feature_importances_

        axes[1].barh(np.arange(len(importance)), importance, color="blue")
        for i, v in enumerate(importance):
            axes[1].text(abs(v) + .01, i - .16, f"{round(v, 3):.3f}", color="black", fontsize=20)
        axes[1].set_xlim(0, max(importance) * 1.25)
        axes[1].set_xlabel("Gini Importance Score", fontsize=26)
        axes[1].set_title("Random Forest\nFeat. Importance")

        # Permutation
        gbc = [clf for clf in clf_list if get_clf_name(clf) == clf_label_map["GradientBoostingClassifier"]][0]
        results = permutation_importance(gbc, X_train.astype(np.float32), y_train.astype(np.float32),
                                         scoring='neg_mean_squared_error', random_state=44)
        importance = results.importances_mean

        axes[2].barh(np.arange(len(importance)), importance, color="blue")
        for i, v in enumerate(importance):
            plt.text(abs(v) + .002, i - .16, f"{round(v, 3):.3f}", color="black", fontsize=20)
        axes[2].set_xlim(0, max(importance) * 1.25)
        axes[2].set_xlabel("Importance Score", fontsize=26)
        axes[2].set_title("Permutation Feat. Importance\n(Gradient Boosted Trees)", fontsize=30)

        plt.tight_layout()

        if out_dir is not None:
            out_file = os.path.join(out_dir, f"{filename}_FeatureImportances.png")
            plt.savefig(out_file, dpi=dpi, bbox_inches="tight")

        plt.show()

    return model_contra_df, clf_list, predictions_df


def draw_section_distribution(in_df,
                              in_mesh=True,
                              alpha=1,
                              out_dir=None,
                              filename="",
                              dpi=300):
    """
    Look at the distribution of sections that the papers are in.

    :param in_df: input dataframe with features extracted
    :param in_mesh: if True, show if the terms are in MeSH headings
    :param alpha: transparency
    :param out_dir: output directory
    :
    """
    fig, axs = plt.subplots(1, 2, sharey=False, figsize=(17, 8))
    fig.suptitle("Section Locations for Extracted Contexts and Relations", y=1.1)

    my_order = ["title", "abstract", "background", "methods", "results", "discussion and conclusion", "Other"]

    norm_rel_sec = list(in_df.groupby(['rel']).sample(n=1)["norm_rel_sec"])
    norm_rel_sec = ['Other' if v is None else v for v in norm_rel_sec]

    ct = Counter(norm_rel_sec)
    labels = [sec.capitalize() for sec in my_order]
    values = [ct[k] for k in my_order]

    axs[0].bar(np.arange(len(labels)), values, width=1, color="green", edgecolor='black', linewidth=1.5)
    axs[0].set_title("Relation")
    axs[0].set_xticks(np.arange(len(labels)) + 0)
    axs[0].set_xticklabels(labels, fontsize=16, rotation=45, ha='right')
    axs[0].set_ylabel("Count")

    norm_con_sec = list(in_df.groupby(['con_sent']).sample(n=1)["norm_con_sec"])
    norm_con_sec = ['Other' if v is None else v for v in norm_con_sec]
    ct = Counter(norm_con_sec)
    values = [ct[k] for k in my_order]

    con_mesh = list(in_df[in_df.con_in_mesh_headings].groupby(['con_sent']).sample(n=1)["norm_con_sec"])
    con_mesh = ['Other' if v is None else v for v in con_mesh]
    ct_mesh = Counter(con_mesh)
    labels_mesh = my_order
    values_mesh = [ct_mesh[k] for k in my_order]

    axs[1].bar(np.arange(len(labels)), values, width=1, color="green", edgecolor='black', linewidth=1.5, alpha=alpha)
    if in_mesh:
        axs[1].bar(np.arange(len(labels_mesh)), values_mesh, width=1, color="gold", edgecolor='white', linewidth=2.5,
                   label="In MeSH Headings")
        axs[1].legend()
    axs[1].set_title("Context")
    axs[1].set_xticks(np.arange(len(labels)) + 0)
    axs[1].set_xticklabels(labels, fontsize=16, rotation=45, ha='right')

    if out_dir is not None:
        out_file = os.path.join(out_dir, f"{filename}_SectionDistributions.png")
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight")

    plt.show()


def draw_CTs_in_mesh_counts(in_df,
                            out_dir=None,
                            filename="",
                            dpi=300):
    """
    Draw the counts of CTs in MeSH headings.

    :params in_df: input dataframe with features extracted
    """
    ct_list = list(set(in_df.con))
    in_out_mesh_headings_counts = in_df[["paper_id", "con", "con_in_mesh_headings"]].drop_duplicates().groupby(
        "con").con_in_mesh_headings.value_counts()

    # Must revisit this if expanding the lexicon of CTs being used, or incorporating more papers, etc.
    immune = {"basophil", "classical monocyte", "common myeloid progenitor", "DN1 thymic pro-T cell", "DN3 thymocyte",
              "DN4 thymocyte", "erythrocyte", "erythroid lineage cell", "erythroid progenitor cell", "granulocyte",
              "hematopoietic stem cell", "immature natural killer cell", "innate lymphoid cell",
              "intermediate monocyte", "Langerhans cell", "leukocyte", "liver dendritic cell", "macrophage",
              "mast cell", "mature conventional dendritic cell", "mature NK T cell", "memory B cell",
              "mesenchymal stem cell", "microglial cell", "myeloid dendritic cell", "naive B cell",
              "naive regulatory T cell", "neutrophil", "non-classical monocyte", "plasma cell", "plasmablast",
              "plasmacytoid dendritic cell", "platelet", "regulatory T cell", "T follicular helper cell", "thymocyte"}
    epithelial = {"basal cell", "bladder urothelial cell", "ciliated epithelial cell", "club cell",
                  "conjunctival epithelial cell", "corneal epithelial cell", "duct epithelial cell",
                  "duodenum glandular cell", "enterocyte", "epithelial cell of uterus", "eye photoreceptor cell",
                  "goblet cell", "hepatocyte", "intestinal crypt stem cell", "intestinal enteroendocrine cell",
                  "intestinal tuft cell", "intrahepatic cholangiocyte", "ionocyte", "keratinocyte", "keratocyte",
                  "kidney epithelial cell", "large intestine goblet cell", "lung ciliated cell",
                  "medullary thymic epithelial cell", "mesothelial cell", "mucus secreting cell", "myoepithelial cell",
                  "pancreatic A cell", "pancreatic acinar cell", "pancreatic D cell", "pancreatic ductal cell",
                  "pancreatic PP cell", "paneth cell of colon", "pigmented ciliary epithelial cell",
                  "pulmonary ionocyte", "respiratory goblet cell", "retinal pigment epithelial cell",
                  "salivary gland cell", "secretory cell", "small intestine goblet cell", "surface ectodermal cell",
                  "tracheal goblet cell", "type B pancreatic cell", "type I pneumocyte", "type II pneumocyte"}
    stromal = {"adventitial cell", "bronchial smooth muscle cell", "cardiac muscle cell", "skeletal muscle cell",
               "connective tissue cell", "fast muscle cell", "fat cell", "fibroblast of breast", "cardiac fibroblast",
               "melanocyte", "mesenchymal stem cell", "mesothelial cell", "Muller cell", "myofibroblast cell",
               "myometrial cell", "pancreatic stellate cell", "pericyte cell", "radial glial cell",
               "retina horizontal cell", "retinal bipolar neuron", "retinal ganglion cell", "Schwann cell",
               "slow muscle cell", "smooth muscle cell", "stromal cell", "tendon cell", "tongue muscle cell"}
    endothelial = {"blood vessel endothelial cell", "capillary endothelial cell", "cardiac endothelial cell",
                   "endothelial cell of artery", "gut endothelial cell", "lung microvascular endothelial cell",
                   "vein endothelial cell"}

    ct2category = {}
    for ct in ct_list:
        if ct in immune:
            ct2category[ct] = "Immune"
        elif ct in epithelial:
            ct2category[ct] = "Epithelial"
        elif ct in stromal:
            ct2category[ct] = "Stromal"
        elif ct in endothelial:
            ct2category[ct] = "Endothelial"

    # Counts of term IN MeSH
    x_values = np.array([in_out_mesh_headings_counts[ct].get(True, 0) for ct in ct_list])

    # Counts of term NOT in MeSH
    y_values = np.array([in_out_mesh_headings_counts[ct].get(False, 0) for ct in ct_list])
    max_val = max(max(x_values), max(y_values))

    # lower bound
    lb = -.1
    # pad affects distance between text and points
    pad = 1.32

    plt.figure(figsize=(12, 12))
    plt.rcParams.update({'font.size': 16})

    # Plot y = x and vertical and horizontal axes
    plt.plot([lb, max_val * pad], [lb, max_val * pad], color='gray', lw=1, linestyle='--')
    plt.plot([lb, max_val * pad], [0, 0], color='grey', lw=1, linestyle='-')
    plt.plot([0, 0], [lb, max_val * pad], color='grey', lw=1, linestyle='-')

    # Plot individual cell types text label
    coord_counts = {}

    # Find number of overlaps for each coordinate
    for i, _ in enumerate(ct_list):
        x, y = x_values[i], y_values[i]
        if (x, y) not in coord_counts:
            coord_counts[(x, y)] = 1
        else:
            coord_counts[(x, y)] += 1

    for i, ct in enumerate(ct_list):
        x, y = x_values[i], y_values[i]
        if coord_counts[(x, y)] != 1:
            continue

        # Needed to manually adjust the label
        if ct in ["plasma cell", "mast cell"]:
            plt.annotate(ct, (max(x_values[i] * 1.05, 0.1), y_values[i] * 1.1), ha="right", alpha=.75, zorder=3,
                         fontsize=12)
        else:
            plt.annotate(ct, (max(x_values[i] * 1.05, 0.1), y_values[i] * 1.1), alpha=.75, zorder=3, fontsize=12)

    # Multiple fall at the same coordinate
    for (x, y) in coord_counts:
        if coord_counts[(x, y)] > 1:
            plt.annotate(f"Various - {coord_counts[(x, y)]} CTs", (max(x * 1.05, 0.1), y * 1.03), alpha=.75, zorder=3,
                         fontsize=12)

    category2marker = {"Immune": "1", "Epithelial": "v", "Stromal": "o", "Endothelial": "P"}

    # Plot individual CT points. Separated this out so I could choose different shapes
    for cat in category2marker.keys():
        ct_cat_list = [ct for ct in ct_list if ct2category[ct] == cat]
        x_cat_values = np.array([in_out_mesh_headings_counts[ct].get(True, 0) for ct in ct_cat_list])
        y_cat_values = np.array([in_out_mesh_headings_counts[ct].get(False, 0) for ct in ct_cat_list])
        if cat == "Immune":  # the point looked tiny otherwise
            size = 200
        else:
            size = 50
        plt.scatter(x_cat_values, y_cat_values, zorder=2, marker=category2marker[cat], label=cat, s=size)

    plt.legend(title="Tabula Sapiens CT Category\n", loc='best')

    plt.xlim([lb, max_val * pad])
    plt.ylim([lb, max_val * pad])
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.xlabel("Term in MeSH (# Papers)")
    plt.ylabel("Term NOT in MeSH (# Papers)")

    # Remove the left and bottom lines
    plt.box(on=False)

    if out_dir is not None:
        out_file = os.path.join(out_dir, f"{filename}_CTMeSHCounts.png")
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_benchmark_fig(df,
                       best_model_name=None,
                       best_model_stats=None,
                       recall_only=False,
                       out_dir=None,
                       filename="",
                       dpi=300):
    """
    Plot figure comparing against benchmarks.

    :param df: input DF
    :param best_model_name: Name of best model
    :param best_model_stats: Statistics from the best model (to be plotted)
    :param recall_only: Only show recall statistics?
    :param out_dir: Output directory
    :param filename: Output filename
    :param dpi: Image resolution
    """
    # Get a rel_con section match ?
    def any_rel_con_section_match(row):
        rel_sec = row["norm_rel_sec"]
        r_section_names = ["title", "abstract", "background", "methods", "results", "discussion and conclusion"]
        c_section_names = ["title", "abstract", "background", "methods", "results", "disc_conc"]
        sec_map = dict(zip(r_section_names, c_section_names))
        return row[f"any_con_{sec_map[rel_sec]}"]  # If True, there's a context in this section

    df['any_rel_con_section_match'] = df.apply(any_rel_con_section_match, axis=1)
    df['con_mention_50'] = df['con_mention_frac'] >= 0.5

    # Get if the minimum sentence distance is <= k
    for k in range(1, 7):
        df[f"min_sent_dist_{k}"] = df["min_sent_dist"] <= k

    benchmark_mapper = {"any_rel_con_section_match": "Any Con-Rel Section Match",
                        "is_con_mention_max": "Is Max Con Mentioned",
                        "con_mention_50": "Con Mention Frac $>$ 50%",
                        "min_sent_dist_1": "Min sent $d \leq 1$",
                        "min_sent_dist_2": "Min sent $d \leq 2$",
                        "min_sent_dist_3": "Min sent $d \leq 3$",
                        "min_sent_dist_4": "Min sent $d \leq 4$",
                        "min_sent_dist_5": "Min sent $d \leq 5$",
                        "min_sent_dist_6": "Min sent $d \leq 6$",
                        "con_in_mesh_headings": "In MeSH"}

    benchmark_order = ["any_rel_con_section_match",
                       "is_con_mention_max",
                       "con_mention_50",
                       "con_in_mesh_headings",
                       "min_sent_dist_1",
                       "min_sent_dist_2",
                       "min_sent_dist_3",
                       "min_sent_dist_4",
                       "min_sent_dist_5",
                       "min_sent_dist_6"]

    precs = []
    recs = []
    fs = []
    for _, benchmark in enumerate(benchmark_order):
        print(f"Looking at benchmark: {benchmark}")
        y_pred = df[benchmark]
        y_test = df["annotation"]
        print("Accuracy", metrics.accuracy_score(y_test, y_pred))
        prec, rec, f, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='binary')
        precs.append(prec)
        recs.append(rec)
        fs.append(f)

    if best_model_stats is not None:
        if recall_only:
            # only input value is a list of single recall value
            recs = [best_model_stats[0]] + recs
        else:
            precs = [best_model_stats[0]] + precs
            recs = [best_model_stats[1]] + recs
            fs = [best_model_stats[2]] + fs

    # Overall metric values
    _ = plt.figure(figsize=(32, 12))
    width = 0.25 if not recall_only else 0.7
    v_space = .015
    h_space = width*.6
    x = np.arange(len(recs))

    # Recall
    if best_model_stats is not None:
        plt.bar(x, recs, width, color='green', edgecolor='grey', hatch='/', alpha=.3, label='Recall')
    else:
        plt.bar(x, recs, width, color='green', label='Recall')
    for i, v in enumerate(recs):
        fontsize = 20 if recall_only else 13
        t2 = plt.text(i, v + v_space, f"{round(v, 3):.3f}", color="black", ha='center', fontsize=fontsize)
        t2.set_bbox({'facecolor': 'white',
                     'alpha': 1,
                     'boxstyle': 'square,pad=0',
                     'edgecolor': 'white'})

    if not recall_only:
        # Precision
        if best_model_stats is not None:
            plt.bar(x - width, precs, width, color='orange', edgecolor='grey', hatch='/', alpha=.3, label='Precision')
        else:
            plt.bar(x - width, precs, width, color='orange', label='Precision')
        for i, v in enumerate(precs):
            t1 = plt.text(i - h_space, v + v_space, f"{round(v, 3):.3f}", color="black", ha='right', fontsize=fontsize)
            t1.set_bbox({'facecolor': 'white',
                         'alpha': 1,
                         'boxstyle': 'square,pad=0',
                         'edgecolor': 'white'})

        # F1
        if best_model_stats is not None:
            plt.bar(x + width, fs, width, color='blue', edgecolor='grey', hatch='/', alpha=.3, label='F1')
        else:
            plt.bar(x + width, fs, width, color='blue', label='F1')
        for i, v in enumerate(fs):
            t3 = plt.text(i + h_space, v + v_space, f"{round(v, 3):.3f}", color="black", ha='left', fontsize=fontsize)
            t3.set_bbox({'facecolor': 'white',
                         'alpha': 1,
                         'boxstyle': 'square,pad=0',
                         'edgecolor': 'white'})

    benchmark_labels = [benchmark_mapper[benchmark] for benchmark in benchmark_order]
    if best_model_stats is not None:
        benchmark_labels = [best_model_name] + benchmark_labels

    if best_model_stats is not None:

        plt.bar(x[0], recs[0], width, color='green')
        if not recall_only:
            plt.bar(x[0] - width, precs[0], width, color='orange')
            plt.bar(x[0] + width, fs[0], width, color='blue')

    plt.xticks(x, benchmark_labels, rotation=20, ha='right')
    if recall_only:
        plt.legend(["Recall"], loc='lower left', framealpha=1)
    else:
        plt.legend(*([x[i] for i in [1, 0, 2]] for x in plt.gca().get_legend_handles_labels()),
                   loc='lower left', framealpha=1)
    plt.ylim([0, 1.05])
    plt.ylabel("Metric")

    if out_dir is not None:
        out_file = os.path.join(out_dir, f"{filename}_BenchmarkMetrics.png")
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_con_mention_frac_precision_plots(CT_df, tissue_df, combined_df, out_dir, dpi=300):
    """
    Plot context mention fraction precision plots.

    :param CT_df: cell type DF
    :param tissue_df: tissue DF
    :param combined_df: combined DF with both context types
    :param out_dir: output directory
    :param dpi: image output resolution
    """
    sns.set(style="white",
            rc={'figure.figsize': (15, 8)},
            font_scale=2)

    def precisionAtRecallP(df, P, digits=3):
        positive_vals = np.array(df[df.annotation].con_mention_frac)
        negative_vals = np.array(df[~df.annotation].con_mention_frac)
        threshold_for_recall_p = round(np.percentile(positive_vals, 100 - P), digits)
        print(f"Threshold for {P}% recall: {threshold_for_recall_p:.{digits}f}")

        tps = np.sum(positive_vals >= threshold_for_recall_p)
        fps = np.sum(negative_vals >= threshold_for_recall_p)

        precision_at_recall_p = round(tps / float(tps + fps), digits)
        print(f"Precision for {P}% recall: {precision_at_recall_p:.{digits}f}")

        return threshold_for_recall_p, precision_at_recall_p

    digits = 3
    alpha = .6
    neg_alpha = alpha

    CT_pos_color, CT_neg_color = '#BDC00E', '#EDEEB0'
    _ = sns.kdeplot(CT_df[CT_df.annotation].con_mention_frac, fill=True, linewidth=0, color=CT_pos_color,
                    label="Cell Type - Pos", alpha=alpha)
    _ = sns.kdeplot(CT_df[~CT_df.annotation].con_mention_frac, fill=True, linewidth=0, color=CT_neg_color,
                    label="Cell Type - Neg", alpha=neg_alpha)
    CTs_thresh, CTs_prec_at_90 = precisionAtRecallP(CT_df, 90, digits)
    plt.axvline(x=CTs_thresh, color=CT_pos_color)
    plt.text(CTs_thresh + 0.02, 9, f'Prec @ 90% Rec = {CTs_prec_at_90:.{digits}f}', color=CT_pos_color, ha='left',
             fontsize=16)

    tissues_pos_color, tissues_neg_color = "#11A7FF", "#C2DDEB"
    _ = sns.kdeplot(tissue_df[tissue_df.annotation].con_mention_frac, fill=True, linewidth=0, color=tissues_pos_color,
                    label="Tissue - Pos", alpha=alpha)
    _ = sns.kdeplot(tissue_df[~tissue_df.annotation].con_mention_frac, fill=True, linewidth=0,
                    color=tissues_neg_color, label="Tissue - Neg", alpha=neg_alpha)
    tissues_thresh, tissues_prec_at_90 = precisionAtRecallP(tissue_df, 90, digits)
    plt.axvline(x=tissues_thresh, color=tissues_pos_color)
    plt.text(tissues_thresh - 0.02, 9, f'Prec @ 90% Rec = {tissues_prec_at_90:.{digits}f}', color=tissues_pos_color,
             ha='right', fontsize=16)

    plt.xlabel("Context Mention Fraction")
    plt.ylim([0, 10])
    plt.legend(prop={'size': 16})

    out_file = os.path.join(out_dir, "MentionFracSeparated.png")
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.show()

    combined_pos_color, combined_neg_color = "#50CF17", "#CCF1BB"
    _ = sns.kdeplot(combined_df[combined_df.annotation].con_mention_frac, fill=True, linewidth=0,
                    color=combined_pos_color, label="Combined - Pos", alpha=alpha)
    _ = sns.kdeplot(combined_df[~combined_df.annotation].con_mention_frac, fill=True, linewidth=0,
                    color=combined_neg_color, label="Combined - Neg", alpha=alpha)
    combined_thresh, combined_prec_at_90 = precisionAtRecallP(combined_df, 90, digits)
    plt.axvline(x=combined_thresh, color=combined_pos_color)
    plt.text(combined_thresh + 0.02, 9, f'Prec @ 90% Rec = {combined_prec_at_90:.{digits}f}', color=combined_pos_color,
             ha='left', fontsize=16)

    plt.xlabel("Context Mention Fraction")
    plt.ylim([0, 10])
    plt.legend(prop={'size': 16})

    out_file = os.path.join(out_dir, "MentionFracCombined.png")
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_mac_hep_neu_venn(dengue_edgelist, fig_out_dir, dpi=300):
    """
    Plot venn diagram for 3 biggest cell types.

    :param dengue_edgelist: Edge list of dengue-relevant PPIs
    :param fig_out_dir: Output directory
    """
    plt.rcParams["figure.figsize"] = (20, 20)

    font = {'family': 'normal',
            'size': 32}

    matplotlib.rc('font', **font)

    mac_set = set(dengue_edgelist[dengue_edgelist.con == 'macrophage'][["entity1_entrez", "entity2_entrez"]].itertuples(
        index=False, name=None))
    hep_set = set(dengue_edgelist[dengue_edgelist.con == 'hepatocyte'][["entity1_entrez", "entity2_entrez"]].itertuples(
        index=False, name=None))
    neu_set = set(dengue_edgelist[dengue_edgelist.con == 'neutrophil'][["entity1_entrez", "entity2_entrez"]].itertuples(
        index=False, name=None))

    mac_only = mac_set.difference(hep_set.union(neu_set))
    hep_only = hep_set.difference(mac_set.union(neu_set))
    neu_only = neu_set.difference(mac_set.union(hep_set))

    mac_hep = mac_set.intersection(hep_set).difference(neu_set)
    mac_neu = mac_set.intersection(neu_set).difference(hep_set)
    neu_hep = neu_set.intersection(hep_set).difference(mac_set)

    all_three = mac_set.intersection(hep_set).intersection(neu_set)

    # Make the diagram
    venn3(subsets=(len(mac_only), len(hep_only), len(mac_hep),
                   len(neu_only), len(mac_neu), len(neu_hep),
                   len(all_three)),
          set_labels=('Macrophage', 'Hepatocyte', 'Neutrophil'))

    out_file = os.path.join(fig_out_dir, "Dengue3CTVenn.png")
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_dengue_networks(dengue_edgelist, fig_out_dir, min_deg=5, max_deg=15, dpi=300):
    """
    Plot dengue subnetworks.

    :param dengue_edgelist: DF of dengue PPIs
    :param fig_out_dir: output directory for fig
    :param min_deg: minimum degree for nodes included in subnetwork
    :param max_deg: maximum degree for nodes included in subnetwork
    :param dpi: image resolution
    """
    plt.rcParams["figure.figsize"] = (20, 20)

    G_dengue = nx.from_pandas_edgelist(dengue_edgelist, 'entity1_text', 'entity2_text', ['con'])
    mid_degree_nodes = [n for n, d in G_dengue.degree if min_deg <= d <= max_deg]

    G_dengue_sub = G_dengue.subgraph(list(mid_degree_nodes))
    Gcc = G_dengue_sub.subgraph(max(nx.connected_components(G_dengue_sub), key=len)).copy()
    Gcc.remove_edges_from(nx.selfloop_edges(Gcc))

    pos = nx.spring_layout(Gcc, seed=44)

    min_ct_count = 200
    color_palette = ["#D81B60", "#1E88E5", "#FFC107", "#004D40", "#90AB78", "#7B93B9"]
    common_cts = [CT for CT, count in Counter(dengue_edgelist.con).most_common() if count >= min_ct_count]
    ct_color_mapper = dict(zip(common_cts, color_palette))

    nx.draw_networkx_nodes(Gcc, pos, node_size=60, node_color='gray', alpha=1)
    nx.draw_networkx_edges(Gcc, pos, alpha=0.5)
    plt.axis('off')

    out_file = os.path.join(fig_out_dir, "DengueAllCTsSubnet.png")
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.show()

    for ct in ['macrophage', 'hepatocyte', 'neutrophil']:
        nx.draw_networkx_nodes(Gcc, pos, node_size=60, node_color='gray', alpha=0.25)

        edge_colors = []
        edge_widths = []

        for e in Gcc.edges(data=True):
            e_con = e[2]['con']
            if e_con == ct:
                edge_colors.append(ct_color_mapper.get(e_con, 'gray'))
                edge_widths.append(5)
            else:
                edge_colors.append('gray')
                edge_widths.append(1)
        nx.draw_networkx_edges(Gcc, pos, alpha=0.5, edge_color=edge_colors, width=edge_widths, label=ct)

        ct_edge_set = {(e[0], e[1]) for e in Gcc.edges(data=True) if e[2]['con'] == ct}
        ct_node_set = set(itertools.chain(*ct_edge_set))

        labels = {k: k for k in ct_node_set}
        nx.draw_networkx_labels(Gcc, pos, labels, font_size=25)

        plt.axis('off')

        out_file = os.path.join(fig_out_dir, f"Dengue_{ct}_subnet.png")
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.show()
