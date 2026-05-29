import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import joblib

OUTPUT_DIR = "outputs"
MODELS_TO_COMPARE = ["decision_tree", "random_forest", "knn", "svc", "xgboost"]


def load_test_data(output_dir):
    X_test = pd.read_csv(os.path.join(output_dir, "X_test_data.csv"))
    y_test = pd.read_csv(os.path.join(output_dir, "y_test_data.csv")).squeeze()
    return X_test, y_test


def load_model(output_dir, model_name):
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def get_predictions_for_all_models(models_dict, X_test, y_test):
    results = pd.DataFrame()
    results["true_class"] = y_test.values
    results["true_label"] = results["true_class"].map({0: "Negative", 1: "Positive"})
    
    for model_name, model in models_dict.items():
        if model is None:
            continue
        clf = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        preds = clf.predict(X_test)
        preds_series = pd.Series(preds)
        results[f"pred_{model_name}"] = preds_series.values
        results[f"pred_label_{model_name}"] = preds_series.map({0: "Negative", 1: "Positive"})
        results[f"correct_{model_name}"] = (preds_series == y_test.values)
    return results


def plot_confusion_matrices(models_dict, X_test, y_test, output_dir):
    valid_models = {k: v for k, v in models_dict.items() if v is not None}
    n_models = len(valid_models)
    if n_models == 0:
        return
    
    n_figures = (n_models + 3) // 4
    for fig_num in range(n_figures):
        start_idx = fig_num * 4
        end_idx = min(start_idx + 4, n_models)
        current_models = list(valid_models.items())[start_idx:end_idx]
        n_current = len(current_models)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(current_models):
            ax = axes[idx]
            clf = model.best_estimator_ if hasattr(model, "best_estimator_") else model
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            annotations = np.empty_like(cm, dtype='<U32')
            for i in range(2):
                for j in range(2):
                    annotations[i, j] = f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)'
            
            ax.imshow(cm, cmap='Blues', interpolation='nearest')
            for i in range(2):
                for j in range(2):
                    text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    ax.text(j, i, annotations[i, j], ha='center', va='center',
                           fontsize=11, fontweight='bold', color=text_color)
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)
            
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ax.set_title(f'{model_name.upper()}\nAcc: {acc:.3f} | Sens: {sensitivity:.2f} | Spec: {specificity:.2f}', 
                        fontsize=10, fontweight='bold')
            
            color = '#2E8B57' if acc >= 0.8 else '#FFD700' if acc >= 0.6 else '#CD5C5C'
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(3)
        
        for idx in range(n_current, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
        plt.tight_layout()
        suffix = f'_part{fig_num+1}' if n_figures > 1 else ''
        plt.savefig(os.path.join(output_dir, f'comparison_confusion_matrices{suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def save_detailed_metrics(valid_models, X_test, y_test, output_dir):
    metrics_list = []
    for model_name, model in valid_models.items():
        clf = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_test, y_pred)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        metrics_list.append({'model': model_name, 'accuracy': acc, 'sensitivity': sensitivity,
                           'specificity': specificity, 'precision': precision, 'f1_score': f1, 'mcc': mcc,
                           'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
    
    metrics_df = pd.DataFrame(metrics_list).sort_values('accuracy', ascending=False)
    metrics_df.to_csv(os.path.join(output_dir, 'detailed_model_metrics.csv'), index=False)
    return metrics_df


def plot_error_heatmap(results_df, output_dir):
    correct_cols = [c for c in results_df.columns if c.startswith("correct_")]
    model_names = [c.replace("correct_", "") for c in correct_cols]
    if not model_names:
        return
    
    status_matrix = 1 - results_df[correct_cols].values
    n_instances = len(status_matrix)
    
    fig, ax = plt.subplots(figsize=(max(8, min(15, n_instances/6)), max(4.5, len(model_names)*0.6)))
    ax.imshow(status_matrix.T, cmap=ListedColormap(['#2E8B57', '#CD5C5C']), aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels([m.upper() for m in model_names], fontsize=9)
    ax.set_title("Error Map (Green=Correct, Red=Wrong)", fontsize=10)
    
    n_ticks = min(12, n_instances)
    tick_positions = np.linspace(0, n_instances-1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, rotation=90, ha='center', fontsize=7)
    
    for i in range(len(model_names) + 1):
        ax.axhline(y=i - 0.5, color='black', linewidth=0.5, alpha=0.2)
    for i in range(0, n_instances, 20):
        ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, alpha=0.15, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_error_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()


def analyze_agreement(results_df, output_dir):
    correct_cols = [c for c in results_df.columns if c.startswith("correct_")]
    if not correct_cols:
        return None, None
    
    all_correct = results_df[correct_cols].all(axis=1)
    all_wrong = (~results_df[correct_cols]).all(axis=1)
    mixed = ~(all_correct | all_wrong)
    
    n_correct, n_wrong, n_mixed = all_correct.sum(), all_wrong.sum(), mixed.sum()
    
    correct_pos = results_df[all_correct & (results_df["true_class"] == 1)].shape[0]
    correct_neg = results_df[all_correct & (results_df["true_class"] == 0)].shape[0]
    mixed_pos = results_df[mixed & (results_df["true_class"] == 1)].shape[0]
    mixed_neg = results_df[mixed & (results_df["true_class"] == 0)].shape[0]
    wrong_pos = results_df[all_wrong & (results_df["true_class"] == 1)].shape[0]
    wrong_neg = results_df[all_wrong & (results_df["true_class"] == 0)].shape[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(11, 11))
    
    inner_sizes = [n_correct, n_mixed, n_wrong]
    wedges_inner, _ = ax.pie(inner_sizes, radius=0.6, labels=None,
        colors=['#2E8B57', '#FFD700', '#CD5C5C'],
        wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2.5), startangle=90)
    
    for i, (wedge, size) in enumerate(zip(wedges_inner, inner_sizes)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        rad = np.radians(angle)
        x, y = (0.6 - 0.15) * np.cos(rad), (0.6 - 0.15) * np.sin(rad)
        pct = size / len(results_df) * 100
        ax.text(x, y, f'{size}\n({pct:.0f}%)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    outer_sizes = [correct_pos, correct_neg, mixed_pos, mixed_neg, wrong_pos, wrong_neg]
    outer_colors = ['#E8A0A0', '#85B7EB', '#E8A0A0', '#85B7EB', '#E8A0A0', '#85B7EB']
    outer_sizes = [s for s in outer_sizes if s > 0]
    outer_colors = [outer_colors[i] for i, s in enumerate([correct_pos, correct_neg, mixed_pos, mixed_neg, wrong_pos, wrong_neg]) if s > 0]
    
    wedges_outer, _ = ax.pie(outer_sizes, radius=1.0, labels=None, colors=outer_colors,
                             wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2), startangle=90)
    
    ax.add_artist(plt.Circle((0, 0), 0.25, fc='white', linewidth=2.5, edgecolor='#DDDDDD'))
    ax.text(0, 0, f'Total\n{len(results_df)}', ha='center', va='center', fontsize=16, fontweight='bold')
    
    legend = [Patch(facecolor=c, edgecolor='white', label=l) for c, l in
              [('#2E8B57', 'All Correct'), ('#FFD700', 'Mixed'), ('#CD5C5C', 'All Wrong'),
               ('#E8A0A0', 'DIA+'), ('#85B7EB', 'DIA-')]]
    ax.legend(handles=legend, loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=10)
    ax.set_title('Model Agreement with Class Distribution', fontsize=13, fontweight='bold', pad=30)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_agreement.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    hard_instances = results_df[all_wrong].copy()
    easy_instances = results_df[all_correct].copy()
    
    if len(hard_instances) > 0:
        hard_instances.to_csv(os.path.join(output_dir, "instances_all_models_wrong.csv"), index=False)
    if len(easy_instances) > 0:
        easy_instances.to_csv(os.path.join(output_dir, "instances_all_models_correct.csv"), index=False)
    
    return hard_instances, easy_instances


def plot_detailed_comparison_table(results_df, X_test, output_dir, top_n=20):
    correct_cols = [c for c in results_df.columns if c.startswith("correct_")]
    model_names = [c.replace("correct_", "") for c in correct_cols]
    if not model_names:
        return
    
    balabanj_values = X_test.iloc[:, 1].values
    balabanj_map = {idx: balabanj_values[idx] for idx in range(len(X_test))}
    
    n_errors = len(model_names) - results_df[correct_cols].sum(axis=1)
    results_df["n_errors"] = n_errors
    
    hardest = results_df.nlargest(min(top_n, len(results_df)), "n_errors")
    easiest = results_df[(results_df["true_class"] == 1) & (results_df[correct_cols].all(axis=1))].copy()
    
    fig_height = max(8, (top_n * 0.4) + (len(easiest) * 0.4))
    fig, axes = plt.subplots(2, 1, figsize=(14, fig_height))
    
    def make_table(ax, data, title):
        if len(data) == 0:
            ax.text(0.5, 0.5, "No instances", ha='center', va='center')
            ax.axis('off')
            return
        d = data.copy()
        d["true"] = d["true_label"]
        d = d.reset_index()
        d["BalabanJ"] = d["index"].map(lambda x: f"{balabanj_map[x]:.4f}").apply(lambda x: x.rstrip('0').rstrip('.'))
        short = [m[:4] for m in model_names]
        for m, s in zip(model_names, short):
            d[s] = d[f"correct_{m}"].map({True: "✓", False: "✗"})
        cols = ["BalabanJ", "true"] + short
        table = ax.table(cellText=d[cols].values, colLabels=cols, cellLoc='center', loc='center',
                         colWidths=[0.10, 0.10] + [0.06]*len(short))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)
        for i in range(len(d)):
            for j, col in enumerate(cols):
                if col in short:
                    cell = table[(i+1, j)]
                    cell.set_facecolor("#90EE90" if d[col].iloc[i] == "✓" else "#FFB6C1")
            cell = table[(i+1, 1)]
            cell.set_facecolor("#E8F4FD" if d["true"].iloc[i] == "Positive" else "#FFF0F0")
        for j in range(len(cols)):
            table[(0, j)].set_facecolor("#4472C4")
            table[(0, j)].set_text_props(color="white", fontweight="bold")
        ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=15)
    
    make_table(axes[0], hardest, f"TOP {min(top_n, len(hardest))} HARDEST INSTANCES")
    make_table(axes[1], easiest, f"ALL {len(easiest)} EASY DIA+ INSTANCES")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_hard_vs_easy_instances.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    if len(hardest) > 0:
        h_out = hardest[["true_class"] + [f"correct_{m}" for m in model_names]].copy()
        h_out.insert(0, "BalabanJ", [balabanj_map[i] for i in hardest.index])
        h_out.to_csv(os.path.join(output_dir, "top_hardest_instances.csv"), index=False)
    if len(easiest) > 0:
        e_out = easiest[["true_class"] + [f"correct_{m}" for m in model_names]].copy()
        e_out.insert(0, "BalabanJ", [balabanj_map[i] for i in easiest.index])
        e_out.to_csv(os.path.join(output_dir, "top_easiest_dia_positive_instances.csv"), index=False)
    
    return hardest, easiest


def plot_model_ranking(results_df, output_dir):
    correct_cols = [c for c in results_df.columns if c.startswith("correct_")]
    model_names = [c.replace("correct_", "") for c in correct_cols]
    if not model_names:
        return
    
    accuracies = [results_df[c].mean() for c in correct_cols]
    sorted_idx = np.argsort(accuracies)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh([model_names[i].upper() for i in sorted_idx], [accuracies[i] for i in sorted_idx],
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names))))
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Performance Ranking", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    for bar, acc in zip(bars, [accuracies[i] for i in sorted_idx]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{acc:.4f}", va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_model_ranking.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curves(valid_models, X_test, y_test, output_dir):
    plt.figure(figsize=(10, 8))
    colors = {'decision_tree': '#1f77b4', 'random_forest': '#2ca02c', 'knn': '#ff7f0e', 'svc': '#d62728', 'xgboost': '#9467bd'}
    
    for name, model in valid_models.items():
        clf = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_test)
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            continue
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        plt.plot(recall, precision, label=f'{name.upper()} (AP={ap:.3f})', color=colors.get(name, '#333'), linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axhline(y=(y_test==1).mean(), color='gray', linestyle=':', alpha=0.7, label='Baseline')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_pr_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    X_test, y_test = load_test_data(OUTPUT_DIR)
    
    models = {}
    for name in MODELS_TO_COMPARE:
        model = load_model(OUTPUT_DIR, name)
        if model:
            models[name] = model
    
    if not models:
        return
    
    results_df = get_predictions_for_all_models(models, X_test, y_test)
    
    plot_confusion_matrices(models, X_test, y_test, OUTPUT_DIR)
    
    valid_models = {k: v for k, v in models.items() if v}
    save_detailed_metrics(valid_models, X_test, y_test, OUTPUT_DIR)
    
    plot_pr_curves(valid_models, X_test, y_test, OUTPUT_DIR)
    plot_error_heatmap(results_df, OUTPUT_DIR)
    analyze_agreement(results_df, OUTPUT_DIR)
    plot_detailed_comparison_table(results_df, X_test, OUTPUT_DIR, top_n=20)
    plot_model_ranking(results_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()