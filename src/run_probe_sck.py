import argparse
import os
import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import all_estimators
from sklearn.exceptions import ConvergenceWarning
import warnings
from glob import glob
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_model", type=str, default="pythia-2.8b")
    parser.add_argument("--train_set", type=str, default="")
    parser.add_argument("--train_set_path", type=str, default="")
    parser.add_argument("--dev_set", type=str, default="")
    parser.add_argument("--dev_set_path", type=str, default="")
    parser.add_argument("--test_set", type=str, default="")
    parser.add_argument("--test_set_path", type=str, default="")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def compute_metrics(prediction, answers, print_result=True):
    fpr, tpr, _ = roc_curve(np.array(answers, dtype=bool), -np.array(prediction))
    auc1 = auc(fpr, tpr)
    tpr_5_fpr = tpr[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else 0.0
    
    if print_result:
        print(" AUC %.4f, TPR@5%%FPR of %.4f\n" % (auc1, tpr_5_fpr))
    
    return fpr, tpr, auc1, tpr_5_fpr


def evaluate(model, X, data):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = model.predict(X)  # fallback
    predictions = -scores
    labels = [ex["label"] for ex in data]
    fpr, tpr, auc1, tpr_5_fpr = compute_metrics(predictions, labels)
    return auc1, tpr_5_fpr


def collect_acts(dataset_name, model_name, layer, device="cpu"):
    directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "acts",
        model_name,
        dataset_name,
    )
    activation_files = glob(os.path.join(directory, f"layer_{layer}_*.pt"))
    acts = [
        torch.load(os.path.join(directory, f"layer_{layer}_{i}.pt"), map_location=device)
        for i in range(0, 25 * len(activation_files), 25)
    ]
    acts = torch.cat(acts, dim=0).cpu().numpy()
    acts = (acts - acts.mean(axis=0)) / (acts.std(axis=0) + 1e-6)
    return acts


def get_all_classifiers():
    classifiers = []
    skip_names = {
        "CategoricalNB",
        "ComplementNB",
        "DummyClassifier",
        "GaussianProcessClassifier",
        "LabelPropagation",
        "LabelSpreading",
        "MultinomialNB",
        "RadiusNeighborsClassifier"
    }

    for name, Clf in all_estimators(type_filter='classifier'):
        if name in skip_names:
            continue  # skip unwanted classifiers
        try:
            clf = make_pipeline(StandardScaler(), Clf())
            classifiers.append((name, clf))
        except Exception:
            continue  # skip classifiers that can't be instantiated
    return classifiers


def collect_combined_acts(dataset_name, model_name, layer1, layer2, device="cpu"):
    acts1 = collect_acts(dataset_name, model_name, layer1, device)
    acts2 = collect_acts(dataset_name, model_name, layer2, device)
    return np.concatenate([acts1, acts2], axis=1)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if "TinyLlama-1.1B" in args.target_model:
        layer_num = 22
    elif "open_llama_3b" in args.target_model:
        layer_num = 26
    elif "open_llama_7b" in args.target_model:
        layer_num = 32
    elif "open_llama_13b" in args.target_model:
        layer_num = 40
    elif "LLaMA-1B-dj-refine-150B" in args.target_model:
        layer_num = 24
    else:
        raise NotImplementedError

    train_set = read_jsonl(args.train_set_path)
    dev_set = read_jsonl(args.dev_set_path)
    test_set = read_jsonl(args.test_set_path)

    classifiers = get_all_classifiers()
    print("\nTrying scikit-learn classifiers: ...")
    
    for clf_name, clf in classifiers:
        print(f"\n========== Trying classifier: {clf_name} ==========")
        dev_auc_list = []
        test_auc_list = []
        failed = False

        for layer in range(layer_num):
            try:
                print(f"[INFO] Testing layer {layer}:")
                X_train = collect_acts(args.train_set, args.target_model, layer)
                y_train = np.array([ex["label"] for ex in train_set])

                X_dev = collect_acts(args.dev_set, args.target_model, layer)
                X_test = collect_acts(args.test_set, args.target_model, layer)

                clf.fit(X_train, y_train)

                dev_auc, _ = evaluate(clf, X_dev, dev_set)
                test_auc, _ = evaluate(clf, X_test, test_set)

                dev_auc_list.append(dev_auc)
                test_auc_list.append(test_auc)
            except Exception as e:
                print(f"  [Layer {layer}] Failed: {e}")
                failed = True
                break

        if not failed and dev_auc_list:
            best_layer = np.argmax(dev_auc_list)
            print(f"[{clf_name}] avg dev AUC: {np.mean(dev_auc_list):.4f}")
            print(f"[{clf_name}] Best dev AUC: {dev_auc_list[best_layer]:.4f} (layer {best_layer})")
            print(f"[{clf_name}] Test AUC at best layer: {test_auc_list[best_layer]:.4f}")
