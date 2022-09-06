import sys
import pandas as pd
from pathlib import Path

def get_results(directory, metrics=["accuracy"], is_print=False, is_return=True, is_save=True):
    paths = list(directory.glob("eval_*/model_*/**/train_size_-1/seed_0/all_metrics.csv"))
    paths += list(directory.glob("eval/model_*/**/train_size_-1/seed_0/all_metrics.csv"))

    if len(paths) == 0:
        if is_print:
            print(f"No results found in {directory}")
        return

    results = dict()
    for p in paths:
        if "eval_" in str(p):
            eval_dir = str(p).split("eval_")[1].split("/")[0]
        else:
            eval_dir = "?"
        results[eval_dir] = pd.read_csv(p, index_col=0)[metrics].T
        results[eval_dir] = results[eval_dir].unstack().to_frame().sort_index(level=1).T
        results[eval_dir].columns = results[eval_dir].columns.map('_'.join)

    all_results = pd.concat(results).droplevel(1)

    if len(metrics) == 1:
        max_row = all_results[all_results[f"test_{metrics[0]}"] == all_results[f"test_{metrics[0]}"].max()].iloc[[0]]  # takes first (arbitrary)
        max_row = max_row.reset_index().rename(columns={"index": "probe"})
        max_row.index = ["best"]
        all_results = pd.concat([all_results, max_row], axis=0)

    if is_print:
        print(all_results)

    if is_save:
        all_results.to_csv(directory/"all_results.csv")

    if is_return:
        return all_results

if __name__ == "__main__":
    get_results(directory=Path(sys.argv[1]), is_print=True, is_return=False)