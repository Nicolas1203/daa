import pandas as pd
import numpy as np


def forgetting_table(acc_dataframe, n_tasks=5):
    return pd.DataFrame(    
        [forgetting_line(acc_dataframe, task_id=i).values[:,-1].tolist() for i in range(0, n_tasks)]
    )

def forgetting_line(acc_dataframe, task_id=4, n_tasks=5):
    if task_id == 0:
        forgettings = [np.nan] * n_tasks
    else:
        forgettings = [forgetting(task_id, p, acc_dataframe) for p in range(task_id)] + [np.nan]*(n_tasks-task_id)

    # Create dataframe to handle NaN
    return pd.DataFrame(forgettings)

def forgetting(q, p, df):
    D = {}
    for i in range(0, q+1):
        D[f"d{i}"] = df.diff(periods=-i)

    # Create datafrmae to handle NaN
    return pd.DataFrame(([D[f'd{k}'].iloc[q-k,p] for k in range(0, q+1)])).max()[0]

def get_rand_perfs(n_tasks, n_classes, metric):
    """Computes the performances (accuracy of forgetting) of a random classifier.
    It assumes every task has the same number of classes.

    Args:
        n_tasks (int): Number of total tasks
        n_classes (int): Number of classes 
        metric (str): acc or forgetting

    Returns:
        Dataframe corresopnding the acuracy or forgetting table respectively.
    """
    rand_perf = np.ones(n_tasks) / ((np.arange(n_tasks) + 1) * (n_classes // n_tasks))
    if metric == "forgetting":
        arr = np.repeat(np.expand_dims(rand_perf, 0), n_tasks).reshape(n_tasks, n_tasks) * np.tril(np.ones(n_tasks), 0)
        df = pd.DataFrame(arr)
        rand_perf = np.nan_to_num(np.nanmean(forgetting_table(df, n_tasks).values, 1))
    rand_perf = np.expand_dims(rand_perf, 0)

    return rand_perf

def raa(acc, n_tasks, n_classes):
    rand_clf_acc = get_rand_perfs(n_tasks, n_classes, metric='acc')
    raa = (acc /rand_clf_acc) / (1/n_classes)

    return raa

def raf(fgt, n_tasks, n_classes):
    rand_clf_fgt = get_rand_perfs(n_tasks, n_classes, metric='forgetting')

    # This uses the fact that max(uRAF) = \frac{1}{AF_K(Rand_{C_K})}
    # This can be seen by combining Lemma B.1 and Lemma B.2 from our paper.
    raf = (fgt / rand_clf_fgt) * rand_clf_fgt.mean(0)[-1] # raf = uRAF * AF_K(Rand_CK)
    return raf