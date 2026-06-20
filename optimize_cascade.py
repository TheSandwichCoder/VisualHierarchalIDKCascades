import ast
from itertools import permutations
from pathlib import Path

import pandas as pd

from read_json import load_groups


class ClassifierNode:
    def __init__(self, cost, idk_prob, recall, precision, name, checkpoint_path, threshold):
        self.c = cost
        self.raw_cost = cost
        self.d = idk_prob

        self.r = recall
        self.p = precision

        self.next_seq = None
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold
        
    def show(self):
        print(f"{self.name} c:{self.c} cost:{self.raw_cost} idk-prob:{self.d} recall:{self.r} precision:{self.p}")


class SpecClassifierNode(ClassifierNode):
    def __init__(self, cost, idk_prob, recall, precision, name=None, checkpoint_path=None, threshold=None):
        super().__init__(cost, idk_prob, recall, precision, name, checkpoint_path, threshold)

class IdenClassifierNode(ClassifierNode):
    def __init__(self, cost, idk_prob, recall, precision, probs, classes=None, name=None, checkpoint_path=None, threshold=None):
        super().__init__(cost, idk_prob, recall, precision, name, checkpoint_path, threshold)
        self.class_probs = probs
        self.n_classes = classes if classes is not None else len(probs)

class GlobClassifierNode(ClassifierNode):
    def __init__(self, cost, idk_prob, recall, precision, name=None, checkpoint_path=None, threshold=None):
        super().__init__(cost, idk_prob, recall, precision, name, checkpoint_path, threshold)
        
def _dependent_specialized_cost(classifiers, c_det):
    total_cost = 0.0
    reach_prob = 1.0
    covered_recall = 0.0

    for classifier in classifiers:
        total_cost += reach_prob * classifier.raw_cost
        covered_recall = max(covered_recall, float(classifier.r))
        reach_prob = 1.0 - covered_recall

    return total_cost + reach_prob * c_det.raw_cost


def _annotate_dependent_specialized_sequence(classifiers, c_det):
    seq = list(classifiers) + [c_det]
    covered_recall = 0.0

    for index, classifier in enumerate(classifiers):
        classifier.c = _dependent_specialized_cost(classifiers[index:], c_det)
        classifier.d = 1.0 - max(covered_recall, float(classifier.r))
        classifier.next_seq = seq[index + 1:]
        covered_recall = max(covered_recall, float(classifier.r))

    c_det.c = c_det.raw_cost
    c_det.d = 0.0
    c_det.next_seq = None
    return seq


def get_specialized_cost(classifiers, c_det):
    best_cost = c_det.raw_cost
    best_classifiers = []

    for length in range(1, len(classifiers) + 1):
        for candidate in permutations(classifiers, length):
            cost = _dependent_specialized_cost(candidate, c_det)
            if cost < best_cost:
                best_cost = cost
                best_classifiers = list(candidate)

    seq = _annotate_dependent_specialized_sequence(best_classifiers, c_det)
    if best_classifiers:
        seq[0].c = best_cost

    return best_cost, seq


def get_identifier_cost(identifier, spec_classifiers):
    total_cost = 0

    for c_i in range(identifier.n_classes):
        d_i = identifier.class_probs[c_i]

        total_cost += spec_classifiers[c_i].c * d_i

    total_cost = identifier.raw_cost + (1 - identifier.d) * total_cost
    identifier.c = total_cost
    return total_cost

# c_all comprises of c_iden and c_glob
def get_cascade_cost(c_all, c_det):
    # no more classifiers then return det
    if not c_all:
        return c_det.c, [c_det]

    best_cost, seq = get_cascade_cost(c_all[1:], c_det)
    
    c_curr = c_all[0]

    if best_cost <= 0:
        c_curr.c = best_cost
        c_curr.next_seq = seq
        return best_cost, seq

    if c_curr.d < (best_cost - c_curr.raw_cost) / best_cost:
        new_cost = c_curr.raw_cost + best_cost * c_curr.d
        c_curr.c = new_cost
        c_curr.next_seq = seq
        return new_cost, [c_curr] + seq
    
    else:
        return best_cost, seq

def _read_stats(path):
    if path is None:
        return pd.DataFrame()

    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    return pd.read_csv(path)


def _parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(value)


def _has_value(value):
    if isinstance(value, list):
        return True
    return not pd.isna(value)


def _get_category_lookup(groups):
    lookup = {}
    for category_id, group in enumerate(groups):
        lookup[group["name"]] = category_id
        lookup[tuple(int(item["index"]) for item in group["classes"])] = category_id
    return lookup


def _infer_specialized_category_id(row, category_lookup):
    class_ids = _parse_list(row.get("class_ids"))
    category_id = category_lookup.get(tuple(int(class_id) for class_id in class_ids))
    if category_id is not None:
        return category_id

    model_name = str(row.get("model_name", ""))
    for group_name, group_category_id in category_lookup.items():
        if isinstance(group_name, str) and model_name.startswith(group_name):
            return group_category_id

    return None


def _get_probability_vector(row):
    distribution_columns = [
        column
        for column in row.index
        if column.startswith("accepted_distribution_")
    ]
    if distribution_columns:
        distribution_columns = sorted(distribution_columns)
        return [float(row[column]) for column in distribution_columns]

    if "accepted_category_distribution" in row and _has_value(row["accepted_category_distribution"]):
        return [float(value) for value in _parse_list(row["accepted_category_distribution"])]

    probability_columns = [
        column
        for column in row.index
        if column.startswith("predicted_probability_")
    ]
    if probability_columns:
        probability_columns = sorted(probability_columns)
        return [float(row[column]) for column in probability_columns]

    if "predicted_category_probabilities" in row and _has_value(row["predicted_category_probabilities"]):
        return [float(value) for value in _parse_list(row["predicted_category_probabilities"])]

    return []


def _best_detector_row(det_df):
    if "threshold_kind" in det_df.columns:
        fixed_rows = det_df[det_df["threshold_kind"] == "fixed"]
        zero_threshold_rows = fixed_rows[fixed_rows["confidence_threshold"].fillna(0.0) == 0.0]
        if not zero_threshold_rows.empty:
            det_df = zero_threshold_rows
        elif not fixed_rows.empty:
            det_df = fixed_rows

    return det_df.sort_values(
        ["recall", "accuracy", "average_runtime"],
        ascending=[False, False, True],
    ).iloc[0]


def _drop_dominated_threshold_rows(df):
    if df.empty or "checkpoint_path" not in df.columns:
        return df

    sort_columns = ["checkpoint_path", "recall", "precision", "average_runtime"]
    sort_columns = [column for column in sort_columns if column in df.columns]
    ascending = [True, False, False, True][:len(sort_columns)]

    return (
        df.sort_values(sort_columns, ascending=ascending)
        .drop_duplicates("checkpoint_path", keep="first")
    )


# for getting the specialized nodes it will be a 2d array of the ones above 95% precision
# specialized nodes should also be in the same order as class_probs so that they line up
def get_classifiers(
    specialized_path=None,
    identifier_path=None,
    globals_path=None,
    det_path=None,
    min_precision=0.75,
):
    groups = load_groups()["groups"]
    category_lookup = _get_category_lookup(groups)
    specialized_nodes = [[] for _ in groups]

    specialized_df = _read_stats(specialized_path)
    if not specialized_df.empty:
        specialized_df = specialized_df[specialized_df["precision"] >= min_precision]
        specialized_df = _drop_dominated_threshold_rows(specialized_df)
        for _, row in specialized_df.iterrows():
            category_id = _infer_specialized_category_id(row, category_lookup)
            if category_id is None:
                continue
            
            specialized_nodes[category_id].append(SpecClassifierNode(
                cost=float(row["average_runtime"]),
                idk_prob=1 - float(row["recall"]),
                recall= row["recall"],
                precision= row["precision"],
                name=row.get("model_name"),
                
                checkpoint_path=row.get("checkpoint_path"),
                threshold=row.get("confidence_threshold"),
            ))

    identifier_nodes = []
    identifier_df = _read_stats(identifier_path)
    if not identifier_df.empty:
        identifier_df = identifier_df[identifier_df["precision"] >= min_precision]
        identifier_df = _drop_dominated_threshold_rows(identifier_df)
        for _, row in identifier_df.iterrows():
            class_probs = _get_probability_vector(row)
            if not class_probs:
                continue
            if len(class_probs) != len(groups):
                raise ValueError(
                    f"Identifier stats for {row.get('model_name')} have "
                    f"{len(class_probs)} class probabilities, expected {len(groups)}. "
                    "Did you pass detector/global stats as identifier_path?"
                )

            identifier_nodes.append(IdenClassifierNode(
                cost=float(row["average_runtime"]),
                idk_prob=1 - float(row["recall"]),
                recall= row["recall"],
                precision= row["precision"],
                probs=class_probs,
                name=row.get("model_name"),
                checkpoint_path=row.get("checkpoint_path"),
                threshold=row.get("confidence_threshold"),
            ))

    globals_nodes = []
    globals_df = _read_stats(globals_path)
    if not globals_df.empty:
        globals_df = globals_df[globals_df["precision"] >= min_precision]
        globals_df = _drop_dominated_threshold_rows(globals_df)
        for _, row in globals_df.iterrows():
            globals_nodes.append(GlobClassifierNode(
                cost=float(row["average_runtime"]),
                idk_prob=1 - float(row["recall"]),
                recall= row["recall"],
                precision= row["precision"],
                name=row.get("model_name"),
                checkpoint_path=row.get("checkpoint_path"),
                threshold=row.get("confidence_threshold"),
            ))

    det_df = _read_stats(det_path)
    if det_df.empty:
        raise ValueError(
            "det_path must point to a non-empty detector stats CSV or pickle. "
            "A zero-cost fallback detector would make the cascade optimizer invalid."
        )
    else:
        det_row = _best_detector_row(det_df)
        det_node = GlobClassifierNode(
            cost=float(det_row["average_runtime"]),
            idk_prob=0.0,
            precision=det_row["precision"],
            recall=det_row["recall"],
            name=det_row.get("model_name"),
            checkpoint_path=det_row.get("checkpoint_path"),
            threshold=det_row.get("confidence_threshold"),
        )

    return specialized_nodes, identifier_nodes, globals_nodes, det_node

# takes in a list of nodes and returns a sorted list (doesn't have to be copies of the objects) based on their cost
def sort_nodes_on_cost(nodes):
    return sorted(nodes, key=lambda node: node.c)
    
def optimize_cascade(
    specialized_path="models/stats/specialized_stats.pkl",
    identifier_path="models/stats/identifier_stats.pkl",
    globals_path="models/stats/global_stats.pkl",
    det_path="models/stats/det_stats.pkl",
    min_precision=0.80,
    return_specialized=False,
):
    specialized_nodes, identifier_nodes, globals_nodes, det_node = get_classifiers(
        specialized_path,
        identifier_path,
        globals_path,
        det_path,
        min_precision=min_precision,
    )

    temp = []
    for nds in specialized_nodes:
        if not nds:
            temp.append(det_node)
            continue
        x, seq = get_specialized_cost(sort_nodes_on_cost(nds), det_node)
        temp.append(seq[0])

    specialized_nodes = temp

    

    for nd in identifier_nodes:
        c = get_identifier_cost(nd, specialized_nodes)


    c, seq = get_cascade_cost(sort_nodes_on_cost(identifier_nodes + globals_nodes), det_node)

    if return_specialized:
        return c, seq, specialized_nodes

    return c, seq

if __name__ == "__main__":
    c, seq = optimize_cascade()

    for s in seq:
        s.show()
