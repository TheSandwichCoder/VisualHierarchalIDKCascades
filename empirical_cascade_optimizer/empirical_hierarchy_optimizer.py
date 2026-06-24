from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from ._paths import ROOT as _ROOT
except ImportError:
    from _paths import ROOT as _ROOT

from empirical_cascade_optimizer.empirical_outcomes import load_empirical_outcomes


@dataclass
class EmpiricalCascade:
    expected_cost: float
    initial: list[str]
    specialized: dict[tuple[str, int], list[str]]
    detector: str


class EmpiricalHierarchyOptimizer:
    def __init__(self, payload):
        self.labels = payload["labels"]
        self.candidates = payload["candidates"].set_index("id", drop=False)
        self.detector = payload["detector"]
        self.num_groups = int(self.labels["true_group"].max()) + 1 if not self.labels.empty else 0
        self.sample_count = len(self.labels)

        self.accepted = {}
        self.prediction = {}
        for candidate_id, group in payload["outcomes"].groupby("candidate_id", sort=False):
            ordered = group.sort_values("sample_id")
            self.accepted[candidate_id] = ordered["accepted"].to_numpy(dtype=bool)
            self.prediction[candidate_id] = ordered["prediction"].to_numpy(dtype=int)

        self.global_ids = tuple(
            self.candidates[self.candidates["kind"] == "global"].index.tolist()
        )
        self.identifier_ids = tuple(
            self.candidates[self.candidates["kind"] == "identifier"].index.tolist()
        )
        self.initial_ids = tuple(self.global_ids + self.identifier_ids)
        self.specialized_by_group = {
            group_id: tuple(
                self.candidates[
                    (self.candidates["kind"] == "specialized")
                    & (self.candidates["group_id"] == group_id)
                ].index.tolist()
            )
            for group_id in range(self.num_groups)
        }
        self.detector_id = "detector"
        self.detector_cost = float(self.detector["cost"])

        self._expand_next = {}
        self._expand_prime_next = {}

    def _cost(self, candidate_id):
        return float(self.candidates.loc[candidate_id, "cost"])

    def _idk_mask(self, candidate_id):
        return ~self.accepted[candidate_id]

    def _eligible_initial(self, rejected_initial):
        mask = np.ones(self.sample_count, dtype=bool)
        for candidate_id in rejected_initial:
            mask &= self._idk_mask(candidate_id)
        return mask

    def _eligible_specialized(self, rejected_initial, group_id, rejected_specialized, router_id):
        mask = self._eligible_initial(rejected_initial)
        mask &= self.accepted[router_id]
        mask &= self.prediction[router_id] == group_id
        for candidate_id in rejected_specialized:
            mask &= self._idk_mask(candidate_id)
        return mask

    @staticmethod
    def _prob(mask):
        denominator = int(mask.size)
        if denominator == 0:
            return 0.0
        return float(mask.sum()) / denominator

    def _initial_probs(self, rejected_initial, candidate_id):
        eligible = self._eligible_initial(rejected_initial)
        denominator = int(eligible.sum())
        if denominator == 0:
            return 1.0, [0.0 for _ in range(self.num_groups)]

        idk_probability = self._prob(eligible & self._idk_mask(candidate_id)) / self._prob(eligible)

        group_probabilities = []
        if candidate_id in self.identifier_ids:
            for group_id in range(self.num_groups):
                group_mask = (
                    eligible
                    & self.accepted[candidate_id]
                    & (self.prediction[candidate_id] == group_id)
                )
                group_probabilities.append(float(group_mask.sum()) / denominator)
        else:
            group_probabilities = [0.0 for _ in range(self.num_groups)]

        return idk_probability, group_probabilities

    def _specialized_idk_prob(self, rejected_initial, group_id, rejected_specialized, router_id, candidate_id):
        eligible = self._eligible_specialized(
            rejected_initial,
            group_id,
            rejected_specialized,
            router_id,
        )
        denominator = int(eligible.sum())
        if denominator == 0:
            return 1.0
        return float((eligible & self._idk_mask(candidate_id)).sum()) / denominator

    @lru_cache(maxsize=None)
    def expand(self, rejected_initial):
        rejected_initial = tuple(rejected_initial)
        rejected_set = set(rejected_initial)
        remaining = [
            candidate_id
            for candidate_id in self.initial_ids
            if candidate_id not in rejected_set
        ]

        best_cost = self.detector_cost
        best_next = self.detector_id

        for candidate_id in remaining:
            idk_probability, group_probabilities = self._initial_probs(
                rejected_initial,
                candidate_id,
            )
            candidate_cost = self._cost(candidate_id)
            cost = candidate_cost + idk_probability * self.expand(
                tuple(sorted(rejected_set | {candidate_id}))
            )

            if candidate_id in self.identifier_ids:
                for group_id, group_probability in enumerate(group_probabilities):
                    if group_probability == 0.0:
                        continue
                    cost += group_probability * self.expand_prime(
                        rejected_initial,
                        group_id,
                        tuple(),
                        candidate_id,
                    )

            if cost < best_cost:
                best_cost = cost
                best_next = candidate_id

        self._expand_next[rejected_initial] = best_next
        return best_cost

    @lru_cache(maxsize=None)
    def expand_prime(self, rejected_initial, group_id, rejected_specialized, router_id):
        rejected_initial = tuple(rejected_initial)
        rejected_specialized = tuple(rejected_specialized)
        rejected_initial_set = set(rejected_initial)
        rejected_specialized_set = set(rejected_specialized)

        remaining_globals = [
            candidate_id
            for candidate_id in self.global_ids
            if candidate_id not in rejected_initial_set
        ]
        remaining_specialized = [
            candidate_id
            for candidate_id in self.specialized_by_group.get(group_id, tuple())
            if candidate_id not in rejected_specialized_set
        ]

        best_cost = self.detector_cost
        best_next = self.detector_id

        for candidate_id in remaining_globals + remaining_specialized:
            idk_probability = self._specialized_idk_prob(
                rejected_initial,
                group_id,
                rejected_specialized,
                router_id,
                candidate_id,
            )
            candidate_cost = self._cost(candidate_id)

            if candidate_id in self.global_ids:
                next_rejected_initial = tuple(sorted(rejected_initial_set | {candidate_id}))
                cost = candidate_cost + idk_probability * self.expand_prime(
                    next_rejected_initial,
                    group_id,
                    rejected_specialized,
                    router_id,
                )
            else:
                next_rejected_specialized = tuple(sorted(rejected_specialized_set | {candidate_id}))
                cost = candidate_cost + idk_probability * self.expand_prime(
                    rejected_initial,
                    group_id,
                    next_rejected_specialized,
                    router_id,
                )

            if cost < best_cost:
                best_cost = cost
                best_next = candidate_id

        key = (rejected_initial, group_id, rejected_specialized, router_id)
        self._expand_prime_next[key] = best_next
        return best_cost

    def synthesize(self):
        expected_cost = self.expand(tuple())
        initial = []
        specialized = {}
        rejected_initial = tuple()

        while True:
            next_id = self._expand_next.get(rejected_initial, self.detector_id)
            initial.append(next_id)
            if next_id == self.detector_id:
                break

            if next_id in self.identifier_ids:
                prefix = rejected_initial
                for group_id in range(self.num_groups):
                    specialized[(next_id, group_id)] = self._synthesize_specialized(
                        prefix,
                        group_id,
                        next_id,
                    )

            rejected_initial = tuple(sorted(set(rejected_initial) | {next_id}))

        return EmpiricalCascade(
            expected_cost=expected_cost,
            initial=initial,
            specialized=specialized,
            detector=self.detector_id,
        )

    def _synthesize_specialized(self, rejected_initial, group_id, router_id):
        chain = []
        rejected_initial = tuple(rejected_initial)
        rejected_specialized = tuple()

        while True:
            key = (rejected_initial, group_id, rejected_specialized, router_id)
            next_id = self._expand_prime_next.get(key, self.detector_id)
            chain.append(next_id)
            if next_id == self.detector_id:
                break
            if next_id in self.global_ids:
                rejected_initial = tuple(sorted(set(rejected_initial) | {next_id}))
            else:
                rejected_specialized = tuple(sorted(set(rejected_specialized) | {next_id}))

        return chain

    def describe_candidate(self, candidate_id):
        if candidate_id == self.detector_id:
            return {
                "id": self.detector_id,
                "kind": "detector",
                "name": self.detector.get("name"),
                "cost": self.detector_cost,
            }
        row = self.candidates.loc[candidate_id]
        group_id = row["group_id"]
        return {
            "id": candidate_id,
            "kind": row["kind"],
            "group_id": None if pd.isna(group_id) else int(group_id),
            "name": row["name"],
            "threshold": row["threshold"],
            "cost": row["cost"],
            "precision": row["precision"],
            "recall": row["recall"],
        }


def optimize_empirical_hierarchy(path="models/stats/empirical_outcomes.pkl"):
    df = load_empirical_outcomes(path)
    optimizer = EmpiricalHierarchyOptimizer(df)
    cascade = optimizer.synthesize()
    return optimizer, cascade


def print_empirical_cascade(path="models/stats/empirical_outcomes.pkl"):
    optimizer, cascade = optimize_empirical_hierarchy(path)
    print(f"expected_cost: {cascade.expected_cost}")
    print("initial:")
    for candidate_id in cascade.initial:
        print("  ", optimizer.describe_candidate(candidate_id))
    print("specialized:")
    for (router_id, group_id), chain in cascade.specialized.items():
        chain_desc = [optimizer.describe_candidate(candidate_id)["name"] for candidate_id in chain]
        print(f"  {router_id} group={group_id}: {chain_desc}")
    return optimizer, cascade


if __name__ == "__main__":
    print_empirical_cascade()
