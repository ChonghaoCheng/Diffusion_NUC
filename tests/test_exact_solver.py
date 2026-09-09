from __future__ import annotations

from itertools import product

import numpy as np

from diffusion_coverage.graph.feasibility import evaluate_assignment
from diffusion_coverage.graph.synthetic_generator import SyntheticGeneratorConfig, generate_synthetic_instance
from diffusion_coverage.problem import GraphSolution
from diffusion_coverage.problem.graph_solution import lexicographic_less, objective_tuple
from diffusion_coverage.solvers import ExactSolver, GreedyGuidedExactSolver


def independent_bruteforce(instance):
    valid_lists = [instance.valid_colours(i) for i in range(instance.num_nodes)]
    best = None
    for colours_tuple in product(*valid_lists):
        colours = np.asarray(colours_tuple, dtype=np.int64)
        sol = evaluate_assignment(instance, colours)
        if sol.better_than(best):
            best = sol
    return best


def test_exact_solver_matches_independent_bruteforce_on_small_random_graphs():
    cfg = SyntheticGeneratorConfig(
        num_nodes=7,
        num_colours=3,
        nq=4,
        edge_probability=0.35,
        valid_colour_probability=0.7,
    )
    for seed in range(20):
        instance = generate_synthetic_instance(seed=seed, config=cfg)
        exact = ExactSolver().solve(instance)
        brute = independent_bruteforce(instance)
        assert exact.feasible
        assert exact.certified_optimal
        assert exact.objective == brute.objective
        assert np.all(instance.valid_colour_mask[np.arange(instance.num_nodes), exact.colours])


def test_branch_prior_changes_search_order_not_optimum():
    cfg = SyntheticGeneratorConfig(num_nodes=6, num_colours=3, valid_colour_probability=1.0)
    instance = generate_synthetic_instance(seed=123, config=cfg)
    prior = np.zeros((instance.num_nodes, instance.num_colours))
    prior[:, 2] = 10.0
    plain = ExactSolver().solve(instance)
    guided = ExactSolver().solve(instance, branch_prior=prior)
    assert guided.objective == plain.objective
    assert np.array_equal(guided.colours >= 0, plain.colours >= 0)


def test_greedy_guided_exact_matches_exact_on_small_random_graphs():
    cfg = SyntheticGeneratorConfig(
        num_nodes=8,
        num_colours=3,
        nq=4,
        edge_probability=0.4,
        valid_colour_probability=0.7,
    )
    for seed in range(30):
        instance = generate_synthetic_instance(seed=1000 + seed, config=cfg)
        exact = ExactSolver().solve(instance)
        guided = GreedyGuidedExactSolver().solve(instance)
        assert exact.objective == guided.objective
        assert exact.certified_optimal
        assert guided.certified_optimal
        assert np.all(instance.valid_colour_mask[np.arange(instance.num_nodes), guided.colours])


def test_infeasible_incumbent_is_not_accepted():
    cfg = SyntheticGeneratorConfig(num_nodes=6, num_colours=3)
    instance = generate_synthetic_instance(seed=44, config=cfg)
    bad_colours = np.zeros(instance.num_nodes, dtype=np.int64)
    bad_colours[0] = -1
    incumbent = GraphSolution(
        colours=bad_colours,
        feasible=True,
        num_lift_offs=0,
        joint_motion_cost=0.0,
    )
    solution = ExactSolver().solve(instance, incumbent=incumbent)
    assert solution.certified_optimal
    assert solution.metadata["incumbent_accepted"] is False
    assert np.all(instance.valid_colour_mask[np.arange(instance.num_nodes), solution.colours])


def test_lexicographic_objective_comparison():
    a = GraphSolution(np.array([0]), True, num_lift_offs=0, joint_motion_cost=100.0)
    b = GraphSolution(np.array([0]), True, num_lift_offs=1, joint_motion_cost=0.0)
    c = GraphSolution(np.array([0]), True, num_lift_offs=0, joint_motion_cost=50.0)
    assert objective_tuple(a) == (0, 100.0)
    assert lexicographic_less(a.objective, b.objective)
    assert lexicographic_less(c.objective, a.objective)
    assert c.better_than(a)


def test_search_statistics_are_nonnegative_and_consistent():
    instance = generate_synthetic_instance(seed=55, config=SyntheticGeneratorConfig(num_nodes=7, num_colours=3))
    exact = ExactSolver().solve(instance)
    guided = GreedyGuidedExactSolver().solve(instance)
    for solution in [exact, guided]:
        assert solution.search_nodes >= 0
        assert solution.evaluated_assignments >= 0
        assert solution.solve_time >= 0.0
        assert solution.metadata["time_to_certified"] is not None
        assert solution.metadata["time_to_certified"] >= 0.0
        if solution.metadata["time_to_best"] is not None:
            assert solution.metadata["time_to_best"] <= solution.metadata["time_to_certified"] + 1e-9
        if solution.metadata["time_to_first_feasible"] is not None:
            assert solution.metadata["time_to_first_feasible"] <= solution.metadata["time_to_certified"] + 1e-9
