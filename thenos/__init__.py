"""Headless simulator for The Nos."""

from thenos.game import Game, GameResult
from thenos.simulation import (
    SimulationReport,
    simulate_argonbrain_vs_planner,
    simulate_beryllium_vs_planner,
    simulate_berylliumbrain_vs_planner,
    simulate_four_argonbrain,
    simulate_four_beryllium,
    simulate_four_berylliumbrain,
    simulate_four_galaxybrain,
    simulate_galaxybrain_vs_planner,
    simulate_games,
    simulate_greedy_vs_random,
    write_report_csv,
)

__all__ = [
    "Game",
    "GameResult",
    "SimulationReport",
    "simulate_argonbrain_vs_planner",
    "simulate_beryllium_vs_planner",
    "simulate_berylliumbrain_vs_planner",
    "simulate_four_argonbrain",
    "simulate_four_beryllium",
    "simulate_four_berylliumbrain",
    "simulate_four_galaxybrain",
    "simulate_galaxybrain_vs_planner",
    "simulate_games",
    "simulate_greedy_vs_random",
    "write_report_csv",
]
