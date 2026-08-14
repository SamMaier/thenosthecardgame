import unittest

from thenos.cards import CARD_REGISTRY
from thenos.simulation import simulate_games


class SimulationTests(unittest.TestCase):
    def test_batch_collects_pick_play_and_win_statistics(self) -> None:
        report = simulate_games(10, seed=2026)

        self.assertEqual(report.games, 10)
        self.assertEqual(set(report.cards), {card.title for card in CARD_REGISTRY.values()})
        self.assertEqual(sum(card.picks for card in report.cards.values()), 777)
        for card in report.cards.values():
            self.assertGreater(card.offers, 0)
            self.assertGreater(card.picks, 0)
            self.assertGreaterEqual(card.offers, card.picks)
            self.assertGreaterEqual(card.acquisitions, card.picks)
            self.assertGreaterEqual(card.plays, 0)
            self.assertLessEqual(card.play_rate, 1.0)
            self.assertGreaterEqual(card.win_rate_when_picked, 0.0)
            self.assertLessEqual(card.win_rate_when_picked, 1.0)
            self.assertGreaterEqual(card.win_rate_when_acquired, 0.0)
            self.assertLessEqual(card.win_rate_when_acquired, 1.0)


if __name__ == "__main__":
    unittest.main()
