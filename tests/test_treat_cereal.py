import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class TreatCerealTests(unittest.TestCase):
    def test_next_exercise_costs_four_less_and_only_once(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("treat-cereal"), make_card("waterski"), make_card("canoe")]
        )

        card = game.play_card(0, 0)
        first_exercise = game.play_card(0, 0)
        later_exercise = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(game.energy_cost(0, later_exercise), 3)
        self.assertEqual(player.energy, 1)
        self.assertEqual(first_exercise.definition.cost, 5)


if __name__ == "__main__":
    unittest.main()
