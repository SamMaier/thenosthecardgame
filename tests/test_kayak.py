import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class EnergyChoiceAI(RandomAI):
    def __init__(self, energy_to_spend: int, rng: random.Random) -> None:
        super().__init__(rng)
        self.energy_to_spend = energy_to_spend

    def choose_energy_to_spend(
        self, game, player_index, card, maximum
    ) -> int:
        return self.energy_to_spend


class KayakTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("kayak")

        self.assertEqual(card.title, "Kayak")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_can_spend_remaining_energy_for_fun(self) -> None:
        game = empty_game()
        game.ais[0] = EnergyChoiceAI(3, game.rng)
        player = game.players[0]
        player.energy = 5
        player.hand.append(make_card("kayak"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(card.markers["energy_cubes"], 3)
        self.assertEqual(game.card_fun(0, card), 5)

    def test_optional_energy_cannot_exceed_energy_left_after_cost(self) -> None:
        game = empty_game()
        game.ais[0] = EnergyChoiceAI(4, game.rng)
        player = game.players[0]
        player.energy = 5
        player.hand.append(make_card("kayak"))

        with self.assertRaisesRegex(ValueError, "optional Energy"):
            game.play_card(0, 0)

    def test_spending_no_additional_energy_keeps_base_fun(self) -> None:
        game = empty_game()
        game.ais[0] = EnergyChoiceAI(0, game.rng)
        player = game.players[0]
        player.energy = 2
        player.hand.append(make_card("kayak"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(card.markers["energy_cubes"], 0)
        self.assertEqual(game.card_fun(0, card), 2)


if __name__ == "__main__":
    unittest.main()
