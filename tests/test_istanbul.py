import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class IstanbulChoiceAI(RandomAI):
    def __init__(self, discard_indices, rng) -> None:
        super().__init__(rng)
        self.discard_indices = tuple(discard_indices)

    def choose_cards_to_discard(self, game, player_index, cards):
        return self.discard_indices


class IstanbulTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("istanbul")

        self.assertEqual(card.title, "Istanbul")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))

    def test_discards_selected_hand_cards_for_two_energy_each(self) -> None:
        game = empty_game()
        game.ais[0] = IstanbulChoiceAI((0, 2), game.rng)
        player = game.players[0]
        player.energy = 7
        istanbul = make_card("istanbul")
        discard_one = make_card("fajitas")
        kept = make_card("nap")
        discard_two = make_card("waterski")
        player.hand.extend([istanbul, discard_one, kept, discard_two])

        played = game.play_card(0, 0)

        self.assertEqual(player.hand, [kept])
        self.assertCountEqual(game.discard, [discard_one, discard_two])
        self.assertEqual(player.energy, 8)
        self.assertEqual(game.card_fun(0, played), 1)

    def test_discarding_zero_cards_gains_no_energy(self) -> None:
        game = empty_game()
        game.ais[0] = IstanbulChoiceAI((), game.rng)
        player = game.players[0]
        player.energy = 7
        istanbul = make_card("istanbul")
        kept = make_card("fajitas")
        player.hand.extend([istanbul, kept])

        played = game.play_card(0, 0)

        self.assertEqual(player.hand, [kept])
        self.assertEqual(game.discard, [])
        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, played), 1)


if __name__ == "__main__":
    unittest.main()
