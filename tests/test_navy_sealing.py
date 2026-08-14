import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class NavySEALingAI(RandomAI):
    def __init__(self, discard_indices, rng) -> None:
        super().__init__(rng)
        self.discard_indices = tuple(discard_indices)

    def choose_cards_to_discard(self, game, player_index, hand):
        return self.discard_indices


class NavySEALingTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("navy-sealing")

        self.assertEqual(card.title, "Navy SEALing")
        self.assertEqual(card.definition.cost, 6)
        self.assertEqual(card.definition.base_fun, 4)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_discards_selected_hand_cards_for_three_fun_each(self) -> None:
        game = empty_game()
        game.ais[0] = NavySEALingAI((0, 2), game.rng)
        player = game.players[0]
        player.energy = 7
        first_discard = make_card("biography")
        retained = make_card("nap")
        second_discard = make_card("waterski")
        second_discard.markers["test"] = True
        navy_sealing = make_card("navy-sealing")
        player.hand = [navy_sealing, first_discard, retained, second_discard]

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 1)
        self.assertEqual(player.hand, [retained])
        self.assertEqual(
            {id(discarded) for discarded in game.discard},
            {id(first_discard), id(second_discard)},
        )
        self.assertEqual(second_discard.markers, {})
        self.assertEqual(card.markers["energy_cubes"], 2)
        self.assertEqual(game.card_fun(0, card), 10)

    def test_may_discard_zero_cards(self) -> None:
        game = empty_game()
        game.ais[0] = NavySEALingAI((), game.rng)
        player = game.players[0]
        player.energy = 6
        navy_sealing = make_card("navy-sealing")
        player.hand.append(navy_sealing)

        card = game.play_card(0, 0)

        self.assertEqual(player.hand, [])
        self.assertEqual(game.discard, [])
        self.assertEqual(card.markers["energy_cubes"], 0)
        self.assertEqual(game.card_fun(0, card), 4)


if __name__ == "__main__":
    unittest.main()
