import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetHandAI(RandomAI):
    def __init__(self, target_title, rng):
        super().__init__(rng)
        self.target_title = target_title

    def choose_card_to_discard(self, game, player_index, hand):
        return next(
            index for index, card in enumerate(hand)
            if card.title == self.target_title
        )


class TellAStoryTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("tell-a-story")

        self.assertEqual(card.title, "Tell a Story")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 4)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))

    def test_moves_a_chosen_hand_card_to_the_top_of_the_trunk(self) -> None:
        game = empty_game()
        game.ais[0] = TargetHandAI("Non-Fiction Book", random.Random(0))
        player = game.players[0]
        player.energy = 7
        fajitas = make_card("fajitas")
        biography = make_card("biography")
        biography.markers["energy_cube"] = True
        game.trunk.append(make_card("azul"))
        player.hand.extend([make_card("tell-a-story"), fajitas, biography])

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 4)
        self.assertEqual(player.hand, [fajitas])
        self.assertEqual(game.discard, [])
        self.assertIs(game.trunk[-1], biography)
        self.assertEqual(biography.markers, {})

    def test_does_nothing_with_an_empty_hand(self) -> None:
        game = empty_game()
        game.players[0].energy = 3
        game.players[0].hand.append(make_card("tell-a-story"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 4)
        self.assertEqual(game.players[0].hand, [])
        self.assertEqual(game.trunk, [])


if __name__ == "__main__":
    unittest.main()
