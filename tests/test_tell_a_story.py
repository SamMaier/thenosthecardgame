import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetDiscardAI(RandomAI):
    def __init__(self, target_title, rng):
        super().__init__(rng)
        self.target_title = target_title

    def choose_card_target(self, game, player_index, eligible_cards):
        return next(
            index for index, card in enumerate(eligible_cards)
            if card.title == self.target_title
        )


class TellAStoryTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("tell-a-story")

        self.assertEqual(card.title, "Tell a Story")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))

    def test_moves_a_chosen_discard_to_the_top_of_the_trunk(self) -> None:
        game = empty_game()
        game.ais[0] = TargetDiscardAI("Biography", random.Random(0))
        player = game.players[0]
        player.energy = 7
        fajitas = make_card("fajitas")
        biography = make_card("biography")
        game.discard.extend([fajitas, biography])
        game.trunk.append(make_card("azul"))
        player.hand.append(make_card("tell-a-story"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.discard, [fajitas])
        self.assertIs(game.trunk[-1], biography)

    def test_does_nothing_with_an_empty_discard_pile(self) -> None:
        game = empty_game()
        game.players[0].energy = 3
        game.players[0].hand.append(make_card("tell-a-story"))

        game.play_card(0, 0)

        self.assertEqual(game.discard, [])


if __name__ == "__main__":
    unittest.main()
