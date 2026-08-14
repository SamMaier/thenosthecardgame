import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class OutdoorMovieTests(unittest.TestCase):
    def test_relax_cards_before_score_bonus_fun_only(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 9
        player.hand.extend(
            [
                make_card("biography"),
                make_card("waterski"),
                make_card("outdoor-movie"),
                make_card("biography"),
            ]
        )

        relax_before = game.play_card(0, 0)
        outdoors_before = game.play_card(0, 0)
        card = game.play_card(0, 0)
        relax_after = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Event", "Outdoors"}))
        self.assertEqual(game.card_fun(0, relax_before), 3)
        self.assertEqual(game.card_fun(0, outdoors_before), 6)
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.card_fun(0, relax_after), 2)


if __name__ == "__main__":
    unittest.main()
