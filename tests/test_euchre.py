import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class EuchreTests(unittest.TestCase):
    def test_cost_tags_fun_and_social_bonus_after_only(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("johnny-appleseed"), make_card("euchre"), make_card("johnny-appleseed")]
        )

        before = game.play_card(0, 0)
        card = game.play_card(0, 0)
        after = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Board Game", "Indoors"}))
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.card_fun(0, before), 1)
        self.assertEqual(game.card_fun(0, after), 2)


if __name__ == "__main__":
    unittest.main()
