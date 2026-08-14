import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class EveningOnTheDockTests(unittest.TestCase):
    def test_all_cards_before_score_bonus_fun_only(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 9
        player.hand.extend(
            [
                make_card("biography"),
                make_card("cheap-white"),
                make_card("evening-on-the-dock"),
                make_card("biography"),
            ]
        )

        first_before = game.play_card(0, 0)
        second_before = game.play_card(0, 0)
        card = game.play_card(0, 0)
        after = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 5)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Social"}))
        self.assertEqual(game.card_fun(0, first_before), 3)
        self.assertEqual(game.card_fun(0, second_before), 4)
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, after), 2)


if __name__ == "__main__":
    unittest.main()
