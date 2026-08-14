import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CouchTubeTests(unittest.TestCase):
    def test_printed_values_and_cost_bonus_before_and_after(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("biography"),
                make_card("couch-tube"),
                make_card("waterski"),
            ]
        )

        before = game.play_card(0, 0)
        couch_tube = game.play_card(0, 0)
        after = game.play_card(0, 0)

        self.assertEqual(couch_tube.title, "Couch Tube")
        self.assertEqual(couch_tube.definition.cost, 4)
        self.assertEqual(couch_tube.definition.base_fun, 0)
        self.assertEqual(
            couch_tube.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )
        self.assertEqual(game.card_fun(0, before), 3)
        self.assertEqual(game.card_fun(0, couch_tube), 0)
        self.assertEqual(game.card_fun(0, after), 11)

    def test_other_card_bonus_resolves_in_visible_order(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("couch-tube"),
                make_card("bring-a-friend"),
                make_card("cheap-white"),
            ]
        )

        couch_tube = game.play_card(0, 0)
        bring_a_friend = game.play_card(0, 0)
        other = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, couch_tube), 0)
        self.assertEqual(game.card_fun(0, bring_a_friend), 2)
        self.assertEqual(game.card_fun(0, other), 10)


if __name__ == "__main__":
    unittest.main()
