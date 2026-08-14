import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WingspanTests(unittest.TestCase):
    def test_cost_tags_and_printed_fun(self) -> None:
        card = make_card("wingspan")

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))
        self.assertEqual(card.definition.base_fun, 1)

    def test_scores_immediately_and_again_at_end_of_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("wingspan"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(player.fun, 1)
        self.assertEqual(game.card_fun(0, card), 1)

        game.end_day()

        self.assertEqual(player.fun, 2)

    def test_immediate_score_uses_visible_modifiers(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 8
        player.hand.extend(
            [make_card("bring-a-friend"), make_card("wingspan")]
        )

        game.play_card(0, 0)
        wingspan = game.play_card(0, 0)

        self.assertEqual(player.fun, 2)
        self.assertEqual(game.card_fun(0, wingspan), 2)

        game.end_day()

        self.assertEqual(player.fun, 4)

    def test_immediate_score_uses_visible_modifiers_but_later_cards_only_affect_end_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("wingspan"), make_card("bring-a-friend")]
        )

        wingspan = game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.fun, 1)
        self.assertEqual(game.card_fun(0, wingspan), 1)

        game.end_day()

        self.assertEqual(player.fun, 2)


if __name__ == "__main__":
    unittest.main()
