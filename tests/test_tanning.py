import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class TanningTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("tanning")

        self.assertEqual(card.title, "Tanning")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 5)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Outdoors"}))

    def test_must_be_first_outdoors_card_but_non_outdoors_cards_can_precede_it(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 5
        player.hand.extend([make_card("biography"), make_card("tanning")])

        game.play_card(0, 0)
        tanning = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, tanning), 5)

        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.extend([make_card("dock-fishing"), make_card("tanning")])

        game.play_card(0, 0)
        with self.assertRaises(ValueError):
            game.play_card(0, 0)

    def test_later_outdoors_cards_score_one_less_and_tanning_is_not_penalized(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("tanning"), make_card("dock-fishing"), make_card("biography")]
        )

        tanning = game.play_card(0, 0)
        outdoors_after = game.play_card(0, 0)
        non_outdoors_after = game.play_card(0, 0)

        self.assertEqual(player.energy, 2)
        self.assertEqual(game.card_fun(0, tanning), 5)
        self.assertEqual(game.card_fun(0, outdoors_after), 1)
        self.assertEqual(game.card_fun(0, non_outdoors_after), 2)


if __name__ == "__main__":
    unittest.main()
