import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class FishingMorningTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("fishing-morning")

        self.assertEqual(card.title, "Fishing Morning")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 5)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Outdoors"}))

    def test_must_be_the_first_card_played_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("biography"), make_card("fishing-morning")])

        game.play_card(0, 0)

        self.assertNotIn(0, game.playable_hand_indices(0))
        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)

    def test_scores_five_fun_when_played_first(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fishing-morning"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, card), 5)
        game.end_day()
        self.assertEqual(player.fun, 5)


if __name__ == "__main__":
    unittest.main()
