import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SunriseTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("sunrise")

        self.assertEqual(card.title, "Sunrise")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 4)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_must_be_the_first_card_played_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("biography"), make_card("sunrise")])

        game.play_card(0, 0)

        self.assertNotIn(0, game.playable_hand_indices(0))
        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)

    def test_scores_four_fun_when_played_first(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("sunrise"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 5)
        self.assertEqual(game.card_fun(0, card), 4)
        game.end_day()
        self.assertEqual(player.fun, 4)


if __name__ == "__main__":
    unittest.main()
