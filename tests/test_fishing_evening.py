import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class FishingEveningTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("fishing-evening")

        self.assertEqual(card.title, "Fishing Evening")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 5)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Outdoors"}))

    def test_cannot_be_played_before_the_fourth_card(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.played_today.extend([make_card("biography"), make_card("biography")])
        player.hand.append(make_card("fishing-evening"))

        self.assertNotIn(0, game.playable_hand_indices(0))
        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)

    def test_is_playable_as_the_fourth_card_and_scores_five_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("early-bedtime"),
                make_card("early-bedtime"),
                make_card("early-bedtime"),
                make_card("fishing-evening"),
            ]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertIn(0, game.playable_hand_indices(0))
        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 1)
        self.assertEqual(game.card_fun(0, card), 5)
        game.end_day()
        self.assertEqual(player.fun, 5)


if __name__ == "__main__":
    unittest.main()
