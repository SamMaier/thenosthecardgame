import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class GroupDinnerTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("group-dinner")

        self.assertEqual(card.title, "Group Dinner")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 4)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Food", "Social", "Outdoors"}),
        )

    def test_all_players_gain_one_energy_after_payment_and_card_scores_four_fun(
        self,
    ) -> None:
        game = empty_game()
        for index, player in enumerate(game.players):
            player.energy = index + 3
        game.players[0].hand.append(make_card("group-dinner"))

        card = game.play_card(0, 0)

        self.assertEqual(
            [player.energy for player in game.players],
            [1, 5, 6, 7],
        )
        self.assertEqual(game.card_fun(0, card), 4)

    def test_cannot_play_after_any_player_has_gone_to_bed(self) -> None:
        game = empty_game()
        game.players[0].energy = 3
        game.players[2].asleep = True
        game.players[0].hand.append(make_card("group-dinner"))

        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)

        self.assertEqual(game.players[0].energy, 3)
        self.assertEqual(len(game.players[0].hand), 1)
        self.assertFalse(game.players[0].played_today)


if __name__ == "__main__":
    unittest.main()
