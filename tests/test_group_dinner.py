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

    def test_blocked_owner_does_not_mark_gave_energy_when_opponents_gain_energy(self) -> None:
        from thenos.ai import RandomAI

        class TrashSleepInAI(RandomAI):
            def choose_cards_to_discard(self, game, player_index, options):
                return [
                    idx
                    for idx, card in enumerate(options)
                    if card.title == "Sleep In"
                ]

        game = empty_game()
        game.ais[0] = TrashSleepInAI(game.rng)
        p0 = game.players[0]
        p0.energy = 7
        p0.hand.extend(
            [
                make_card("sleep-in"),
                make_card("group-dinner"),
                make_card("bohnanza"),
                make_card("bubly"),
            ]
        )

        game.play_card(0, 0)  # Sleep In (+2 -> 9)
        group_dinner = game.play_card(0, 0)  # cost 3 -> 6; Sleep In blocks +1 for p0
        game.play_card(0, 0)  # Bohnanza trashes Sleep In (cost 2 -> 4)
        bubly = game.play_card(0, 0)  # cost 0

        self.assertEqual(p0.energy, 4)
        self.assertEqual([p.energy for p in game.players[1:]], [1, 1, 1])
        self.assertNotIn("_gave_energy", group_dinner.markers)
        self.assertEqual(game.card_fun(0, bubly), 0)


if __name__ == "__main__":
    unittest.main()
