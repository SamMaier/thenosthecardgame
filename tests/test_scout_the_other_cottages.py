import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ScoutTheOtherCottagesTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("scout-the-other-cottages")

        self.assertEqual(card.title, "Scout the Other Cottages")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Social", "Indoors"}),
        )

    def test_marks_all_current_suitcase_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("scout-the-other-cottages"))
        game.suitcase = [
            make_card("biography"),
            make_card("fajitas"),
            make_card("waterski"),
            make_card("nap"),
        ]

        scout = game.play_card(0, 0)
        owner_marker = f"_scout_energy_cube_{scout.instance_id}"

        self.assertEqual(player.energy, 6)
        self.assertTrue(all(card.markers["energy_cube"] for card in game.suitcase))
        self.assertTrue(
            all(card.markers[owner_marker] for card in game.suitcase)
        )
        self.assertNotIn("energy_cube", scout.markers)

    def test_only_marked_cards_remaining_are_acquired_at_end_of_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("scout-the-other-cottages"))
        first = make_card("biography")
        taken = make_card("fajitas")
        third = make_card("waterski")
        fourth = make_card("nap")
        replacements = [
            make_card("solo"),
            make_card("fajitas"),
            make_card("biography"),
            make_card("waterski"),
        ]
        game.suitcase = [first, taken, third, fourth]
        game.trunk = list(replacements)

        game.play_card(0, 0)
        game.pick_suitcase_cards(0, (taken,))
        game.end_day()

        self.assertIn(first, player.hand)
        self.assertIn(third, player.hand)
        self.assertIn(fourth, player.hand)
        self.assertIn(taken, player.hand)
        self.assertIs(game.suitcase[0], replacements[2])
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(sum(player.picked_cards.values()), 1)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 1)
        self.assertEqual(sum(player.acquired_cards.values()), 4)

    def test_wedding_copy_only_acquires_cards_marked_by_that_copy(self) -> None:
        game = empty_game()
        scout_player = game.players[0]
        wedding_player = game.players[1]
        scout_player.energy = 2
        wedding_player.energy = 2
        scout_player.hand.append(make_card("scout-the-other-cottages"))
        wedding_player.hand.append(make_card("wedding-anniversary"))
        originals = [
            make_card("biography"),
            make_card("fajitas"),
            make_card("waterski"),
            make_card("nap"),
        ]
        refills = [
            make_card("splendor"),
            make_card("azul"),
            make_card("risk"),
            make_card("kneeboard"),
            make_card("chalk-art"),
        ]
        game.suitcase = list(originals)
        game.trunk = list(refills)

        game.play_card(0, 0)
        game.pick_suitcase_cards(2, (originals[0],))
        wedding_only_target = refills[-1]
        game.play_card(1, 0)

        game.end_day()

        self.assertNotIn(wedding_only_target, scout_player.hand)
        self.assertIn(wedding_only_target, wedding_player.hand)
        self.assertTrue(all(card in scout_player.hand for card in originals[1:]))

    def test_penalizes_only_cards_played_before_scout(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("biography"),
                make_card("scout-the-other-cottages"),
                make_card("biography"),
            ]
        )
        game.suitcase = [make_card("fajitas") for _ in range(4)]
        game.trunk = [make_card("biography") for _ in range(4)]

        previous = game.play_card(0, 0)
        scout = game.play_card(0, 0)
        later = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, previous), 1)
        self.assertEqual(game.card_fun(0, scout), 0)
        self.assertEqual(game.card_fun(0, later), 2)

        game.end_day()

        self.assertEqual(player.fun, 3)

    def test_acquires_marked_cards_only_after_every_player_scores(self) -> None:
        game = empty_game()
        scout_player = game.players[0]
        hand_size_player = game.players[1]
        scout = make_card("scout-the-other-cottages")
        hand_size_card = make_card("play-with-the-kids")
        scout_player.played_today.append(scout)
        hand_size_player.played_today.append(hand_size_card)
        hand_size_player.hand.append(make_card("biography"))
        game.suitcase = [make_card("fajitas") for _ in range(4)]
        game.trunk = [make_card("biography") for _ in range(4)]
        scout.effective_behavior.on_play(game, scout_player, scout)

        game.end_day()

        self.assertEqual(hand_size_player.fun, 6)
        self.assertEqual(len(scout_player.hand), 4)


if __name__ == "__main__":
    unittest.main()
