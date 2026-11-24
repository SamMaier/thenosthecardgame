import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstSuitcaseAI(RandomAI):
    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class DatesFirstNosTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("dates-first-nos")

        self.assertEqual(card.title, "Date's First Nos")
        self.assertEqual(card.definition.cost, 7)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))

    def test_does_not_pick_after_later_plays_today(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseAI(game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("dates-first-nos"), make_card("nap"), make_card("nap")]
        )
        first_pick = make_card("biography")
        second_pick = make_card("waterski")
        game.suitcase = [
            first_pick,
            make_card("fajitas"),
            make_card("solo"),
            make_card("wingspan"),
        ]
        game.trunk = [make_card("fajitas"), second_pick]

        date = game.play_card(0, 0)

        self.assertEqual(sum(player.picked_cards.values()), 0)

        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertNotIn(first_pick, player.hand)
        self.assertNotIn(second_pick, player.hand)
        self.assertEqual(sum(player.picked_cards.values()), 0)
        self.assertEqual(game.card_fun(0, date), 0)

    def test_discarded_when_scoring_less_than_three_fun_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend([make_card("dates-first-nos"), make_card("biography")])

        date = game.play_card(0, 0)
        game.play_card(0, 0)  # scores 2 Fun today (< 3)
        game.end_day()

        self.assertNotIn(date, player.tomorrow_cards)
        self.assertIn(date, game.discard)

    def test_keeps_tomorrow_effect_when_scoring_at_least_three_fun_today(
        self,
    ) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("dates-first-nos"),
                make_card("biography"),
                make_card("tres-fute"),
            ]
        )

        date = game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)  # 2 + 1 = 3 Fun today (>= 3)
        game.end_day()
        self.assertIn(date, player.tomorrow_cards)

        game.start_day()
        player.hand.append(make_card("biography"))
        played_tomorrow = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, date), 0)
        self.assertEqual(game.card_fun(0, played_tomorrow), 4)

    def test_counts_fit_to_print_and_old_fogey_toward_three_fun_today(
        self,
    ) -> None:
        from thenos.characters import CHARACTERS

        old_fogey = next(c for c in CHARACTERS if c.title == "Old Fogey")
        game = empty_game()
        game.characters = True
        player = game.players[0]
        player.character = old_fogey
        player.fun = 10
        game.start_day()  # Old Fogey +2 -> 12, fun_at_start_of_day == 10
        player.energy = 10
        player.hand.extend([make_card("dates-first-nos"), make_card("tres-fute")])

        date = game.play_card(0, 0)
        game.play_card(0, 0)  # 2 + 1 = 3 Fun today
        game.end_day()
        self.assertIn(date, player.tomorrow_cards)

        ftp_game = empty_game()
        p2 = ftp_game.players[0]
        p2.energy = 10
        p2.hand.extend([make_card("dates-first-nos"), make_card("fit-to-print")])
        date2 = ftp_game.play_card(0, 0)
        ftp_game.play_card(0, 0)  # +5 from Fit to Print at end of day
        ftp_game.end_day()
        self.assertIn(date2, p2.tomorrow_cards)


if __name__ == "__main__":
    unittest.main()
