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
        self.assertEqual(card.definition.cost, 6)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))

    def test_picks_after_each_later_play_but_not_when_played(self) -> None:
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

        self.assertIn(first_pick, player.hand)
        self.assertIn(second_pick, player.hand)
        self.assertEqual(sum(player.picked_cards.values()), 2)
        self.assertEqual(game.card_fun(0, date), 0)

    def test_tomorrow_every_card_played_scores_one_more_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("dates-first-nos"))

        date = game.play_card(0, 0)
        game.end_day()
        game.start_day()

        player.hand.append(make_card("biography"))
        played_tomorrow = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, date), 0)
        self.assertEqual(game.card_fun(0, played_tomorrow), 3)


if __name__ == "__main__":
    unittest.main()
