import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SingSongTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("sing-song")

        self.assertEqual(card.title, "Sing Song")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Event"}))

    def test_scores_one_fun_for_each_unique_tag_played_by_end_of_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("sing-song"),
                make_card("waterski"),
                make_card("biography"),
            ]
        )

        sing_song = game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        # Event, Exercise, Outdoors, and Relax are four distinct tags.
        self.assertEqual(game.card_fun(0, sing_song), 5)

        game.end_day()

        self.assertEqual(player.fun, 13)

    def test_duplicate_tags_count_once_and_active_tomorrow_tags_do_not_count(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.append(make_card("sing-song"))

        tomorrow_card = make_card("waterski")
        tomorrow_card.is_tomorrow = True
        player.tomorrow_cards.append(tomorrow_card)

        sing_song = game.play_card(0, 0)

        # Sing Song's Event tag is the only tag played today; the duplicate
        # Outdoors/Exercise tags on the active Tomorrow card are excluded.
        self.assertEqual(game.card_fun(0, sing_song), 2)


if __name__ == "__main__":
    unittest.main()
