import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class NewfangledTresFuteTests(unittest.TestCase):
    def test_cost_tags_and_printed_fun(self) -> None:
        card = make_card("newfangled-tres-fute")

        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))
        self.assertEqual(card.definition.base_fun, 1)

    def test_scores_for_unique_tags_including_its_own_and_counts_each_tag_once(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("newfangled-tres-fute"),
                make_card("nap"),
                make_card("m-ms"),
                make_card("nap"),
            ]
        )

        newfangled = game.play_card(0, 0)
        nap = game.play_card(0, 0)
        m_and_ms = game.play_card(0, 0)
        duplicate_relax = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, newfangled), 5)
        self.assertEqual(game.card_fun(0, nap), 0)
        self.assertEqual(game.card_fun(0, m_and_ms), 0)
        self.assertEqual(game.card_fun(0, duplicate_relax), 0)

        game.end_day()

        self.assertEqual(player.fun, 5)


if __name__ == "__main__":
    unittest.main()
