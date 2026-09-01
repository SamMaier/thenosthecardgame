import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WildlifeSpottingTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("wildlife-spotting")
        self.assertEqual(card.title, "Wildlife Spotting")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_tomorrow_first_outdoors_card_scores_two_more(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 1
        player.hand.append(make_card("wildlife-spotting"))
        spotting = game.play_card(0, 0)
        game.end_day()
        game.start_day()
        player.hand.extend([make_card("dock-fishing"), make_card("canoe")])
        first = game.play_card(0, 0)
        second = game.play_card(0, 0)
        self.assertIn(spotting, player.tomorrow_cards)
        self.assertEqual(game.card_fun(0, first), 4)
        self.assertEqual(game.card_fun(0, second), 2)

    def test_no_bonus_if_first_card_is_not_outdoors(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        spotting = make_card("wildlife-spotting")
        spotting.is_tomorrow = True
        player.tomorrow_cards.append(spotting)
        player.hand.extend([make_card("biography"), make_card("dock-fishing")])
        first = game.play_card(0, 0)
        later_outdoors = game.play_card(0, 0)
        self.assertEqual(game.card_fun(0, first), 2)
        self.assertEqual(game.card_fun(0, later_outdoors), 2)


if __name__ == "__main__":
    unittest.main()
