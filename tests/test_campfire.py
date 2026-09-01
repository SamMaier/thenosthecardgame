import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CampfireTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("campfire")

        self.assertEqual(card.title, "Campfire")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Social", "Event", "Outdoors"}),
        )

    def test_scores_two_fun_for_each_active_tomorrow_card(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        tomorrow_cards = [make_card("biography"), make_card("biography")]
        for tomorrow_card in tomorrow_cards:
            tomorrow_card.is_tomorrow = True
        player.tomorrow_cards.extend(tomorrow_cards)
        player.hand.append(make_card("campfire"))

        campfire = game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, campfire), 4)

    def test_does_not_count_cards_played_today_or_without_tomorrow_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        player.played_today.append(make_card("biography"))
        player.hand.append(make_card("campfire"))

        campfire = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, campfire), 0)


if __name__ == "__main__":
    unittest.main()
