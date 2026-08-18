import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class EuchreTournamentTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("euchre-tournament"))

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Euchre Tournament")
        self.assertEqual(card.definition.cost, 7)
        self.assertEqual(card.definition.base_fun, 5)
        self.assertEqual(
            card.definition.tags, frozenset({"Board Game", "Indoors"})
        )
        self.assertEqual(game.card_fun(0, card), 5)
        self.assertEqual(player.energy, 0)

    def test_picks_current_item_and_food_cards_only(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("euchre-tournament"))

        item = make_card("nos-shirt")
        other = make_card("biography")
        food = make_card("fajitas")
        other_item = make_card("waterski")
        replacement_item = make_card("nos-book")
        replacement_food = make_card("cheap-white")
        game.suitcase = [item, other, food, other_item]
        game.trunk = [replacement_food, replacement_item]

        game.play_card(0, 0)

        self.assertIn(item, player.hand)
        self.assertIn(food, player.hand)
        self.assertNotIn(other, player.hand)
        self.assertNotIn(other_item, player.hand)
        self.assertNotIn(replacement_item, player.hand)
        self.assertNotIn(replacement_food, player.hand)
        self.assertEqual(
            game.suitcase,
            [replacement_item, other, replacement_food, other_item],
        )
        self.assertEqual(sum(player.picked_cards.values()), 2)
        self.assertEqual(sum(player.acquired_cards.values()), 2)
        self.assertEqual(game.stats.suitcase_picks["Nos Shirt"], 1)
        self.assertEqual(game.stats.suitcase_picks["Fajitas"], 1)
        self.assertEqual(sum(game.stats.suitcase_offers.values()), 4)

    def test_does_nothing_when_no_item_or_food_is_visible(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("euchre-tournament"))
        game.suitcase = [make_card("biography") for _ in range(4)]

        game.play_card(0, 0)

        self.assertEqual(len(player.hand), 0)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 0)
        self.assertEqual(sum(game.stats.suitcase_offers.values()), 0)


if __name__ == "__main__":
    unittest.main()
