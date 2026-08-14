import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class TreatStoreRunTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("treat-store-run")

        self.assertEqual(card.title, "Treat Store Run")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food", "Event"}))

    def test_picks_all_currently_visible_food_cards_not_their_replacements(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("treat-store-run"))

        first_food = make_card("fajitas")
        non_food = make_card("biography")
        second_food = make_card("cheap-white")
        other_non_food = make_card("waterski")
        replacement_food = make_card("m-ms")
        replacement_non_food = make_card("thriller-book")
        game.suitcase = [first_food, non_food, second_food, other_non_food]
        game.trunk = [replacement_food, replacement_non_food]

        game.play_card(0, 0)

        self.assertIn(first_food, player.hand)
        self.assertIn(second_food, player.hand)
        self.assertNotIn(replacement_food, player.hand)
        self.assertNotIn(non_food, player.hand)
        self.assertNotIn(other_non_food, player.hand)
        self.assertEqual(
            game.suitcase,
            [replacement_non_food, non_food, replacement_food, other_non_food],
        )
        self.assertEqual(sum(player.picked_cards.values()), 2)
        self.assertEqual(sum(player.acquired_cards.values()), 2)
        self.assertEqual(sum(game.stats.suitcase_offers.values()), 4)

    def test_does_nothing_when_no_food_is_visible(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("treat-store-run"))
        game.suitcase = [make_card("biography") for _ in range(4)]

        game.play_card(0, 0)

        self.assertEqual(len(player.hand), 0)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(sum(player.picked_cards.values()), 0)
        self.assertEqual(sum(game.stats.suitcase_offers.values()), 0)


if __name__ == "__main__":
    unittest.main()
