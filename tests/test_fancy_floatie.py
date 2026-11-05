import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class FancyFloatieTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("fancy-floatie")

        self.assertEqual(card.title, "Fancy Floatie")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Relax", "Item", "Outdoors"}),
        )

    def test_picks_all_relax_cards_including_replacements_until_none_remain(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fancy-floatie"))

        first_relax = make_card("biography")
        non_relax = make_card("fajitas")
        second_relax = make_card("nap")
        other_non_relax = make_card("waterski")
        replacement_relax = make_card("thriller-book")
        replacement_non_relax = make_card("cheap-white")
        final_non_relax = make_card("solo")
        game.suitcase = [first_relax, non_relax, second_relax, other_non_relax]
        game.trunk = [final_non_relax, replacement_relax, replacement_non_relax]

        game.play_card(0, 0)

        self.assertIn(first_relax, player.hand)
        self.assertIn(second_relax, player.hand)
        self.assertIn(replacement_relax, player.hand)
        self.assertNotIn(replacement_non_relax, player.hand)
        self.assertNotIn(final_non_relax, player.hand)
        self.assertNotIn(non_relax, player.hand)
        self.assertNotIn(other_non_relax, player.hand)
        self.assertEqual(
            game.suitcase,
            [replacement_non_relax, non_relax, final_non_relax, other_non_relax],
        )
        self.assertEqual(sum(player.picked_cards.values()), 3)
        self.assertEqual(sum(player.acquired_cards.values()), 3)
        self.assertEqual(sum(game.stats.suitcase_offers.values()), 8)

    def test_does_nothing_when_no_relax_card_is_visible(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fancy-floatie"))
        game.suitcase = [make_card("fajitas") for _ in range(4)]

        game.play_card(0, 0)

        self.assertEqual(len(player.hand), 0)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(sum(player.picked_cards.values()), 0)
        self.assertEqual(sum(game.stats.suitcase_offers.values()), 0)

    def test_tomorrow_relax_cards_score_one_more_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        player.hand.append(make_card("fancy-floatie"))
        game.suitcase = [make_card("fajitas") for _ in range(4)]

        floatie = game.play_card(0, 0)
        game.end_day()
        game.start_day()
        player.hand.extend([make_card("biography"), make_card("tres-fute")])
        relax = game.play_card(0, 0)
        board_game = game.play_card(0, 0)

        self.assertIn(floatie, player.tomorrow_cards)
        self.assertEqual(game.card_fun(0, relax), 3)
        self.assertEqual(game.card_fun(0, board_game), 1)


if __name__ == "__main__":
    unittest.main()
