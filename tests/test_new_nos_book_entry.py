import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class NewNosBookEntryTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("new-nos-book-entry")

        self.assertEqual(card.title, "New Nos Book Entry")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Social", "Event"}),
        )

    def test_scores_only_base_fun_with_four_previous_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        player.played_today.extend(make_card("tres-fute") for _ in range(4))
        player.hand.append(make_card("new-nos-book-entry"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 1)

    def test_scores_bonus_with_five_previous_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        player.played_today.extend(make_card("tres-fute") for _ in range(5))
        player.hand.append(make_card("new-nos-book-entry"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 5)

    def test_active_tomorrow_cards_do_not_count_as_previous_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        player.tomorrow_cards.extend(make_card("tres-fute") for _ in range(5))
        player.hand.append(make_card("new-nos-book-entry"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 1)


if __name__ == "__main__":
    unittest.main()
