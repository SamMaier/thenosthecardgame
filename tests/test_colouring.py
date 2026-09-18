import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ColouringTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("colouring")

        self.assertEqual(card.title, "Colouring")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_cards_played_tomorrow_count_as_having_all_tags(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("colouring"))

        colouring = game.play_card(0, 0)
        game.end_day()
        game.start_day()

        played_tomorrow = make_card("biography")
        player.hand.append(played_tomorrow)
        game.play_card(0, 0)

        self.assertIn(colouring, player.tomorrow_cards)
        self.assertEqual(player.hand, [])
        self.assertIn("Food", played_tomorrow.tags)
        self.assertIn("Item", played_tomorrow.tags)
        self.assertIn("Indoors", played_tomorrow.tags)

    def test_tomorrow_all_tags_are_cleared_when_the_card_is_discarded(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("colouring"))

        colouring = game.play_card(0, 0)
        game.end_day()
        game.start_day()
        played_tomorrow = make_card("biography")
        player.hand.append(played_tomorrow)
        game.play_card(0, 0)
        game.end_day()

        self.assertNotIn(colouring, player.tomorrow_cards)
        self.assertIn(colouring, game.discard)
        self.assertEqual(played_tomorrow.tags, frozenset({"Relax"}))


if __name__ == "__main__":
    unittest.main()
