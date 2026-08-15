import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class EarlyBedtimeTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("early-bedtime")

        self.assertEqual(card.title, "Early Bedtime")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_cannot_be_played_as_fourth_card(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.played_today.extend(
            [make_card("biography"), make_card("biography"), make_card("biography")]
        )
        player.hand.append(make_card("early-bedtime"))

        self.assertNotIn(0, game.playable_hand_indices(0))
        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)

    def test_third_card_is_legal_and_starts_next_day_with_three_more_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.played_today.extend([make_card("biography"), make_card("biography")])
        player.hand.append(make_card("early-bedtime"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(game.card_fun(0, card), 0)

        game.end_day()

        self.assertEqual(player.tomorrow_cards, [card])
        self.assertTrue(card.is_tomorrow)

        game.start_day()

        self.assertEqual(player.energy, 10)


if __name__ == "__main__":
    unittest.main()
