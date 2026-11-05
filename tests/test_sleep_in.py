import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SleepInTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("sleep-in")

        self.assertEqual(card.title, "Sleep In")
        self.assertEqual(card.definition.cost, 0)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Indoors"}))

    def test_must_be_first_card_and_gains_two_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("biography"), make_card("sleep-in")])

        game.play_card(0, 0)

        self.assertNotIn(0, game.playable_hand_indices(0))
        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)

        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("sleep-in"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 9)
        self.assertEqual(game.card_fun(0, card), 0)

    def test_blocks_energy_gains_from_other_cards_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        sleep_in = make_card("sleep-in")
        nap = make_card("nap")
        player.hand = [sleep_in, nap]

        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.energy, 9)
        self.assertNotIn("_gave_energy", nap.markers)

    def test_energy_gains_are_allowed_again_on_the_next_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("sleep-in"))

        game.play_card(0, 0)
        game.end_day()
        game.start_day()

        player.hand.append(make_card("nap"))
        game.play_card(0, 0)

        self.assertEqual(player.energy, 8)


if __name__ == "__main__":
    unittest.main()
