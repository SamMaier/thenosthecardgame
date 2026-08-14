import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BublyTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("bubly")

        self.assertEqual(card.title, "Bubly")
        self.assertEqual(card.definition.cost, 0)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_scores_bonus_after_a_previous_card_gives_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 0
        player.hand.extend([make_card("m-ms"), make_card("bubly")])

        game.play_card(0, 0)
        bubly = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, bubly), 2)

    def test_does_not_score_bonus_without_a_previous_energy_gain(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 1
        player.hand.extend([make_card("biography"), make_card("bubly")])

        game.play_card(0, 0)
        bubly = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, bubly), 0)

    def test_later_energy_gain_does_not_count(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 0
        player.hand.extend([make_card("bubly"), make_card("m-ms")])

        bubly = game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, bubly), 0)


if __name__ == "__main__":
    unittest.main()
