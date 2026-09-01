import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class TeachKidToSkiTests(unittest.TestCase):
    def test_tomorrow_exercise_bonus(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("teach-kid-to-ski"))

        teach = game.play_card(0, 0)

        self.assertEqual(teach.definition.cost, 2)
        self.assertEqual(teach.definition.base_fun, 0)
        self.assertEqual(teach.definition.tags, frozenset({"Event", "Outdoors"}))
        self.assertEqual(player.energy, 5)
        self.assertEqual(game.card_fun(0, teach), 0)

        game.end_day()
        self.assertEqual(player.tomorrow_cards, [teach])
        self.assertTrue(teach.is_tomorrow)

        game.start_day()
        player.hand.extend([make_card("waterski"), make_card("biography")])
        exercise = game.play_card(0, 0)
        other = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, exercise), 7)
        self.assertEqual(game.card_fun(0, other), 2)

        game.end_day()
        self.assertEqual(player.tomorrow_cards, [])
        self.assertEqual(game.card_fun(0, exercise), 6)


if __name__ == "__main__":
    unittest.main()
