import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SwimmingTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("swimming")

        self.assertEqual(card.title, "Swimming")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 3)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_scores_six_fun_when_immediately_preceded_by_exercise(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend([make_card("waterski"), make_card("swimming")])

        game.play_card(0, 0)
        swimming = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, swimming), 6)

    def test_scores_three_fun_when_played_first(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("swimming"))

        swimming = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, swimming), 3)

    def test_scores_three_fun_when_non_exercise_played_between_exercise_and_swimming(
        self,
    ) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        player.hand.extend(
            [
                make_card("waterski"),
                make_card("fajitas"),
                make_card("swimming"),
            ]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        swimming = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, swimming), 3)


if __name__ == "__main__":
    unittest.main()
