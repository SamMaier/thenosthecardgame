import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SpaDayTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("spa-day")

        self.assertEqual(card.title, "Spa Day")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Relax", "Social"}),
        )

    def test_relax_cards_after_score_one_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("biography"),
                make_card("spa-day"),
                make_card("thriller-book"),
                make_card("cheap-white"),
            ]
        )

        relax_before = game.play_card(0, 0)
        spa_day = game.play_card(0, 0)
        relax_after = game.play_card(0, 0)
        non_relax_after = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, relax_before), 2)
        self.assertEqual(game.card_fun(0, spa_day), 2)
        self.assertEqual(game.card_fun(0, relax_after), 4)
        self.assertEqual(game.card_fun(0, non_relax_after), 3)


if __name__ == "__main__":
    unittest.main()
