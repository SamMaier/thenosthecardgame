import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SevenWondersTests(unittest.TestCase):
    def test_cost_tags_and_fun(self) -> None:
        card = make_card("7-wonders")

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))

    def test_bonus_requires_strictly_more_board_games_than_both_neighbours(
        self,
    ) -> None:
        # Seat 0 has 1 Board Game; neighbours are seat 3 and seat 1 (both 0); seat 2 is opposite.
        game = empty_game()
        p0 = game.players[0]
        p0.energy = 7
        p0.hand.append(make_card("7-wonders"))
        game.players[2].played_today.extend(
            [make_card("tres-fute"), make_card("splendor")]
        )

        card = game.play_card(0, 0)
        self.assertEqual(p0.energy, 5)
        self.assertEqual(game.card_fun(0, card), 5)

        # Left neighbour (seat 3) ties -> bonus lost.
        game.players[3].played_today.append(make_card("splendor"))
        self.assertEqual(game.card_fun(0, card), 1)

        # Playing another Board Game beats seat 3 again -> 5.
        p0.hand.append(make_card("tres-fute"))
        game.play_card(0, 0)
        self.assertEqual(game.card_fun(0, card), 5)

        # Right neighbour (seat 1) ties via Colouring? No, Colouring grants Relax.
        # Seat 1 plays two Board Games -> ties seat 0's 2 -> bonus lost.
        game.players[1].played_today.extend(
            [make_card("tres-fute"), make_card("fit-to-print")]
        )
        self.assertEqual(game.card_fun(0, card), 1)


if __name__ == "__main__":
    unittest.main()
