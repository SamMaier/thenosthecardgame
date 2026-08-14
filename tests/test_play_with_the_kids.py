import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class PlayWithTheKidsTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("play-with-the-kids")

        self.assertEqual(card.title, "Play With the Kids")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 3)
        self.assertEqual(card.definition.tags, frozenset({"Exercise"}))

    def test_scores_bonus_when_hand_is_larger_than_every_opponent(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("play-with-the-kids"))
        card = game.play_card(0, 0)
        self.assertEqual(player.energy, 3)
        player.hand.extend(make_card("fajitas") for _ in range(3))

        game.players[1].hand.append(make_card("fajitas"))
        game.players[2].hand.extend(make_card("fajitas") for _ in range(2))

        self.assertEqual(game.card_fun(0, card), 6)
        game.end_day()

        self.assertEqual(player.fun, 6)

    def test_does_not_score_bonus_when_an_opponent_is_tied_or_ahead(self) -> None:
        for opponent_count in (3, 4):
            with self.subTest(opponent_count=opponent_count):
                game = empty_game()
                player = game.players[0]
                player.energy = 7
                player.hand.append(make_card("play-with-the-kids"))
                card = game.play_card(0, 0)
                player.hand.extend(make_card("fajitas") for _ in range(3))
                game.players[1].hand.extend(
                    make_card("fajitas") for _ in range(opponent_count)
                )

                self.assertEqual(game.card_fun(0, card), 3)


if __name__ == "__main__":
    unittest.main()
