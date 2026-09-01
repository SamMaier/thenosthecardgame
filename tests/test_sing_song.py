import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SingSongTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("sing-song")

        self.assertEqual(card.title, "Sing Song")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 4)
        self.assertEqual(card.definition.tags, frozenset({"Event"}))

    def test_costs_one_less_for_each_opponent_with_an_event_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("sing-song"))
        game.players[1].played_today.append(make_card("work-call"))
        game.players[2].played_today.append(make_card("photo-shoot"))
        game.players[3].played_today.append(make_card("biography"))

        self.assertEqual(game.energy_cost(0, player.hand[0]), 2)
        sing_song = game.play_card(0, 0)
        self.assertEqual(player.energy, 5)
        self.assertEqual(game.card_fun(0, sing_song), 4)

    def test_each_opponent_reduces_cost_at_most_once(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("sing-song"))
        game.players[1].played_today.extend(
            [make_card("work-call"), make_card("photo-shoot")]
        )
        self.assertEqual(game.energy_cost(0, player.hand[0]), 3)


if __name__ == "__main__":
    unittest.main()
