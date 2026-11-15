import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WurstchenTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("wurstchen")

        self.assertEqual(card.title, "Würstchen")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_gains_one_energy_per_earlier_card_played_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 5
        player.tomorrow_cards.append(make_card("morning-walk"))
        player.hand.extend(
            [
                make_card("biography"),
                make_card("doritos"),
                make_card("wurstchen"),
                make_card("fajitas"),
            ]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 2)
        self.assertEqual(len(player.hand), 1)
        self.assertTrue(card.markers["_gave_energy"])

    def test_gains_no_energy_when_played_first(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.extend([make_card("wurstchen"), make_card("biography")])

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertNotIn("_gave_energy", card.markers)

    def test_leftovers_copy_and_sleep_in_restriction(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("biography"),
                make_card("wurstchen"),
                make_card("leftovers"),
            ]
        )

        game.play_card(0, 0)  # cost 1 -> energy 6
        game.play_card(0, 0)  # cost 4, +1 energy -> energy 3
        leftovers = game.play_card(0, 0)  # cost 2, copies Wurstchen (+2 energy) -> energy 3
        self.assertEqual(player.energy, 3)
        self.assertTrue(leftovers.markers["_gave_energy"])

        blocked_game = empty_game()
        blocked_player = blocked_game.players[0]
        blocked_player.energy = 2
        blocked_player.hand.extend(
            [make_card("sleep-in"), make_card("wurstchen")]
        )

        blocked_game.play_card(0, 0)  # Sleep In (+2 energy -> 4, blocks later energy gain)
        second_wurstchen = blocked_game.play_card(0, 0)  # cost 4 -> energy 0
        self.assertEqual(blocked_player.energy, 0)
        self.assertNotIn("_gave_energy", second_wurstchen.markers)


if __name__ == "__main__":
    unittest.main()
