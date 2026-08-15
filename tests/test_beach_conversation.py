import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BeachConversationTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("beach-conversation")

        self.assertEqual(card.title, "Beach Conversation")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Social", "Outdoors"}),
        )

    def test_gains_two_energy_after_an_outdoors_card(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("campfire"), make_card("beach-conversation")]
        )

        game.play_card(0, 0)
        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(game.card_fun(0, card), 2)

    def test_only_the_immediately_previous_card_counts(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("campfire"),
                make_card("biography"),
                make_card("beach-conversation"),
            ]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.energy, 3)

    def test_active_tomorrow_card_does_not_count_as_last_played(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("evening-chat"))

        game.play_card(0, 0)
        game.end_day()
        game.start_day()

        player.hand.append(make_card("beach-conversation"))
        game.play_card(0, 0)

        self.assertEqual(player.energy, 4)


if __name__ == "__main__":
    unittest.main()
