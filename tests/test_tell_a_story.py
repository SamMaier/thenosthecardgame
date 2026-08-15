import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class TellAStoryTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("tell-a-story"))

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Tell a Story")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, card), 2)

    def test_previous_event_adds_three_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        player.hand.extend([make_card("stay-up-late"), make_card("tell-a-story")])

        game.play_card(0, 0)
        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 5)

    def test_event_played_after_does_not_add_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 5
        player.hand.extend([make_card("tell-a-story"), make_card("work-call")])

        card = game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 2)

    def test_active_tomorrow_event_does_not_count_as_previous(self) -> None:
        game = empty_game()
        tomorrow_event = make_card("work-call")
        tomorrow_event.is_tomorrow = True
        game.players[0].tomorrow_cards.append(tomorrow_event)

        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("tell-a-story"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 2)


if __name__ == "__main__":
    unittest.main()
