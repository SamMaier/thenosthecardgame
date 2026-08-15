import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstPlayableAI(RandomAI):
    def choose_card_to_play(self, game, player_index, playable_hand_indices):
        return playable_hand_indices[0]


class DeclineExtraPlayAI(FirstPlayableAI):
    def __init__(self, rng) -> None:
        super().__init__(rng)
        self.events = []

    def choose_card_to_play(self, game, player_index, playable_hand_indices):
        self.events.append("normal")
        return super().choose_card_to_play(
            game, player_index, playable_hand_indices
        )

    def choose_extra_card_to_play(
        self, game, player_index, playable_hand_indices
    ):
        self.events.append("declined-extra")
        return None


class EpicDuelsTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("epic-duels")

        self.assertEqual(card.title, "Epic Duels")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Board Game", "Indoors"}),
        )

    def test_allows_additional_cards_in_the_same_turn(self) -> None:
        game = empty_game()
        game.ais[0] = FirstPlayableAI(game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("epic-duels"), make_card("biography"), make_card("biography")]
        )

        game.playing_phase()

        self.assertEqual(
            [card.title for card in player.played_today],
            ["Epic Duels", "Biography", "Biography"],
        )
        self.assertEqual(player.energy, 3)
        self.assertEqual(game.card_fun(0, player.played_today[0]), 2)

    def test_player_may_decline_additional_cards_this_turn(self) -> None:
        game = empty_game()
        ai = DeclineExtraPlayAI(game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("epic-duels"), make_card("biography")]
        )

        game.playing_phase()

        self.assertEqual(ai.events, ["normal", "declined-extra", "normal"])
        self.assertEqual(
            [card.title for card in player.played_today],
            ["Epic Duels", "Biography"],
        )


if __name__ == "__main__":
    unittest.main()
