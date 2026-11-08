import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class QuacksChoiceAI(RandomAI):
    def __init__(self, decisions, rng) -> None:
        super().__init__(rng)
        self.decisions = list(decisions)

    def choose_optional_action(self, game, player_index, description: str) -> bool:
        return self.decisions.pop(0) if self.decisions else False


class QuacksTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("quacks")

        self.assertEqual(card.title, "Quacks")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))

    def test_reveals_trunk_cards_until_stopping_for_two_fun_each(self) -> None:
        game = empty_game()
        game.ais[0] = QuacksChoiceAI([True, True, False], game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("quacks"))
        first_reveal = make_card("fajitas")
        second_reveal = make_card("biography")
        unrevealed = make_card("solo")
        game.trunk = [unrevealed, second_reveal, first_reveal]

        quacks = game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(game.card_fun(0, quacks), 4)
        self.assertEqual(game.discard, [first_reveal, second_reveal])
        self.assertEqual(game.trunk, [unrevealed])

    def test_scores_zero_fun_and_stops_when_outdoors_card_is_revealed(self) -> None:
        game = empty_game()
        game.ais[0] = QuacksChoiceAI([True, True, True], game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("quacks"))
        first_reveal = make_card("fajitas")
        outdoors_reveal = make_card("waterski")
        unrevealed = make_card("biography")
        game.trunk = [unrevealed, outdoors_reveal, first_reveal]

        quacks = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, quacks), 0)
        self.assertEqual(game.discard, [first_reveal, outdoors_reveal])
        self.assertEqual(game.trunk, [unrevealed])


if __name__ == "__main__":
    unittest.main()
