import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class OptionalUnpackAI(RandomAI):
    def __init__(self, take_second: bool, rng) -> None:
        super().__init__(rng)
        self.take_second = take_second
        self.optional_actions = []

    def choose_optional_action(self, game, player_index, action):
        self.optional_actions.append(
            (action, tuple(card.instance_id for card in game.suitcase))
        )
        return self.take_second


class FancyCraftTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("fancy-craft")

        self.assertEqual(card.title, "Fancy Craft")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_first_unpack_gains_fun_and_optional_second_unpack_uses_refill(self) -> None:
        game = empty_game()
        ai = OptionalUnpackAI(True, game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fancy-craft"))
        first_suitcase = [make_card("biography") for _ in range(4)]
        game.suitcase = first_suitcase
        game.trunk = [make_card("fajitas") for _ in range(8)]

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 5)
        self.assertEqual(player.fun, 2)
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(ai.optional_actions[0][0], "unpack")
        self.assertNotEqual(
            set(ai.optional_actions[0][1]),
            {suitcase_card.instance_id for suitcase_card in first_suitcase},
        )
        self.assertEqual(len(game.discard), 8)
        self.assertEqual(len(game.suitcase), 4)

    def test_declining_second_unpack_still_keeps_first_result(self) -> None:
        game = empty_game()
        ai = OptionalUnpackAI(False, game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fancy-craft"))
        game.suitcase = [make_card("biography") for _ in range(4)]
        game.trunk = [make_card("fajitas") for _ in range(4)]

        game.play_card(0, 0)

        self.assertEqual(player.fun, 1)
        self.assertEqual(len(game.discard), 4)
        self.assertEqual(len(game.suitcase), 4)


if __name__ == "__main__":
    unittest.main()
