import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class ScrabbleAI(RandomAI):
    def __init__(self, discard_indices, rng) -> None:
        super().__init__(rng)
        self.discard_indices = tuple(discard_indices)

    def choose_cards_to_discard(self, game, player_index, hand):
        return self.discard_indices

    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class ScrabbleTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("scrabble")

        self.assertEqual(card.title, "Scrabble")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Board Game", "Outdoors"}),
        )

    def test_discards_selected_hand_cards_and_picks_the_same_number(self) -> None:
        game = empty_game()
        game.ais[0] = ScrabbleAI((0, 2), game.rng)
        player = game.players[0]
        player.energy = 7
        discarded_first = make_card("biography")
        retained = make_card("nap")
        discarded_second = make_card("waterski")
        discarded_second.markers["test"] = True
        player.hand.extend(
            [
                make_card("scrabble"),
                discarded_first,
                retained,
                discarded_second,
            ]
        )

        first_pick = make_card("fajitas")
        second_pick = make_card("biography")
        replacement_after_second_pick = make_card("nap")
        game.suitcase = [
            first_pick,
            make_card("solo"),
            make_card("carcassonne"),
            make_card("wingspan"),
        ]
        game.trunk = [
            replacement_after_second_pick,
            second_pick,
        ]

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(player.hand, [retained, first_pick, second_pick])
        self.assertEqual(
            {id(discarded) for discarded in game.discard},
            {id(discarded_first), id(discarded_second)},
        )
        self.assertEqual(discarded_second.markers, {})
        self.assertIn(first_pick, player.hand)
        self.assertIn(second_pick, player.hand)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(player.picked_cards["Fajitas"], 1)
        self.assertEqual(player.picked_cards["Biography"], 1)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 2)
        self.assertEqual(game.card_fun(0, card), 2)

    def test_can_discard_zero_cards(self) -> None:
        game = empty_game()
        game.ais[0] = ScrabbleAI((), game.rng)
        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("scrabble"))
        game.suitcase = [make_card("biography") for _ in range(4)]

        game.play_card(0, 0)

        self.assertEqual(player.hand, [])
        self.assertEqual(game.discard, [])
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 0)


if __name__ == "__main__":
    unittest.main()
