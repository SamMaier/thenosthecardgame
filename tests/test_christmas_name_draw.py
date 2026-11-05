import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetCardAI(RandomAI):
    def __init__(self, target, rng):
        super().__init__(rng)
        self.target = target

    def choose_card_target(self, game, player_index, eligible_cards):
        return eligible_cards.index(self.target)


class ChristmasNameDrawTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("christmas-name-draw")

        self.assertEqual(card.title, "Christmas Name Draw")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))

    def test_targets_an_opponent_card_and_adds_two_fun_to_matching_cards_after(self) -> None:
        game = empty_game()
        target = make_card("waterski")
        game.players[1].played_today.append(target)

        game.ais[0] = TargetCardAI(target, random.Random(0))
        player = game.players[0]
        player.energy = 15
        player.hand.extend(
            [
                make_card("kneeboard"),
                make_card("christmas-name-draw"),
                make_card("waterski"),
                make_card("biography"),
            ]
        )

        before_matching = game.play_card(0, 0)
        card = game.play_card(0, 0)
        after_matching = game.play_card(0, 0)
        after_non_matching = game.play_card(0, 0)

        self.assertTrue(target.markers["energy_cube"])
        self.assertIs(card.markers["target_card"], target)
        self.assertEqual(game.card_fun(0, before_matching), 5)
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, after_matching), 8)
        self.assertEqual(game.card_fun(0, after_non_matching), 2)

        game.end_day()

        self.assertEqual(player.fun, 15)

    def test_active_tomorrow_cards_are_not_eligible_targets(self) -> None:
        game = empty_game()
        tomorrow_card = make_card("waterski")
        tomorrow_card.is_tomorrow = True
        game.players[1].tomorrow_cards.append(tomorrow_card)

        player = game.players[0]
        player.energy = 4
        player.hand.append(make_card("christmas-name-draw"))

        card = game.play_card(0, 0)

        self.assertNotIn("target_card", card.markers)
        self.assertNotIn("energy_cube", tomorrow_card.markers)
        self.assertEqual(game.card_fun(0, card), 0)

    def test_trashed_target_no_longer_grants_tag_bonus(self) -> None:
        game = empty_game()
        target = make_card("waterski")
        game.players[1].played_today.append(target)

        game.ais[0] = TargetCardAI(target, random.Random(0))
        player = game.players[0]
        player.energy = 10
        player.hand.extend([make_card("christmas-name-draw"), make_card("waterski")])

        game.play_card(0, 0)
        after_matching = game.play_card(0, 0)
        self.assertEqual(game.card_fun(0, after_matching), 8)

        game.players[1].played_today.remove(target)
        game.discard_card(target)

        self.assertEqual(game.card_fun(0, after_matching), 6)


if __name__ == "__main__":
    unittest.main()
