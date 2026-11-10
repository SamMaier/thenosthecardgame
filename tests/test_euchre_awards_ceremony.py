import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstSuitcaseAI(RandomAI):
    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class EuchreAwardsCeremonyTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("euchre-awards-ceremony")

        self.assertEqual(card.title, "Euchre Awards Ceremony")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Social", "Outdoors"}),
        )

    def test_food_and_board_game_cards_played_after_pick_one_card_from_suitcase(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseAI(game.rng)
        player = game.players[0]
        player.energy = 8
        earlier = make_card("m-ms")
        ceremony = make_card("euchre-awards-ceremony")
        after_food = make_card("m-ms")
        after_relax = make_card("chalk-art")
        after_board_game = make_card("tres-fute")
        player.hand.extend([earlier, ceremony, after_food, after_relax, after_board_game])

        first_pick = make_card("waterski")
        second_pick = make_card("fajitas")
        replacement = make_card("cheap-white")
        game.suitcase = [
            first_pick,
            make_card("solo"),
            make_card("nap"),
            make_card("wingspan"),
        ]
        game.trunk = [replacement, second_pick]

        game.play_card(0, 0)
        self.assertEqual(sum(player.picked_cards.values()), 0)

        card = game.play_card(0, 0)
        self.assertEqual(sum(player.picked_cards.values()), 0)
        self.assertEqual(game.card_fun(0, card), 0)

        game.play_card(0, 0)
        self.assertEqual(sum(player.picked_cards.values()), 1)
        self.assertIn(first_pick, player.hand)

        game.play_card(0, 0)
        self.assertEqual(sum(player.picked_cards.values()), 1)

        game.play_card(0, 0)
        self.assertEqual(sum(player.picked_cards.values()), 2)
        self.assertIn(second_pick, player.hand)
        self.assertIs(game.suitcase[0], replacement)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 2)

    def test_card_with_both_food_and_board_game_tags_picks_only_once(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseAI(game.rng)
        player = game.players[0]
        player.energy = 8
        player.hand.extend(
            [
                make_card("euchre-awards-ceremony"),
                make_card("colouring"),
                make_card("nap"),
            ]
        )
        game.suitcase = [make_card("solo") for _ in range(4)]
        game.trunk = [make_card("wingspan") for _ in range(4)]

        game.play_card(0, 0)
        game.play_card(0, 0)
        self.assertEqual(sum(player.picked_cards.values()), 0)

        game.play_card(0, 0)
        self.assertEqual(sum(player.picked_cards.values()), 1)


if __name__ == "__main__":
    unittest.main()
