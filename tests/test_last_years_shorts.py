import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetCopyAI(RandomAI):
    def __init__(self, target_title: str, rng) -> None:
        super().__init__(rng)
        self.target_title = target_title
        self.eligible_titles = None

    def choose_card_to_copy(self, game, player_index, eligible_cards):
        self.eligible_titles = tuple(card.title for card in eligible_cards)
        for index, card in enumerate(eligible_cards):
            if card.title == self.target_title:
                return index
        return 0


class LastYearsShortsTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("last-years-shorts")

        self.assertEqual(card.title, "Last Year's Shorts")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

    def test_copies_any_players_active_item_without_its_cost_or_tags(self) -> None:
        game = empty_game()
        ai = TargetCopyAI("Nos Shirt", random.Random(0))
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7

        own_item = make_card("bug-spray")
        opponent_item = make_card("nos-shirt")
        non_item = make_card("fajitas")
        tomorrow_item = make_card("epic-playlist")
        tomorrow_item.is_tomorrow = True
        player.played_today.append(own_item)
        game.players[1].played_today.extend([opponent_item, non_item])
        game.players[1].tomorrow_cards.append(tomorrow_item)
        player.hand.extend([make_card("last-years-shorts"), make_card("work-call")])

        card = game.play_card(0, 0)

        self.assertEqual(ai.eligible_titles, ("Bug Spray", "Nos Shirt"))
        self.assertEqual(player.energy, 5)
        self.assertIs(card.effective_behavior, opponent_item.definition.behavior)
        self.assertEqual(card.effective_cost, 2)
        self.assertEqual(card.tags, frozenset({"Item"}))
        self.assertTrue(opponent_item.markers["energy_cube"])
        self.assertNotIn("energy_cube", tomorrow_item.markers)

        event = game.play_card(0, 0)
        self.assertEqual(game.card_fun(0, event), -2)

    def test_does_nothing_when_no_active_item_is_eligible(self) -> None:
        game = empty_game()
        ai = TargetCopyAI("Nos Shirt", game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        tomorrow_item = make_card("nos-shirt")
        tomorrow_item.is_tomorrow = True
        game.players[1].tomorrow_cards.append(tomorrow_item)
        player.hand.append(make_card("last-years-shorts"))

        card = game.play_card(0, 0)

        self.assertIsNone(ai.eligible_titles)
        self.assertEqual(player.energy, 5)
        self.assertIs(card.effective_behavior, card.definition.behavior)
        self.assertNotIn("energy_cube", tomorrow_item.markers)


if __name__ == "__main__":
    unittest.main()
