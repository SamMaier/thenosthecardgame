import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetCardAI(RandomAI):
    def __init__(self, target_title: str, rng) -> None:
        super().__init__(rng)
        self.target_title = target_title
        self.eligible_titles = None

    def choose_card_target(self, game, player_index, eligible_cards):
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

    def test_takes_any_card_from_the_discard_pile(self) -> None:
        game = empty_game()
        ai = TargetCardAI("Biography", random.Random(0))
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        discarded = [make_card("fajitas"), make_card("biography")]
        game.discard.extend(discarded)
        player.hand.append(make_card("last-years-shorts"))

        card = game.play_card(0, 0)

        self.assertEqual(ai.eligible_titles, ("Fajitas", "Biography"))
        self.assertEqual(player.energy, 5)
        self.assertIn(discarded[1], player.hand)
        self.assertEqual(game.discard, [discarded[0]])

    def test_does_nothing_when_discard_pile_is_empty(self) -> None:
        game = empty_game()
        ai = TargetCardAI("Nos Shirt", game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("last-years-shorts"))

        card = game.play_card(0, 0)

        self.assertIsNone(ai.eligible_titles)
        self.assertEqual(player.energy, 5)
        self.assertEqual(game.discard, [])


if __name__ == "__main__":
    unittest.main()
