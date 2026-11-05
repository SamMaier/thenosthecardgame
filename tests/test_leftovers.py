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


class LeftoversTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("leftovers")

        self.assertEqual(card.title, "Leftovers")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_copies_an_earlier_food_effect_without_paying_its_cost(self) -> None:
        game = empty_game()
        ai = TargetCopyAI("Fajitas", game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        target = make_card("fajitas")
        player.hand.extend([target, make_card("leftovers")])

        game.play_card(0, 0)
        leftovers = game.play_card(0, 0)

        self.assertEqual(ai.eligible_titles, ("Fajitas",))
        self.assertEqual(player.energy, 10)
        self.assertIs(leftovers.effective_behavior, target.definition.behavior)
        self.assertEqual(leftovers.effective_cost, 2)
        self.assertEqual(leftovers.tags, frozenset({"Food"}))
        self.assertTrue(target.markers["energy_cube"])

    def test_copies_base_fun_and_effect_but_not_tags(self) -> None:
        game = empty_game()
        ai = TargetCopyAI("Charcuterie", game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        target = make_card("charcuterie")
        player.hand.extend([target, make_card("leftovers")])

        game.play_card(0, 0)
        leftovers = game.play_card(0, 0)

        self.assertEqual(leftovers.effective_base_fun, 2)
        self.assertEqual(game.card_fun(0, leftovers), 2)
        self.assertEqual(leftovers.tags, frozenset({"Food"}))

    def test_only_earlier_food_cards_are_eligible(self) -> None:
        game = empty_game()
        ai = TargetCopyAI("Fajitas", game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("biography"),
                make_card("leftovers"),
                make_card("fajitas"),
            ]
        )

        game.play_card(0, 0)
        leftovers = game.play_card(0, 0)

        self.assertIsNone(ai.eligible_titles)
        self.assertEqual(player.energy, 4)
        self.assertIs(leftovers.effective_behavior, leftovers.definition.behavior)

    def test_restricted_food_card_that_is_not_legal_in_this_slot_is_excluded(self) -> None:
        game = empty_game()
        ai = TargetCopyAI("Morning Coffee", game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("morning-coffee"), make_card("leftovers")])

        game.play_card(0, 0)
        leftovers = game.play_card(0, 0)

        self.assertIsNone(ai.eligible_titles)
        self.assertIs(leftovers.effective_behavior, leftovers.definition.behavior)


if __name__ == "__main__":
    unittest.main()
