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


class WeddingAnniversaryTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("wedding-anniversary")

        self.assertEqual(card.title, "Wedding Anniversary")
        self.assertEqual(card.definition.cost, 0)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Event"}))

    def test_copies_an_opponent_effect_and_pays_written_cost(self) -> None:
        game = empty_game()
        ai = TargetCopyAI("Wingspan", game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("wedding-anniversary"))
        target = make_card("wingspan")
        game.players[1].played_today.append(target)
        tomorrow_target = make_card("fajitas")
        tomorrow_target.is_tomorrow = True
        game.players[1].tomorrow_cards.append(tomorrow_target)

        card = game.play_card(0, 0)

        self.assertEqual(ai.eligible_titles, ("Wingspan",))
        self.assertEqual(player.energy, 4)
        self.assertIs(card.effective_behavior, target.definition.behavior)
        self.assertEqual(card.effective_base_fun, target.definition.base_fun)
        self.assertEqual(card.tags, frozenset({"Event"}))
        self.assertTrue(target.markers["energy_cube"])
        self.assertEqual(player.fun, 1)

    def test_does_not_copy_card_the_player_cannot_afford(self) -> None:
        game = empty_game()
        ai = TargetCopyAI("Fajitas", game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 2
        player.hand.append(make_card("wedding-anniversary"))
        target = make_card("fajitas")
        game.players[1].played_today.append(target)

        card = game.play_card(0, 0)

        self.assertEqual(ai.eligible_titles, None)
        self.assertEqual(player.energy, 2)
        self.assertIs(card.effective_behavior, card.definition.behavior)
        self.assertNotIn("energy_cube", target.markers)

    def test_does_not_copy_card_that_is_illegal_in_the_current_slot(self) -> None:
        game = empty_game()
        ai = TargetCopyAI("Puerto Rico", game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        player.played_today.extend([make_card("biography"), make_card("biography")])
        player.hand.append(make_card("wedding-anniversary"))
        target = make_card("puerto-rico")
        game.players[1].played_today.append(target)

        card = game.play_card(0, 0)

        self.assertEqual(ai.eligible_titles, None)
        self.assertEqual(player.energy, 7)
        self.assertIs(card.effective_behavior, card.definition.behavior)
        self.assertNotIn("energy_cube", target.markers)


if __name__ == "__main__":
    unittest.main()
