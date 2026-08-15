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

    def test_first_slot_can_copy_a_must_be_first_card(self) -> None:
        game = empty_game()
        target = make_card("morning-coffee")
        game.ais[0] = TargetCopyAI(target.title, game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("wedding-anniversary"))
        game.players[1].played_today.append(target)

        card = game.play_card(0, 0)

        self.assertIs(card.effective_behavior, target.effective_behavior)
        self.assertEqual(player.energy, 9)

    def test_second_slot_can_copy_puerto_rico(self) -> None:
        game = empty_game()
        target = make_card("puerto-rico")
        game.ais[0] = TargetCopyAI(target.title, game.rng)
        player = game.players[0]
        player.energy = 7
        player.played_today.append(make_card("biography"))
        player.hand.append(make_card("wedding-anniversary"))
        game.players[1].played_today.append(target)
        game.suitcase = [make_card("fajitas") for _ in range(4)]

        card = game.play_card(0, 0)

        self.assertIs(card.effective_behavior, target.effective_behavior)
        self.assertEqual(player.energy, 3)

    def test_third_slot_can_copy_early_bedtime(self) -> None:
        game = empty_game()
        target = make_card("early-bedtime")
        game.ais[0] = TargetCopyAI(target.title, game.rng)
        player = game.players[0]
        player.energy = 7
        player.played_today.extend(
            [make_card("biography"), make_card("biography")]
        )
        player.hand.append(make_card("wedding-anniversary"))
        game.players[1].played_today.append(target)

        card = game.play_card(0, 0)

        self.assertIs(card.effective_behavior, target.effective_behavior)
        self.assertEqual(player.energy, 6)

    def test_copying_chuck_reverts_to_wedding_after_returning_to_hand(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        wedding = make_card("wedding-anniversary")
        player.hand.append(wedding)
        game.players[1].played_today.append(make_card("chuck-a-frisbee"))

        game.play_card(0, 0)
        game.end_day()

        self.assertIn(wedding, player.hand)
        self.assertIs(wedding.effective_behavior, wedding.definition.behavior)
        self.assertEqual(wedding.effective_cost, 0)

        player.energy = 7
        target = make_card("fajitas")
        game.players[1].played_today.append(target)
        replayed = game.play_card(0, player.hand.index(wedding))

        self.assertIs(replayed.effective_behavior, target.effective_behavior)
        self.assertEqual(player.energy, 8)

    def test_copying_leftovers_resolves_its_nested_food_copy(self) -> None:
        game = empty_game()
        game.ais[0] = TargetCopyAI("Leftovers", game.rng)
        player = game.players[0]
        player.energy = 10
        food = make_card("fajitas")
        player.played_today.append(food)
        player.hand.append(make_card("wedding-anniversary"))
        game.players[1].played_today.append(make_card("leftovers"))

        wedding = game.play_card(0, 0)

        self.assertIs(wedding.effective_behavior, food.effective_behavior)
        self.assertEqual(wedding.effective_cost, 2)
        self.assertEqual(player.energy, 12)
        self.assertTrue(food.markers["energy_cube"])

    def test_copying_last_years_shorts_can_copy_a_different_item(self) -> None:
        game = empty_game()
        game.ais[0] = TargetCopyAI("Last Year's Shorts", game.rng)
        player = game.players[0]
        player.energy = 10
        item = make_card("booby-prize")
        player.played_today.append(item)
        player.hand.append(make_card("wedding-anniversary"))
        source = make_card("last-years-shorts")
        game.players[1].played_today.append(source)
        drawn = make_card("biography")
        game.trunk.append(drawn)

        wedding = game.play_card(0, 0)

        self.assertIs(wedding.effective_behavior, item.effective_behavior)
        self.assertEqual(wedding.effective_cost, 3)
        self.assertIn(drawn, player.hand)
        self.assertTrue(item.markers["energy_cube"])
        self.assertTrue(source.markers["energy_cube"])


if __name__ == "__main__":
    unittest.main()
