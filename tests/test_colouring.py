import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetCardAI(RandomAI):
    def __init__(self, target, rng) -> None:
        super().__init__(rng)
        self.target = target

    def choose_card_target(self, game, player_index, eligible_cards):
        return eligible_cards.index(self.target)


class ColouringTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("colouring")

        self.assertEqual(card.title, "Colouring")
        self.assertEqual(card.definition.cost, 5)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_cards_played_after_gain_marked_card_tags(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 15
        unmarked_before = make_card("fajitas")
        marked_before = make_card("waterski")
        colouring = make_card("colouring")
        played_after = make_card("biography")
        player.hand.extend(
            [unmarked_before, marked_before, colouring, played_after]
        )
        game.ais[0] = TargetCardAI(marked_before, random.Random(0))

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertTrue(marked_before.markers.get("energy_cube"))
        self.assertIs(colouring.markers.get("target_card"), marked_before)
        self.assertEqual(unmarked_before.tags, frozenset({"Food"}))
        self.assertEqual(colouring.tags, frozenset({"Relax"}))
        self.assertEqual(
            played_after.tags,
            frozenset({"Relax", "Exercise", "Outdoors"}),
        )

    def test_playing_as_first_card_today_marks_nothing(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        colouring = make_card("colouring")
        played_after = make_card("biography")
        player.hand.extend([colouring, played_after])

        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertNotIn("target_card", colouring.markers)
        self.assertEqual(played_after.tags, frozenset({"Relax"}))

    def test_gained_tags_are_cleared_when_discarded(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        marked_before = make_card("fajitas")
        colouring = make_card("colouring")
        played_after = make_card("biography")
        player.hand.extend([marked_before, colouring, played_after])
        game.ais[0] = TargetCardAI(marked_before, random.Random(0))

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)
        game.end_day()

        self.assertNotIn(colouring, player.tomorrow_cards)
        self.assertIn(colouring, game.discard)
        self.assertEqual(played_after.tags, frozenset({"Relax"}))

    def test_hand_cost_modifiers_and_bohnanza_removal(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 15
        sunscreen = make_card("sunscreen")
        waterski = make_card("waterski")
        colouring = make_card("colouring")
        fajitas = make_card("fajitas")
        player.hand.extend([sunscreen, waterski, colouring, fajitas])
        game.ais[0] = TargetCardAI(waterski, random.Random(0))

        game.play_card(0, 0)
        game.play_card(0, 0)
        self.assertEqual(game.energy_cost(0, fajitas), 3)

        game.play_card(0, 0)
        # Fajitas (Food) in hand gains Outdoors from marked Waterski, so Sunscreen reduces its cost by 1.
        self.assertEqual(game.energy_cost(0, fajitas), 2)
        self.assertNotIn("_extra_tags", fajitas.markers)
        game.play_card(0, 0)
        self.assertEqual(
            fajitas.tags,
            frozenset({"Food", "Exercise", "Outdoors"}),
        )

        # If the marked card is removed from played_today (e.g. Bohnanza), Fajitas loses its gained tags.
        player.played_today.remove(waterski)
        game.discard_card(waterski)
        game._sync_played_today_tags(player)
        self.assertEqual(fajitas.tags, frozenset({"Food"}))

    def test_removing_colouring_clears_gained_tags(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        marked_before = make_card("fajitas")
        colouring = make_card("colouring")
        played_after = make_card("biography")
        player.hand.extend([marked_before, colouring, played_after])
        game.ais[0] = TargetCardAI(marked_before, random.Random(0))

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)
        self.assertEqual(played_after.tags, frozenset({"Relax", "Food"}))

        player.played_today.remove(colouring)
        game.discard_card(colouring)
        game._sync_played_today_tags(player)
        self.assertEqual(played_after.tags, frozenset({"Relax"}))

    def test_fast_sim_parity_for_energy_cost_and_scoring(self) -> None:
        from thenos.ais.hydrogen import _hydrogen_copy_game

        game = empty_game()
        player = game.players[0]
        player.energy = 15
        sunscreen = make_card("sunscreen")
        waterski = make_card("waterski")
        colouring = make_card("colouring")
        fajitas = make_card("fajitas")
        ice_wine = make_card("ice-wine")
        player.hand.extend([sunscreen, waterski, colouring, fajitas, ice_wine])
        game.ais[0] = TargetCardAI(waterski, random.Random(0))

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        hsim = _hydrogen_copy_game(game)

        self.assertEqual(game.energy_cost(0, fajitas), 2)
        self.assertEqual(hsim.energy_cost(0, hsim.players[0].hand[0]), 2)
        self.assertEqual(hsim.hand_playable_with_costs(0), [(0, 2), (1, 2)])

        game.play_card(0, 0)
        hsim.play_card(0, 0)
        # Play Cliff Climbing after Fajitas (which gained Outdoors from Waterski)
        cliff = make_card("cliff-climbing")
        player.hand.append(cliff)
        hsim.players[0].hand.append(make_card("cliff-climbing"))
        game.play_card(0, 1)
        hsim.play_card(0, 1)

        exact_total = sum(game.card_fun(0, c) for c in player.played_today)
        self.assertEqual(hsim.score_played_today_fast(0)[0], exact_total)

    def test_istanbul_and_hydrogen_zone_ctx_handle_extra_tags(self) -> None:
        from thenos.ais.hydrogen import (
            _build_quick_zone_ctx,
            _evaluate_istanbul_discards_fast,
            _tags_to_mask,
        )

        game = empty_game()
        player = game.players[0]
        player.energy = 15
        waterski = make_card("waterski")
        colouring = make_card("colouring")
        istanbul = make_card("istanbul")
        fajitas = make_card("fajitas")
        walk = make_card("morning-walk")
        player.hand.extend([waterski, colouring, istanbul, fajitas, walk])
        game.ais[0] = TargetCardAI(waterski, random.Random(0))

        game.play_card(0, 0)
        game.play_card(0, 0)
        player.hand.remove(istanbul)
        player.played_today.append(istanbul)
        game._sync_played_today_tags(player)

        _evaluate_istanbul_discards_fast(game, 0, tuple(player.hand))
        self.assertNotIn("_extra_tags", fajitas.markers)
        self.assertNotIn("_extra_tags", walk.markers)

        # Zone context counts _extra_tags on played_today cards (Istanbul gained Outdoors from Waterski).
        ctx = _build_quick_zone_ctx(player)
        outdoors_mask = _tags_to_mask(frozenset({"Outdoors"}))[0]
        self.assertTrue(ctx.get_played_masks()[2] & outdoors_mask)


if __name__ == "__main__":
    unittest.main()
