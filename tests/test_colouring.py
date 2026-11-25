import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ColouringTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("colouring")

        self.assertEqual(card.title, "Colouring")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_cards_played_after_gain_relax_tag_and_no_energy_cube(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 15
        before = make_card("fajitas")
        colouring = make_card("colouring")
        after = make_card("tres-fute")
        player.hand.extend([before, colouring, after])

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertNotIn("energy_cube", before.markers)
        self.assertNotIn("target_card", colouring.markers)
        self.assertEqual(before.tags, frozenset({"Food"}))
        self.assertEqual(colouring.tags, frozenset({"Relax"}))
        self.assertEqual(after.tags, frozenset({"Board Game", "Relax"}))

    def test_gained_relax_tag_is_cleared_when_discarded_or_removed(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        colouring = make_card("colouring")
        after = make_card("fajitas")
        player.hand.extend([colouring, after])

        game.play_card(0, 0)
        game.play_card(0, 0)
        self.assertEqual(after.tags, frozenset({"Food", "Relax"}))

        player.played_today.remove(colouring)
        game.discard_card(colouring)
        game._sync_played_today_tags(player)
        self.assertEqual(after.tags, frozenset({"Food"}))

        player.played_today.append(colouring)
        game.end_day()
        self.assertNotIn(colouring, player.tomorrow_cards)
        self.assertIn(colouring, game.discard)
        self.assertEqual(after.tags, frozenset({"Food"}))

    def test_hand_cost_modifiers_and_fast_sim_scoring_parity(self) -> None:
        from thenos.ais.carbon import _carbon_copy_game
        from thenos.ais.hydrogen import _hydrogen_copy_game

        game = empty_game()
        player = game.players[0]
        player.energy = 15
        slc = make_card("sweet-lawn-chair")
        colouring = make_card("colouring")
        coffee = make_card("morning-coffee")
        tres = make_card("tres-fute")
        player.hand.extend([slc, colouring, coffee, tres])

        # Sweet Lawn Chair (+2 to Exercise/Relax after) -> Colouring -> Tres Fute
        game.play_card(0, 0)
        game.play_card(0, 0)
        # Unplayed hand card does not gain Relax yet for hand cost checks
        self.assertEqual(game.energy_cost(0, coffee), 1)
        game.play_card(0, 1)  # Tres Fute gains Relax from Colouring and +2 from SLC!

        exact_total = sum(game.card_fun(0, c) for c in player.played_today)
        self.assertEqual(game.card_fun(0, colouring), 2)
        self.assertEqual(game.card_fun(0, tres), 3)

        hsim = _hydrogen_copy_game(game)
        self.assertEqual(hsim.score_played_today_fast(0)[0], exact_total)

        csim = _carbon_copy_game(game)
        csim.end_day()
        self.assertEqual(csim.players[0].fun, exact_total)

    def test_istanbul_and_hydrogen_zone_ctx_handle_relax_extra_tag(self) -> None:
        from thenos.ais.hydrogen import (
            _build_quick_zone_ctx,
            _evaluate_istanbul_discards_fast,
            _tags_to_mask,
        )

        game = empty_game()
        player = game.players[0]
        player.energy = 15
        colouring = make_card("colouring")
        istanbul = make_card("istanbul")
        fajitas = make_card("fajitas")
        walk = make_card("morning-walk")
        player.hand.extend([colouring, istanbul, fajitas, walk])

        game.play_card(0, 0)
        player.hand.remove(istanbul)
        player.played_today.append(istanbul)
        game._sync_played_today_tags(player)

        _evaluate_istanbul_discards_fast(game, 0, tuple(player.hand))
        self.assertNotIn("_extra_tags", fajitas.markers)
        self.assertNotIn("_extra_tags", walk.markers)

        ctx = _build_quick_zone_ctx(player)
        relax_mask = _tags_to_mask(frozenset({"Relax"}))[0]
        self.assertTrue(ctx.get_played_masks()[1] & relax_mask)


if __name__ == "__main__":
    unittest.main()
