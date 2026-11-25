import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class NosShirtTests(unittest.TestCase):
    def test_printed_values_and_scores_zero_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("nos-shirt"))

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Nos Shirt")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(player.energy, 6)
        self.assertEqual(game.card_fun(0, card), 0)

    def test_next_card_plus_two_left_to_right_visibility_and_child_rule(
        self,
    ) -> None:
        # Boosts only the immediately next visible card, never earlier or later ones.
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("tres-fute"),
                make_card("nos-shirt"),
                make_card("biography"),
                make_card("tres-fute"),
            ]
        )
        before = game.play_card(0, 0)
        shirt = game.play_card(0, 0)
        next_card = game.play_card(0, 0)
        later = game.play_card(0, 0)
        self.assertEqual(game.card_fun(0, before), 1)
        self.assertEqual(game.card_fun(0, shirt), 0)
        self.assertEqual(game.card_fun(0, next_card), 4)
        self.assertEqual(game.card_fun(0, later), 1)

        # Visibility: removing Nos Shirt stops its +2 bonus.
        player.played_today.remove(shirt)
        game.discard_card(shirt)
        self.assertEqual(game.card_fun(0, next_card), 2)

        # Left-to-right ordering with multipliers:
        # 1) Bring a Friend (*2) -> Nos Shirt -> Waterski: 6 * 2 + 2 = 14
        g_baf = empty_game()
        g_baf.players[0].energy = 20
        g_baf.players[0].hand.extend(
            [
                make_card("bring-a-friend"),
                make_card("nos-shirt"),
                make_card("waterski"),
            ]
        )
        g_baf.play_card(0, 0)
        g_baf.play_card(0, 0)
        ws_baf = g_baf.play_card(0, 0)
        self.assertEqual(g_baf.card_fun(0, ws_baf), 14)

        # 2) Nos Shirt -> Work Call (-4 + 2 = -2) -> Waterski (6 * 2 = 12)
        g_wc = empty_game()
        g_wc.players[0].energy = 10
        g_wc.players[0].hand.extend(
            [
                make_card("nos-shirt"),
                make_card("work-call"),
                make_card("waterski"),
            ]
        )
        g_wc.play_card(0, 0)
        wc = g_wc.play_card(0, 0)
        ws_wc = g_wc.play_card(0, 0)
        self.assertEqual(g_wc.card_fun(0, wc), -2)
        self.assertEqual(g_wc.card_fun(0, ws_wc), 12)

        # Child character forces Food to 0 after Nos Shirt's +2.
        from thenos.characters import CHARACTERS

        child = next(c for c in CHARACTERS if c.title == "Child")
        g_child = empty_game()
        g_child.characters = True
        g_child.players[0].character = child
        g_child.players[0].energy = 10
        g_child.players[0].hand.extend(
            [make_card("nos-shirt"), make_card("fajitas")]
        )
        g_child.play_card(0, 0)
        fajitas = g_child.play_card(0, 0)
        self.assertEqual(g_child.card_fun(0, fajitas), 0)

    def test_returns_to_hand_only_when_scoring_more_than_seven_fun_today(
        self,
    ) -> None:
        # Exact boundary 7 today (0 + (5 + 2) = 7, despite high cumulative score) -> discarded.
        g_seven = empty_game()
        p7 = g_seven.players[0]
        p7.fun = 25
        p7.energy = 7
        p7.hand.extend([make_card("nos-shirt"), make_card("kneeboard")])
        shirt7 = g_seven.play_card(0, 0)
        g_seven.play_card(0, 0)
        g_seven.end_day()
        self.assertNotIn(shirt7, p7.hand)
        self.assertIn(shirt7, g_seven.discard)

        # Waterski played BEFORE Nos Shirt scores 6 <= 7 -> discarded.
        g_order = empty_game()
        po = g_order.players[0]
        po.energy = 7
        po.hand.extend([make_card("waterski"), make_card("nos-shirt")])
        g_order.play_card(0, 0)
        shirt_late = g_order.play_card(0, 0)
        g_order.end_day()
        self.assertNotIn(shirt_late, po.hand)
        self.assertIn(shirt_late, g_order.discard)

        # Boundary 8 today (0 + (6 + 2) = 8 > 7) -> returns to hand cleanly.
        g_eight = empty_game()
        p8 = g_eight.players[0]
        p8.energy = 7
        p8.hand.extend([make_card("nos-shirt"), make_card("waterski")])
        shirt8 = g_eight.play_card(0, 0)
        g_eight.play_card(0, 0)
        g_eight.end_day()
        self.assertIn(shirt8, p8.hand)
        self.assertNotIn(shirt8, g_eight.discard)
        self.assertEqual(shirt8.markers, {})
        self.assertFalse(shirt8.is_tomorrow)

    def test_wedding_anniversary_copying_nos_shirt_returns_as_printed_card(
        self,
    ) -> None:
        game = empty_game()
        p0 = game.players[0]
        p0.energy = 10
        opp_shirt = make_card("nos-shirt")
        game.players[1].played_today.append(opp_shirt)
        p0.hand.extend(
            [
                make_card("wedding-anniversary"),
                make_card("waterski"),
            ]
        )
        wedding = game.play_card(0, 0)
        ws = game.play_card(0, 0)  # 0 + (6 + 2) = 8 today
        self.assertEqual(game.card_fun(0, ws), 8)
        game.end_day()

        self.assertIn(wedding, p0.hand)
        self.assertEqual(wedding.markers, {})
        self.assertIs(wedding.effective_behavior, wedding.definition.behavior)
        self.assertEqual(wedding.effective_cost, 0)

    def test_fast_sim_scoring_parity(self) -> None:
        from thenos.ais.carbon import _carbon_copy_game
        from thenos.ais.hydrogen import _hydrogen_copy_game

        # Fast Path 2 (purely additive)
        g_fp2 = empty_game()
        g_fp2.players[0].energy = 10
        g_fp2.players[0].hand.extend([make_card("nos-shirt"), make_card("waterski")])
        g_fp2.play_card(0, 0)
        g_fp2.play_card(0, 0)
        exact_fp2 = sum(g_fp2.card_fun(0, c) for c in g_fp2.players[0].played_today)
        self.assertEqual(exact_fp2, 8)
        self.assertEqual(_hydrogen_copy_game(g_fp2).score_played_today_fast(0)[0], exact_fp2)
        csim2 = _carbon_copy_game(g_fp2)
        csim2.end_day()
        self.assertEqual(csim2.players[0].fun, exact_fp2)

        # Fast Path 3 (non-additive Bring a Friend + Nos Shirt)
        g_fp3 = empty_game()
        g_fp3.players[0].energy = 20
        g_fp3.players[0].hand.extend(
            [
                make_card("bring-a-friend"),
                make_card("nos-shirt"),
                make_card("waterski"),
            ]
        )
        for _ in range(3):
            g_fp3.play_card(0, 0)
        exact_fp3 = sum(g_fp3.card_fun(0, c) for c in g_fp3.players[0].played_today)
        self.assertEqual(exact_fp3, 14)
        self.assertEqual(_hydrogen_copy_game(g_fp3).score_played_today_fast(0)[0], exact_fp3)
        csim3 = _carbon_copy_game(g_fp3)
        csim3.end_day()
        self.assertEqual(csim3.players[0].fun, exact_fp3)

        # Copied Nos Shirt (Fast Path 3b / copied path)
        g_copy = empty_game()
        g_copy.players[0].energy = 10
        g_copy.players[1].played_today.append(make_card("nos-shirt"))
        g_copy.players[0].hand.extend(
            [make_card("wedding-anniversary"), make_card("waterski")]
        )
        g_copy.play_card(0, 0)
        g_copy.play_card(0, 0)
        exact_copy = sum(g_copy.card_fun(0, c) for c in g_copy.players[0].played_today)
        self.assertEqual(exact_copy, 8)
        self.assertEqual(_hydrogen_copy_game(g_copy).score_played_today_fast(0)[0], exact_copy)
        csim_c = _carbon_copy_game(g_copy)
        csim_c.end_day()
        self.assertEqual(csim_c.players[0].fun, exact_copy)

        from thenos.ais import OsmiumAI

        osmium = OsmiumAI()

        def assert_canonical_fast_sim(
            hand_slugs: list[str],
            expected: int,
            *,
            opp_played_slugs: list[str] | None = None,
        ) -> None:
            g_exact = empty_game()
            g_exact.trunk = [make_card("nap") for _ in range(12)]
            g_exact.suitcase = [make_card("nap") for _ in range(4)]
            g_exact.players[0].energy = 20
            g_exact.players[0].hand.extend(make_card(s) for s in hand_slugs)
            if opp_played_slugs:
                g_exact.players[1].played_today.extend(
                    make_card(s) for s in opp_played_slugs
                )
            sim_live = osmium._prepare_observable_game(g_exact, 0)
            self.assertTrue(sim_live._all_canonical_defs)
            for _ in hand_slugs:
                g_exact.play_card(0, 0)
                sim_live.play_card(0, 0)
            exact_score = sum(
                g_exact.card_fun(0, c) for c in g_exact.players[0].played_today
            )
            self.assertEqual(exact_score, expected)
            self.assertEqual(sim_live.score_played_today_fast(0)[0], exact_score)
            sim_post = osmium._prepare_observable_game(g_exact, 0)
            self.assertTrue(sim_post._all_canonical_defs)
            self.assertEqual(sim_post.score_played_today_fast(0)[0], exact_score)

        assert_canonical_fast_sim(["nos-shirt", "waterski"], 8)
        assert_canonical_fast_sim(["bring-a-friend", "nos-shirt", "waterski"], 14)
        assert_canonical_fast_sim(
            ["wedding-anniversary", "waterski"],
            8,
            opp_played_slugs=["nos-shirt"],
        )


if __name__ == "__main__":
    unittest.main()
