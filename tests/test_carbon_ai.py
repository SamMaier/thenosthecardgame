import os
import random
import subprocess
import sys
import unittest

from thenos.ai import Carbon as CarbonFromAIModule, CarbonAI as CarbonAIFromAIModule
from thenos.ais import (
    BerylliumBrain,
    Carbon,
    CarbonAI,
    CarbonBrain,
    CarbonbrainAI,
)
from thenos.ais.carbon import (
    _CARBON_EVAL_AI_SINGLETON,
    _carbon_copy_game_for_play,
    _carbon_extra_energy_support,
    _carbon_free_play_value,
    _carbon_hand_keep_value,
    _carbon_own_suitcase_markers,
    _carbon_quick_card_desirability,
    _suitcase_denial_value,
)
from thenos.ais.carbonbrain import CarbonAI as CarbonAlias
from thenos.cards import make_card
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
from thenos.characters import CHARACTERS
from thenos.daily_conditions import DAILY_CONDITIONS
from thenos.game import Game
from thenos.simulation import Competitor, simulate_games


def _dummy_card(
    instance_id: int,
    title: str,
    cost: int,
    fun: int,
    tags: frozenset[str] = frozenset(),
) -> CardInstance:
    return CardInstance(
        instance_id,
        CardDefinition(
            slug=title.lower().replace(" ", "-"),
            title=title,
            tags=tags,
            cost=cost,
            base_fun=fun,
            behavior=CardBehavior(),
        ),
    )


def _carbon_game(seed: int = 41) -> Game:
    return Game(
        [],
        [CarbonAI(random.Random(seed + idx)) for idx in range(4)],
        random.Random(seed + 100),
    )


class CarbonAITests(unittest.TestCase):
    def test_alias_exports(self):
        self.assertIs(Carbon, CarbonAI)
        self.assertIs(CarbonBrain, CarbonAI)
        self.assertIs(CarbonbrainAI, CarbonAI)
        self.assertIs(CarbonFromAIModule, CarbonAI)
        self.assertIs(CarbonAIFromAIModule, CarbonAI)
        self.assertIs(CarbonAlias, CarbonAI)

    def test_observable_game_preserves_acting_hand_and_resamples_hidden_zones(self):
        game = Game(
            create_default_deck(),
            [CarbonAI(random.Random(40 + idx)) for idx in range(4)],
            random.Random(99),
        )
        game.setup()
        acting_index = 1
        live_hand_ids = [c.instance_id for c in game.players[acting_index].hand]
        live_trunk_ids = [c.instance_id for c in game.trunk]

        sample = game.ais[acting_index]._prepare_observable_game(
            game, acting_index
        )

        self.assertEqual(
            [c.instance_id for c in sample.players[acting_index].hand],
            live_hand_ids,
        )
        self.assertEqual([c.instance_id for c in game.trunk], live_trunk_ids)

        game.ais[acting_index].rng = random.Random(777)
        sample_a = game.ais[acting_index]._prepare_observable_game(
            game, acting_index
        )

        game.trunk.reverse()
        swapped_from_trunk = game.trunk.pop()
        swapped_from_opp = game.players[0].hand.pop()
        game.trunk.append(swapped_from_opp)
        game.players[0].hand.append(swapped_from_trunk)

        game.ais[acting_index].rng = random.Random(777)
        sample_b = game.ais[acting_index]._prepare_observable_game(
            game, acting_index
        )

        self.assertEqual(
            [c.instance_id for c in sample_a.trunk],
            [c.instance_id for c in sample_b.trunk],
        )
        for opp_idx in (0, 2, 3):
            self.assertEqual(
                [c.instance_id for c in sample_a.players[opp_idx].hand],
                [c.instance_id for c in sample_b.players[opp_idx].hand],
            )

    def test_lookahead_never_mutates_live_card_markers(self):
        game = _carbon_game()
        game.day = 3
        player = game.players[0]
        player.energy = 7
        player.hand = [
            make_card("scout-other-cottages"),
            make_card("puerto-rico"),
            make_card("wedding-anniversary"),
            make_card("nos-book"),
        ]
        game.suitcase = [
            _dummy_card(10 + i, f"Suitcase {i}", 3, 2, frozenset({"Relax"}))
            for i in range(4)
        ]
        game.trunk = [
            _dummy_card(100 + i, f"Trunk {i}", 2, 2, frozenset({"Relax"}))
            for i in range(20)
        ]
        game._is_setup = True

        _ = game.ais[0].choose_suitcase_card(game, 0, game.suitcase)
        _ = game.ais[0].choose_to_go_to_bed(game, 0, (0, 1, 2, 3))
        _ = game.ais[0].choose_card_to_play(game, 0, (0, 1, 2, 3))

        for c in player.hand:
            self.assertEqual(c.markers, {})
        for c in game.suitcase:
            self.assertEqual(c.markers, {})
        for c in game.trunk:
            self.assertEqual(c.markers, {})

    def test_opponent_suitcase_marker_denial_breaks_equal_picks(self):
        game = _carbon_game()
        game.day = 3
        game._is_setup = True
        game.players[0].energy = 4
        game.players[0].played_today = [_dummy_card(1, "Played", 1, 1)]
        opp_pr = make_card("puerto-rico")
        marker_key = f"_puerto_rico_target_{opp_pr.instance_id}"
        opp_pr.markers["suitcase_target_marker"] = marker_key
        game.players[1].played_today = [opp_pr]

        unmarked = _dummy_card(10, "Plain A", 2, 3)
        marked = _dummy_card(11, "Plain B", 2, 3)
        marked.markers[marker_key] = True
        game.suitcase = [unmarked, marked]

        self.assertGreater(_suitcase_denial_value(game, 0, marked), 1.0)
        self.assertEqual(
            game.ais[0].choose_suitcase_card(game, 0, game.suitcase),
            1,
        )

    def test_shady_spot_energy_support_for_adventure_race(self):
        game = _carbon_game()
        game.day = 3
        player = game.players[0]
        player.energy = 7
        adv = make_card("adventure-race")
        player.hand = [adv]
        self.assertEqual(_carbon_extra_energy_support(game, 0, adv), 0)

        player.hand = [adv, make_card("shady-spot"), make_card("sunscreen")]
        self.assertGreaterEqual(_carbon_extra_energy_support(game, 0, adv), 5)

    def test_day6_hand_keep_value_does_not_double_count_active_or_spent_discounts(
        self,
    ):
        # Active Shady Spot on Day 6 with only 1 Energy left cannot fund 6-cost Adventure Race
        game = _carbon_game()
        game.day = 6
        player = game.players[0]
        player.energy = 1
        player.played_today = [make_card("shady-spot")]
        adv = make_card("adventure-race")
        splendor = make_card("splendor")
        player.hand = [adv, splendor]

        self.assertEqual(_carbon_extra_energy_support(game, 0, adv), 0)
        self.assertLess(_carbon_hand_keep_value(game, 0, adv, {}), 0.0)
        self.assertGreater(_carbon_hand_keep_value(game, 0, splendor, {}), 0.0)
        self.assertEqual(
            game.ais[0].choose_card_to_discard(game, 0, player.hand), 0
        )

        # Already-consumed Shady Spot ([shady_spot, dock_fishing]) gives 0 extra support
        player.played_today = [
            make_card("shady-spot"),
            make_card("dock-fishing"),
        ]
        self.assertEqual(_carbon_extra_energy_support(game, 0, adv), 0)

        # Already-spent Morning Run in tomorrow_cards on Day 6 gives 0 remaining support
        player.played_today = []
        mr = make_card("morning-run")
        mr.is_tomorrow = True
        player.tomorrow_cards = [mr]
        navy = make_card("navy-sealing")
        player.hand = [navy, splendor]
        self.assertEqual(_carbon_extra_energy_support(game, 0, navy), 0)
        self.assertLess(_carbon_hand_keep_value(game, 0, navy, {}), 0.0)

        # Future-day support credits Istanbul even when today's Energy gain is blocked by Sleep In
        game.day = 3
        player.played_today = [make_card("sleep-in")]
        player.hand = [
            navy,
            make_card("istanbul"),
            _dummy_card(70, "Fodder 1", 1, 1),
            _dummy_card(71, "Fodder 2", 1, 1),
        ]
        self.assertEqual(_carbon_extra_energy_support(game, 0, navy), 0)
        self.assertGreater(
            _carbon_extra_energy_support(
                game, 0, navy, extra_fodder=2, include_future_tomorrow=True
            ),
            0,
        )

    def test_extra_suitcase_pick_conserves_day6_energy_and_handles_singleton_deck(
        self,
    ):
        game = _carbon_game()
        game.day = 6
        game.trunk = [_dummy_card(100 + i, f"T{i}", 2, 2) for i in range(6)]
        player = game.players[0]
        player.energy = 6
        player.hand = [
            _dummy_card(1, "Payoff A", 3, 5),
            _dummy_card(2, "Payoff B", 3, 5),
        ]
        mediocre_options = [
            _dummy_card(10 + i, f"Filler {i}", 3, 3) for i in range(4)
        ]
        self.assertFalse(
            game.ais[0].choose_extra_suitcase_pick(game, 0, mediocre_options)
        )

        ramp_options = [
            make_card("butter-tarts"),
            make_card("afternoon-coffee"),
            _dummy_card(20, "Filler", 3, 3),
            _dummy_card(21, "Filler 2", 3, 3),
        ]
        self.assertTrue(
            game.ais[0].choose_extra_suitcase_pick(game, 0, ramp_options)
        )

        # Even with 4 Suitcase options, <2 cards in Trunk + Discard cannot refill two picks
        game.day = 2
        game.trunk = [_dummy_card(150, "Last Trunk Card", 2, 2)]
        game.discard = []
        self.assertFalse(
            game.ais[0].choose_extra_suitcase_pick(game, 0, ramp_options)
        )
        self.assertFalse(
            _CARBON_EVAL_AI_SINGLETON.choose_extra_suitcase_pick(
                game, 0, ramp_options
            )
        )

    def test_pythonhashseed_determinism(self):
        snippet = (
            "import random; "
            "from thenos.ais.carbon import CarbonAI, _carbon_free_play_value, _carbon_quick_card_desirability; "
            "from thenos.cards import make_card; "
            "from thenos.cards.base import CardBehavior, CardDefinition, CardInstance; "
            "from thenos.cards.catalog import create_default_deck; "
            "from thenos.game import Game; "
            "g = Game(create_default_deck(), [CarbonAI(random.Random(10+i)) for i in range(4)], random.Random(77)); "
            "g.setup(); g.day = 2; ai = g.ais[0]; "
            "c = make_card('evening-on-the-dock'); c.markers['_all_tags'] = True; "
            "w = {'Board Game': 1.3, 'Exercise': 0.7, 'Food': -0.4, 'Indoors': 1.1, 'Item': 0.9, 'Outdoors': 1.7, 'Relax': 2.1, 'Social': 0.8}; "
            "dual = lambda i: CardInstance(800+i, CardDefinition(f'd{i}', f'Dual {i}', frozenset({'Food', 'Outdoors'}), 1, 2, CardBehavior())); "
            "p0 = g.players[0]; orig_hand = list(p0.hand); orig_played = list(p0.played_today); "
            "p0.hand = [make_card('cliff-climbing'), make_card('ice-wine'), dual(1), dual(2)]; "
            "p0.played_today = [make_card('colouring')]; "
            "vals = ("
            "repr(_carbon_quick_card_desirability(g, 0, c, w)), "
            "repr(_carbon_free_play_value(g, 0, c, w)), "
            "repr(ai._draft_synergy_adjustment(g, 0, c, w)), "
            "repr(ai._continuation_potential(g, 0)), "
            "repr(ai._evaluate_tomorrow_and_reserve_fast(g, 0, [make_card('colouring')], days_left=3))"
            "); "
            "p0.hand = orig_hand; p0.played_today = orig_played; ai._reserve_eval_cache.clear(); "
            "res = g.run(); "
            "print(vals, res.scores)"
        )
        outputs = []
        for hash_seed in ("1", "2", "3", "5", "7"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = hash_seed
            proc = subprocess.run(
                [sys.executable, "-c", snippet],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            outputs.append(proc.stdout.strip())
        for out in outputs[1:]:
            self.assertEqual(outputs[0], out)

    def test_child_character_and_daily_conditions_support(self):
        game = _carbon_game()
        game.day = 3
        child_char = next(c for c in CHARACTERS if c.title == "Child")
        player = game.players[0]
        player.character = child_char
        player.energy = 7

        colouring = make_card("colouring")
        beaver = make_card("beaver-burgers")
        butter_tarts = make_card("butter-tarts")
        rouladen = make_card("rouladen")
        wanted = {"Food": -5.0, "Social": 1.5}
        self.assertLess(
            _carbon_quick_card_desirability(game, 0, colouring, wanted), -4.0
        )
        self.assertLess(
            _carbon_quick_card_desirability(game, 0, beaver, wanted), 0.0
        )
        self.assertGreater(
            _carbon_quick_card_desirability(game, 0, butter_tarts, wanted), 0.5
        )
        self.assertAlmostEqual(
            _carbon_free_play_value(game, 0, rouladen, wanted), 1.62
        )

        # DailyCondition ordering places Outdoor-friendly condition before Rainy Day when holding Outdoors
        player.character = None
        player.hand = [make_card("waterski"), make_card("dock-fishing")]
        rainy = next(d for d in DAILY_CONDITIONS if d.title == "Rainy Day")
        beautiful = next(
            d for d in DAILY_CONDITIONS if d.title == "Beautiful Day"
        )
        order = game.ais[0].order_daily_conditions(game, 0, [rainy, beautiful])
        self.assertEqual(tuple(order), (1, 0))

        # Full 6-day game smoke check with both characters and daily_conditions enabled
        full_game = Game(
            create_default_deck(daily_conditions=True),
            [CarbonAI(random.Random(300 + i)) for i in range(4)],
            random.Random(909),
            characters=True,
            daily_conditions=True,
        )
        result = full_game.run()
        self.assertEqual(result.days_played, 6)
        self.assertAlmostEqual(sum(result.win_shares), 1.0)

    def test_copied_scout_self_markers_live_trunk_memory_and_instance_reuse(
        self,
    ):
        # Wedding Anniversary copying Scout Other Cottages registers own marker
        wa = make_card("wedding-anniversary")
        scout = make_card("scout-other-cottages")
        wa.markers["_copied_definition"] = scout.definition
        p = _carbon_game().players[0]
        p.played_today = [wa]
        self.assertIn(
            f"_scout_energy_cube_{wa.instance_id}",
            _carbon_own_suitcase_markers(p),
        )

        # Live Trunk-top memory boosts Assorted Cutlery desirability on live game
        game = _carbon_game()
        game.day = 3
        game._is_setup = True
        game.players[0].energy = 5
        cutlery = make_card("assorted-cutlery")
        baseline = _carbon_quick_card_desirability(game, 0, cutlery, {})
        game.trunk = [_dummy_card(50 + i, f"T{i}", 2, 2) for i in range(8)]
        game.return_cards_to_trunk_top(0, [make_card("waterski")])
        boosted = _carbon_quick_card_desirability(game, 0, cutlery, {})
        self.assertGreater(boosted, baseline + 1.5)

        # Reusing the same CarbonAI instance on a brand-new Game clears stale Trunk memory
        reused_ai = game.ais[0]
        del game
        game2 = Game(
            [make_card("waterski") for _ in range(20)],
            [reused_ai, *(CarbonAI(random.Random(i)) for i in range(3))],
            random.Random(123),
        )
        game2.day = 3
        self.assertEqual(reused_ai._get_live_known_trunk_top(game2, 0), ())

    def test_serial_and_parallel_simulation_parity(self):
        competitors = (
            Competitor("Carbon", CarbonAI),
            Competitor("Beryllium", BerylliumBrain),
            Competitor("Carbon", CarbonAI),
            Competitor("Beryllium", BerylliumBrain),
        )
        serial = simulate_games(
            2,
            seed=20261199,
            competitors=competitors,
            rotate_seats=True,
            workers=1,
        )
        parallel = simulate_games(
            2,
            seed=20261199,
            competitors=competitors,
            rotate_seats=True,
            workers=2,
        )
        self.assertEqual(serial.rows(), parallel.rows())
        self.assertEqual(serial.ais["Carbon"], parallel.ais["Carbon"])
        self.assertEqual(serial.ais["Beryllium"], parallel.ais["Beryllium"])

    def test_cow_cloning_and_cache_isolation_edge_cases(self):
        game = _carbon_game()
        game.day = 2
        ai = game.ais[0]

        # 1. Intra-opponent Nos Book + cross-opponent Christmas Name Draw stay isolated across CoW branches
        opp1_card = make_card("waterski")
        opp1_nos = make_card("nos-book")
        opp1_card.markers["energy_cube"] = True
        opp1_nos.markers["target_card"] = opp1_card
        game.players[1].played_today = [opp1_card, opp1_nos]

        opp2_cnd = make_card("christmas-name-draw")
        opp2_cnd.markers["target_card"] = opp1_card
        opp2_cnd2 = make_card("christmas-name-draw")
        opp2_cnd2.markers["target_card"] = opp1_card
        game.players[2].played_today = [opp2_cnd, opp2_cnd2]

        tom = make_card("zumba")
        tom.is_tomorrow = True
        game.players[0].tomorrow_cards = [tom]
        opp1_chuck = make_card("chuck-a-frisbee")
        game.players[1].played_today.append(opp1_chuck)

        base_sim = ai._prepare_observable_game(game, 0)
        orig_opp2_target = base_sim.players[2].played_today[0].markers["target_card"]
        orig_opp1_hand_len = len(base_sim.players[1].hand)

        branch = _carbon_copy_game_for_play(base_sim, 0)
        branch.go_to_bed(1)  # unshares opp 1; must also unshare opp 2 before remapping opp2_cnd(s)
        self.assertIs(base_sim.players[2].played_today[0].markers["target_card"], orig_opp2_target)
        self.assertIs(base_sim.players[2].played_today[1].markers["target_card"], orig_opp2_target)
        self.assertIs(
            branch.players[1].played_today[1].markers["target_card"],
            branch.players[1].played_today[0],
        )
        self.assertIs(
            branch.players[2].played_today[0].markers["target_card"],
            branch.players[1].played_today[0],
        )
        self.assertIs(
            branch.players[2].played_today[1].markers["target_card"],
            branch.players[1].played_today[0],
        )

        # 2. Direct end_day() on a CoW play clone never mutates parent tomorrow_cards or opponent hand
        branch.end_day()
        self.assertTrue(base_sim.players[0].tomorrow_cards[0].is_tomorrow)
        self.assertEqual(len(base_sim.players[1].hand), orig_opp1_hand_len)
        self.assertFalse(branch._has_card_ref)

        # 3. copy_card_effect sets _has_card_ref so subsequent CoW clones preserve Nos Book +3 Fun
        game2 = _carbon_game()
        game2.day = 3
        opp_nos = make_card("nos-book")
        game2.players[1].played_today = [opp_nos]
        p0_first = make_card("waterski")
        p0_wa = make_card("wedding-anniversary")
        game2.players[0].played_today = [p0_first, p0_wa]
        game2.players[0].energy = 10
        sim2 = ai._prepare_observable_game(game2, 0)
        sim2.copy_card_effect(0, sim2.players[1].played_today[0], sim2.players[0].played_today[1])
        self.assertTrue(sim2._has_card_ref)
        sim2_child = _carbon_copy_game_for_play(sim2, 0)
        self.assertEqual(sim2_child.score_played_today_fast(0)[0], 9)

        # 4. Non-canonical CardDefinitions (including copied tomorrow cards) do not collide in _reserve_eval_cache
        game3 = _carbon_game()
        game3.players[0].hand = [_dummy_card(901, "Synthetic Same Slug", 1, 2)]
        v_low = ai._evaluate_tomorrow_and_reserve_fast(game3, 0, [], days_left=3)
        game3.players[0].hand = [_dummy_card(902, "Synthetic Same Slug", 1, 9)]
        v_high = ai._evaluate_tomorrow_and_reserve_fast(game3, 0, [], days_left=3)
        self.assertGreater(v_high, v_low + 2.0)
        wa_tom = make_card("wedding-anniversary")
        wa_tom.is_tomorrow = True
        wa_tom.markers["_copied_definition"] = make_card("zumba").definition
        game3.players[0].hand = [make_card("waterski")]
        self.assertGreater(
            ai._evaluate_tomorrow_and_reserve_fast(game3, 0, [wa_tom], days_left=3),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
