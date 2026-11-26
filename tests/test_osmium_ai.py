from dataclasses import replace
import os
import random
import subprocess
import sys
import unittest

from thenos.ai import (
    Osmium as OsmiumFromAIModule,
    OsmiumAI as OsmiumAIFromAIModule,
    OsmiumBrain as OsmiumBrainFromAIModule,
    OsmiumbrainAI as OsmiumbrainAIFromAIModule,
)
from thenos.ais import (
    HydrogenAI,
    Osmium,
    OsmiumAI,
    OsmiumBrain,
    OsmiumbrainAI,
    RandomAI,
)
from thenos.ais.argonbrain import _CATALOG_TAG_COUNTS, _unseen_tag_counts_and_total
from thenos.ais.hydrogen import (
    _COMM_NONE,
    _CQ_BOUGIE_COFFEE_MACHINE,
    _CQ_CHEESY_PHONE_GAME,
    _CQ_GROUP_DINNER,
    _CQ_NONE,
    _HQ_NONE,
    _HYDROGEN_ISTANBUL_CACHE,
    _HYDROGEN_META_BY_SLUG,
    _HYDROGEN_RESERVE_EVAL_CACHE,
    _MASK_BOARD_GAME,
    _MASK_EXERCISE,
    _MASK_FOOD,
    _MASK_RELAX,
    _MASK_SOCIAL,
    _MF_ADE,
    _PF_HAS_FUN_VAL,
    _PF_HAS_PHOTO_SHOOT,
    _PF_HAS_SIG_MARKER,
    _QGLOB_COUCH_TUBE,
    _QGLOB_DOCK,
    _QGLOB_NONE,
    _QSYN_NEG_SMB,
    _QSYN_NONE,
    _HydrogenEvalAI,
    _build_quick_zone_ctx,
    _build_w_arr_only,
    _build_wanted_and_ctx,
    _eval_q_const_with_arr,
    _evaluate_istanbul_discards_fast,
    _get_card_meta,
    _get_hydrogen_sim_game_cls,
    _hydrogen_copy_game,
    _hydrogen_quick_card_desirability,
)
from thenos.ais.osmium import (
    _NEUTRAL_Q_OK_D0,
    _NEUTRAL_Q_OK_D1,
    _OSMIUM_RESERVE_CACHE_MAX,
    _OSMIUM_RESERVE_EVAL_CACHE,
    _OSMIUM_SPEC_BY_SLUG,
    _SC_PLAIN_CQ_CODES,
    _awake_opp_new_cards_adjustment,
    _compute_blocked_tag_mask,
    _ensure_osmium_caches,
    _eval_plain_sc_spec,
    _osmium_reserve_score,
    _osmium_reserve_score_plain_specs,
    _suitcase_pick1_edge,
)
from thenos.cards import make_card
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import CARD_REGISTRY, create_default_deck
from thenos.characters import CHARACTERS
from thenos.daily_conditions import DAILY_CONDITIONS
from thenos.game import Game
from thenos.simulation import Competitor, simulate_games


class _FastOsmiumAI(OsmiumAI):
    BEAM_WIDTH = 6
    DAY35_BEAM_WIDTH = 6
    DAY6_BEAM_WIDTH = 8
    DAY6_BRANCH_LIMIT = 6
    DAY6_MAX_PLAYS = 8
    BRANCH_LIMIT = 5
    DRAFT_BEAM_WIDTH = 5
    DAY6_DRAFT_BEAM_WIDTH = 6
    DRAFT_SAMPLES = 2


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


def _osmium_game(seed: int = 41) -> Game:
    return Game(
        [],
        [OsmiumAI(random.Random(seed + idx)) for idx in range(4)],
        random.Random(seed + 100),
    )


class OsmiumAITests(unittest.TestCase):
    def test_alias_exports(self):
        self.assertIs(Osmium, OsmiumAI)
        self.assertIs(OsmiumBrain, OsmiumAI)
        self.assertIs(OsmiumbrainAI, OsmiumAI)
        self.assertIs(OsmiumFromAIModule, OsmiumAI)
        self.assertIs(OsmiumAIFromAIModule, OsmiumAI)
        self.assertIs(OsmiumBrainFromAIModule, OsmiumAI)
        self.assertIs(OsmiumbrainAIFromAIModule, OsmiumAI)
        self.assertIsNot(OsmiumAI, HydrogenAI)

    def test_reserve_cache_isolation_and_subclass_isolation(self):
        ai = OsmiumAI(random.Random(7))
        self.assertIs(ai._reserve_eval_cache, _OSMIUM_RESERVE_EVAL_CACHE)
        self.assertIsNot(_OSMIUM_RESERVE_EVAL_CACHE, _HYDROGEN_RESERVE_EVAL_CACHE)

        class _SubOsmium(OsmiumAI):
            pass

        sub_ai = _SubOsmium(random.Random(8))
        self.assertIsNot(sub_ai._reserve_eval_cache, _OSMIUM_RESERVE_EVAL_CACHE)

        game = _osmium_game()
        game.day = 3
        game.players[0].hand = [make_card("waterski"), make_card("zumba")]
        val1 = sub_ai._evaluate_tomorrow_and_reserve_fast(game, 0, [], days_left=3)
        game.players[0].hand = [make_card("zumba"), make_card("waterski")]
        val2 = sub_ai._evaluate_tomorrow_and_reserve_fast(game, 0, [], days_left=3)
        self.assertTrue(sub_ai._reserve_eval_cache)
        self.assertEqual(len(sub_ai._reserve_eval_cache), 1)
        self.assertAlmostEqual(val1, val2)
        self.assertGreater(_OSMIUM_RESERVE_CACHE_MAX, 10000)

    def test_observable_state_invariance(self):
        def snapshot_decisions(scramble_seed: int) -> tuple[int, int | None, int, bool]:
            g = Game(
                create_default_deck(daily_conditions=True),
                [OsmiumAI(random.Random(20 + i)) for i in range(4)],
                random.Random(500),
                daily_conditions=True,
                characters=True,
            )
            g.setup()
            g.day = 2
            g.start_day()
            sc_rng = random.Random(scramble_seed)
            for opp_i in range(1, 4):
                sc_rng.shuffle(g.players[opp_i].hand)
            sc_rng.shuffle(g.trunk)
            sc_rng.shuffle(g._condition_deck)
            g.rng.seed(scramble_seed + 999)
            ai = g.ais[0]
            pick = ai.choose_suitcase_card(g, 0, g.suitcase)
            play = ai.choose_card_to_play(g, 0, g.playable_hand_indices(0))
            g.players[0].played_today.append(make_card("epic-prank"))
            target_p = ai.choose_player(g, 0, (1, 2, 3))
            disc = ai.choose_card_to_discard(g, 0, tuple(g.players[0].hand))
            return pick, play, disc, target_p

        self.assertEqual(snapshot_decisions(1), snapshot_decisions(987654))

    def test_live_non_mutation_across_decisions(self):
        game = _osmium_game(seed=77)
        game.day = 3
        player = game.players[0]
        player.energy = 7
        player.hand = [
            make_card("scout-other-cottages"),
            make_card("puerto-rico"),
            make_card("wedding-anniversary"),
            make_card("kayak"),
            make_card("istanbul"),
        ]
        game.players[1].played_today = [make_card("carcassonne"), make_card("movie")]
        game.suitcase = [
            _dummy_card(10 + i, f"Suitcase {i}", 3, 2, frozenset({"Relax"}))
            for i in range(4)
        ]
        game.trunk = [
            _dummy_card(100 + i, f"Trunk {i}", 2, 2, frozenset({"Relax"}))
            for i in range(20)
        ]
        game._is_setup = True
        rng_before = game.rng.getstate()

        ai = game.ais[0]
        _ = ai.choose_suitcase_card(game, 0, game.suitcase)
        playable = tuple(game.playable_hand_indices(0))
        _ = ai.choose_to_go_to_bed(game, 0, playable)
        _ = ai.choose_card_to_play(game, 0, playable)
        _ = ai.choose_card_to_discard(game, 0, tuple(player.hand[:3]))
        _ = ai.choose_cards_to_discard(game, 0, tuple(player.hand[:3]))
        _ = ai.choose_card_to_copy(game, 0, tuple(game.players[1].played_today))
        _ = ai.choose_energy_to_spend(game, 0, player.hand[3], 3)
        prank_card = make_card("epic-prank")
        player.played_today.append(prank_card)
        _ = ai.choose_player(game, 0, (1, 2, 3))
        player.played_today.pop()
        food_opt1 = make_card("doritos")
        food_opt2 = make_card("fajitas")
        game.suitcase[:2] = [food_opt1, food_opt2]
        atf_card = make_card("address-the-food")
        player.played_today.append(atf_card)
        _ = ai.choose_card_target(game, 0, (food_opt1, food_opt2))
        player.played_today.pop()

        self.assertEqual(game.rng.getstate(), rng_before)
        for c in player.hand:
            self.assertEqual(c.markers, {})
        for c in game.suitcase:
            self.assertEqual(c.markers, {})
        for c in game.trunk:
            self.assertEqual(c.markers, {})
        self.assertEqual(prank_card.markers, {})
        self.assertEqual(atf_card.markers, {})

    def test_custom_unknown_cards_do_not_crash(self):
        game = _osmium_game(seed=55)
        game.day = 5
        player = game.players[0]
        player.energy = 7
        player.hand = [
            _dummy_card(9001, "Custom Alpha", 2, 5, frozenset({"Food"})),
            _dummy_card(9002, "Custom Beta", 1, 3, frozenset({"Social"})),
        ]
        game.suitcase = [
            _dummy_card(9003 + i, f"Custom Suit {i}", i, 2 + i, frozenset({"Relax"}))
            for i in range(4)
        ]
        game.trunk = [
            _dummy_card(9100 + i, f"Custom Trunk {i}", 1, 2, frozenset({"Outdoors"}))
            for i in range(16)
        ]
        game._is_setup = True
        ai = game.ais[0]
        val = ai._evaluate_tomorrow_and_reserve_fast(game, 0, [], days_left=1)
        pick = ai.choose_suitcase_card(game, 0, game.suitcase)
        self.assertIsInstance(val, float)
        self.assertIn(pick, range(len(game.suitcase)))

    def test_choose_card_to_discard_respects_tell_a_story_setup_and_drops_weak_card(self):
        game = _osmium_game(seed=91)
        game.day = 6
        player = game.players[0]
        player.energy = 5
        junk = _dummy_card(901, "Junk Card", 6, 0)
        strong = make_card("waterski")
        player.hand = [junk, strong]
        game.trunk = [
            _dummy_card(500 + i, f"Trunk {i}", 1, 1, frozenset({"Relax"}))
            for i in range(12)
        ]
        game._is_setup = True
        hand_before = list(player.hand)
        idx = game.ais[0].choose_card_to_discard(game, 0, (junk, strong))
        self.assertEqual(idx, 0)
        self.assertEqual(player.hand, hand_before)

        # Tell a Story places a high-value combo card on top of the Trunk (e.g. Rouladen for BCM).
        tas_game = _osmium_game(seed=92)
        tas_game.day = 3
        tas_p = tas_game.players[0]
        tas_p.energy = 5
        tas_p.played_today = [make_card("epic-duels"), make_card("tell-a-story")]
        tas_p.hand = [
            make_card("bougie-coffee-machine"),
            make_card("rouladen"),
            make_card("zumba"),
        ]
        tas_idx = tas_game.ais[0].choose_card_to_discard(
            tas_game, 0, tuple(tas_p.hand)
        )
        self.assertEqual(tas_p.hand[tas_idx].definition.slug, "rouladen")

    def test_leftovers_branches_align_with_copy_chain_not_marker(self):
        game = _osmium_game(seed=93)
        game.day = 6
        p = game.players[0]
        p.energy = 3
        fajitas = make_card("fajitas")
        left1 = make_card("leftovers")
        left1.markers["_copying_effect"] = True
        left1.markers["_copy_chain"] = (fajitas.instance_id,)
        rouladen = make_card("rouladen")
        left2 = make_card("leftovers")
        p.played_today = [fajitas, left1, rouladen]
        p.hand = [left2]
        game._is_setup = True

        branches = game.ais[0]._simulate_play_branches(game, 0, 0)
        copied_chains = {
            tuple(b.players[0].played_today[-1].markers.get("_copy_chain", ()))
            for b in branches
        }
        # Must include Rouladen (id not in _copy_chain) alongside earlier Food targets.
        self.assertTrue(
            any(rouladen.instance_id in ch for ch in copied_chains),
            copied_chains,
        )

    def test_child_zero_fun_tag_penalizes_colouring_and_ignores_food_value(self):
        child = next(c for c in CHARACTERS if c.zero_fun_tag == "Food")
        game = _osmium_game(seed=94)
        game.day = 5
        p = game.players[0]
        p.character = child
        ai = game.ais[0]

        p.hand = [make_card("rouladen")]
        roul_val = ai._evaluate_tomorrow_and_reserve_fast(game, 0, [], days_left=1)
        p.hand = [make_card("waterski")]
        ski_val = ai._evaluate_tomorrow_and_reserve_fast(game, 0, [], days_left=1)
        self.assertAlmostEqual(roul_val, 0.0)
        self.assertGreater(ski_val, roul_val)

        # Child must not receive an after_all_tags scaler synergy bonus.
        class _AfterAllTags(CardBehavior):
            after_all_tags = True

        after_all = CardInstance(
            9900,
            CardDefinition(
                slug="after-all",
                title="After All",
                tags=frozenset({"Relax"}),
                cost=3,
                base_fun=0,
                behavior=_AfterAllTags(),
            ),
        )
        p.hand = [make_card("carcassonne"), make_card("movie")]
        without_after_all = ai._evaluate_tomorrow_and_reserve_fast(
            game, 0, [], days_left=1
        )
        p.hand = [make_card("carcassonne"), make_card("movie"), after_all]
        with_after_all = ai._evaluate_tomorrow_and_reserve_fast(
            game, 0, [], days_left=1
        )
        self.assertLessEqual(with_after_all, without_after_all)

        # A tomorrow card granting all tags zeroes all tomorrow cards under Child.
        class _TomAllTags(CardBehavior):
            tomorrow_all_tags = True

        tom_all = CardInstance(
            9901,
            CardDefinition(
                slug="tom-all",
                title="Tom All",
                tags=frozenset(),
                cost=1,
                base_fun=0,
                behavior=_TomAllTags(),
            ),
        )
        p.hand = [make_card("waterski"), make_card("zumba")]
        no_tom = ai._evaluate_tomorrow_and_reserve_fast(game, 0, [], days_left=1)
        with_tom_all = ai._evaluate_tomorrow_and_reserve_fast(
            game, 0, [tom_all], days_left=1
        )
        self.assertLess(with_tom_all, no_tom - 15.0)

    def test_ftb_ctx_ignores_unshared_opponents(self):
        game = _osmium_game(seed=95)
        game.day = 3
        game.suitcase = [make_card("istanbul"), make_card("movie"), make_card("zumba"), make_card("nap")]
        for pl in game.players:
            pl.hand = [make_card("movie"), make_card("carcassonne")]
            pl.energy = 2
        game._is_setup = True
        ai = game.ais[0]
        p = game.players
        ftb_ctx = (p[0], p[1], p[2], p[3], 3, {})

        unshared = _hydrogen_copy_game(game)
        unshared.players[1].energy = 7
        _ = ai._evaluate_terminal_day(
            unshared, 0, turns_from_root=2, root_skipped_turns=0, _ftb_ctx=ftb_ctx
        )

        p[1].energy = 1
        shared_with_ctx = ai._evaluate_terminal_day(
            game, 0, turns_from_root=2, root_skipped_turns=0, _ftb_ctx=ftb_ctx
        )
        shared_fresh = ai._evaluate_terminal_day(
            game, 0, turns_from_root=2, root_skipped_turns=0
        )
        self.assertAlmostEqual(shared_with_ctx, shared_fresh)

    def test_pythonhashseed_determinism(self):
        snippet = (
            "import random; "
            "from tests.test_osmium_ai import _FastOsmiumAI; "
            "from thenos.cards import make_card; "
            "from thenos.cards.catalog import create_default_deck; "
            "from thenos.game import Game; "
            "g = Game(create_default_deck(daily_conditions=True), [_FastOsmiumAI(random.Random(10+i)) for i in range(4)], random.Random(77), daily_conditions=True, characters=True); "
            "g.setup(); g.day = 2; ai = g.ais[0]; "
            "val = ai._evaluate_tomorrow_and_reserve_fast(g, 0, [make_card('epic-prank')], days_left=3); "
            "ai._reserve_eval_cache.clear(); "
            "res = g.run(); "
            "print(repr(val), res.scores, res.win_shares)"
        )
        outputs = []
        for hash_seed in ("1", "2", "3"):
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

    def test_full_six_day_smoke_with_characters_and_daily_conditions(self):
        game = Game(
            create_default_deck(daily_conditions=True),
            [
                _FastOsmiumAI(random.Random(300)),
                RandomAI(random.Random(301)),
                RandomAI(random.Random(302)),
                RandomAI(random.Random(303)),
            ],
            random.Random(909),
            characters=True,
            daily_conditions=True,
        )
        result = game.run()
        self.assertEqual(result.days_played, 6)
        self.assertAlmostEqual(sum(result.win_shares), 1.0)

    def test_serial_and_parallel_simulation_parity(self):
        competitors = (
            Competitor("Osmium", _FastOsmiumAI),
            Competitor("Random", RandomAI),
            Competitor("Osmium", _FastOsmiumAI),
            Competitor("Random", RandomAI),
        )
        serial = simulate_games(
            2,
            seed=20261299,
            competitors=competitors,
            rotate_seats=True,
            daily_conditions=True,
            characters=True,
            workers=1,
        )
        parallel = simulate_games(
            2,
            seed=20261299,
            competitors=competitors,
            rotate_seats=True,
            daily_conditions=True,
            characters=True,
            workers=2,
        )
        self.assertEqual(serial.rows(), parallel.rows())
        self.assertEqual(serial.ais["Osmium"], parallel.ais["Osmium"])
        self.assertEqual(serial.ais["Random"], parallel.ais["Random"])

    def test_colouring_played_before_non_relax_cards_with_relax_scaler(self):
        game = _osmium_game(seed=101)
        game.day = 6
        p = game.players[0]
        p.energy = 7
        p.hand = [
            make_card("sweet-lawn-chair"),
            make_card("colouring"),
            make_card("tres-fute"),
        ]
        game.suitcase = [make_card("nap"), make_card("nap"), make_card("nap"), make_card("nap")]
        game.trunk = [_dummy_card(300 + i, f"Trunk {i}", 2, 1) for i in range(12)]
        game._is_setup = True

        play_order: list[str] = []
        while game.playable_hand_indices(0):
            playable = tuple(game.playable_hand_indices(0))
            if game.ais[0].choose_to_go_to_bed(game, 0, playable):
                break
            idx = game.ais[0].choose_card_to_play(game, 0, playable)
            play_order.append(p.hand[idx].title)
            game.play_card(0, idx)
        # Optimal: Sweet Lawn Chair (0) -> Colouring (0+2=2) -> Tres Fute (1+2=3) => 5 Fun.
        self.assertEqual(play_order, ["Sweet Lawn Chair", "Colouring", "Tres Fute"])
        self.assertEqual(sum(game.card_fun(0, c) for c in p.played_today), 5)

    def test_after_dinner_entertainment_orders_food_then_ade_then_exercise(self):
        game = _osmium_game(seed=102)
        game.day = 3
        p = game.players[0]
        p.energy = 9
        p.hand = [
            make_card("after-dinner-entertainment"),
            make_card("waterski"),
            make_card("cheap-white"),
        ]
        game.suitcase = [make_card("movie"), make_card("nap"), make_card("zumba"), make_card("solo")]
        game.trunk = [_dummy_card(400 + i, f"Trunk {i}", 2, 1) for i in range(12)]
        game._is_setup = True

        play_order: list[str] = []
        while game.playable_hand_indices(0):
            playable = tuple(game.playable_hand_indices(0))
            if game.ais[0].choose_to_go_to_bed(game, 0, playable):
                break
            idx = game.ais[0].choose_card_to_play(game, 0, playable)
            play_order.append(p.hand[idx].title)
            game.play_card(0, idx)
        self.assertEqual(
            play_order,
            ["Cheap White", "After Dinner Entertainment", "Waterski"],
        )
        self.assertEqual(sum(game.card_fun(0, c) for c in p.played_today), 15)

    def test_epic_prank_targeting_reserve_and_first_card_tomorrow(self):
        game = _osmium_game(seed=103)
        game.day = 3
        game.start_day()
        # Opp 1: asleep with 1 Fun today (lowest projected).
        game.players[1].asleep = True
        game.players[1].hand = [make_card("nap")]
        game.players[1].played_today = [make_card("tres-fute")]
        # Opp 2: asleep with 6 Fun today.
        game.players[2].asleep = True
        game.players[2].played_today = [make_card("waterski")]
        # Opp 3: awake with 7 energy and 4 cards in hand.
        game.players[3].asleep = False
        game.players[3].energy = 7
        game.players[3].hand = [make_card("nap") for _ in range(4)]

        game.players[0].played_today = [make_card("epic-prank")]
        chosen = game.ais[0].choose_player(game, 0, (1, 2, 3))
        self.assertEqual(chosen, 1)

        # Reserve values Epic Prank tomorrow strictly more with a high-Fun non-late hand card.
        ai = game.ais[0]
        tom_ep = [make_card("epic-prank")]
        game.players[0].hand = [make_card("nap"), make_card("nap")]
        v_low_ep = ai._evaluate_tomorrow_and_reserve_fast(game, 0, tom_ep, days_left=2)
        v_low_no = ai._evaluate_tomorrow_and_reserve_fast(game, 0, [], days_left=2)
        game.players[0].hand = [make_card("waterski"), make_card("nap")]
        v_high_ep = ai._evaluate_tomorrow_and_reserve_fast(game, 0, tom_ep, days_left=2)
        v_high_no = ai._evaluate_tomorrow_and_reserve_fast(game, 0, [], days_left=2)
        self.assertGreater(v_high_ep - v_high_no, (v_low_ep - v_low_no) + 1.5)

        # Next day with Epic Prank active, Osmium plays Waterski FIRST to double it.
        g2 = _osmium_game(seed=104)
        g2.day = 4
        p2 = g2.players[0]
        p2.energy = 7
        ep = make_card("epic-prank")
        ep.is_tomorrow = True
        p2.tomorrow_cards = [ep]
        p2.hand = [make_card("nap"), make_card("waterski"), make_card("cheap-white")]
        g2.suitcase = [make_card("movie"), make_card("nap"), make_card("zumba"), make_card("solo")]
        g2.trunk = [_dummy_card(500 + i, f"Trunk {i}", 2, 1) for i in range(12)]
        g2._is_setup = True
        first_idx = g2.ais[0].choose_card_to_play(g2, 0, g2.playable_hand_indices(0))
        self.assertEqual(p2.hand[first_idx].title, "Waterski")

    def test_dates_first_nos_avoids_stranded_low_fun_and_plays_when_supported(self):
        game = _osmium_game(seed=113)
        game.day = 3
        p = game.players[0]
        p.energy = 7
        p.hand = [make_card("dates-first-nos")]
        game.suitcase = [make_card("nap") for _ in range(4)]
        game.trunk = [_dummy_card(520 + i, f"Trunk {i}", 2, 1) for i in range(12)]
        game._is_setup = True
        self.assertTrue(
            game.ais[0].choose_to_go_to_bed(game, 0, tuple(game.playable_hand_indices(0)))
        )

        # 1 Fun (< 2 threshold) still strands Date's First Nos -> plays Splendor, not Date's First Nos.
        p.energy = 7
        p.played_today = [make_card("tres-fute")]
        p.hand = [make_card("dates-first-nos"), make_card("nap"), make_card("splendor")]
        playable_1 = tuple(game.playable_hand_indices(0))
        if not game.ais[0].choose_to_go_to_bed(game, 0, playable_1):
            self.assertNotEqual(game.ais[0].choose_card_to_play(game, 0, playable_1), 0)

        # Exact 2 Fun threshold (Biography already played) keeps Tomorrow -> plays Date's First Nos.
        p.energy = 7
        p.played_today = [make_card("biography")]
        p.hand = [make_card("dates-first-nos"), make_card("nap"), make_card("splendor")]
        playable_2 = tuple(game.playable_hand_indices(0))
        self.assertFalse(game.ais[0].choose_to_go_to_bed(game, 0, playable_2))
        self.assertEqual(game.ais[0].choose_card_to_play(game, 0, playable_2), 0)

        # When able to play Date's First Nos AND Waterski (>= 2 Fun), Osmium plays them.
        p.energy = 12
        p.played_today = []
        p.hand = [make_card("dates-first-nos"), make_card("waterski")]
        self.assertFalse(
            game.ais[0].choose_to_go_to_bed(game, 0, tuple(game.playable_hand_indices(0)))
        )

    def test_paddleboard_played_late_after_shrinking_hand(self):
        game = _osmium_game(seed=114)
        game.day = 3
        p = game.players[0]
        p.energy = 7
        p.hand = [
            make_card("paddleboard"),
            make_card("nap"),
            make_card("tres-fute"),
            make_card("cheap-white"),
        ]
        game.suitcase = [make_card("movie") for _ in range(4)]
        game.trunk = [_dummy_card(540 + i, f"Trunk {i}", 2, 1) for i in range(12)]
        game._is_setup = True

        play_order: list[str] = []
        while game.playable_hand_indices(0):
            playable = tuple(game.playable_hand_indices(0))
            if game.ais[0].choose_to_go_to_bed(game, 0, playable):
                break
            idx = game.ais[0].choose_card_to_play(game, 0, playable)
            play_order.append(p.hand[idx].title)
            game.play_card(0, idx)
        self.assertIn("Paddleboard", play_order)
        self.assertEqual(play_order[-1], "Paddleboard")

    def test_7_wonders_and_solo_terminal_eval_discount_awake_opponents(self):
        game = _osmium_game(seed=115)
        game.day = 6
        p = game.players[0]
        p.played_today = [make_card("7-wonders"), make_card("solo")]
        for idx in (1, 2, 3):
            game.players[idx].hand = [make_card("nap") for _ in range(4)]
            game.players[idx].energy = 7

        # Opponents awake -> uncertainty discounts 7 Wonders and Solo.
        v_awake = game.ais[0]._evaluate_terminal_day(game, 0)
        # Identical board with opponents asleep -> deterministic full credit.
        for idx in (1, 2, 3):
            game.players[idx].asleep = True
        v_asleep = game.ais[0]._evaluate_terminal_day(game, 0)
        self.assertGreater(v_asleep, v_awake + 1.0)

    def test_cities_and_knights_end_of_day_pick_while_asleep(self):
        game = _osmium_game(seed=116)
        game.day = 3
        p = game.players[0]
        p.asleep = True
        p.played_today = [
            make_card("cities-and-knights"),
            make_card("tres-fute"),
            make_card("splendor"),
            make_card("biography"),
            make_card("nap"),
        ]
        weak = make_card("cheap-red")
        strong = make_card("bring-a-friend")
        game.suitcase = [weak, strong, _dummy_card(9901, "Custom Low", 0, 0), make_card("solo")]
        game.trunk = [_dummy_card(600 + i, f"T{i}", 1, 1) for i in range(12)]
        game._is_setup = True
        rng_before = game.rng.getstate()

        pick_d3 = game.ais[0].choose_suitcase_card(game, 0, game.suitcase)
        self.assertEqual(pick_d3, 1)
        self.assertEqual(game.rng.getstate(), rng_before)

        game.day = 6
        self.assertEqual(game.ais[0].choose_suitcase_card(game, 0, game.suitcase), 0)

    def test_address_the_food_search_targeting_and_ordering_off_by_one(self):
        game = _osmium_game(seed=105)
        game.day = 6
        p = game.players[0]
        p.energy = 3
        p.played_today = [make_card("address-the-food")]
        p.hand = [make_card("leftovers")]
        d_opt = make_card("doritos")
        f_opt = make_card("fajitas")
        game.suitcase = [d_opt, f_opt, make_card("movie"), make_card("zumba")]
        game.trunk = [_dummy_card(700 + i, f"Trunk {i}", 1, 1) for i in range(12)]
        game._is_setup = True

        chosen = game.ais[0].choose_card_target(game, 0, (d_opt, f_opt))
        self.assertEqual(chosen, 1)

        # Off-by-one check in candidate ordering:
        # 1) played_today=[] and only Morning Coffee in Suitcase -> Address the Food cannot target it.
        g_order = _osmium_game(seed=106)
        g_order.day = 3
        po = g_order.players[0]
        po.energy = 4
        po.played_today = []
        po.hand = [make_card("address-the-food"), make_card("zumba")]
        g_order.suitcase = [make_card("morning-coffee"), make_card("movie"), make_card("nap"), make_card("solo")]
        g_order.trunk = [_dummy_card(800 + i, f"Trunk {i}", 1, 1) for i in range(12)]
        g_order._is_setup = True
        cand_no_elig = g_order.ais[0]._ordered_candidate_indices_with_costs(g_order, 0, 4)
        self.assertNotEqual(cand_no_elig[0][0], 0)

        # 2) len(played_today)==1 and Afternoon Coffee in Suitcase -> playing Address the Food makes Afternoon Coffee 3rd card (legal!).
        po.played_today = [make_card("movie")]
        g_order.suitcase[0] = make_card("afternoon-coffee")
        po.hand = [make_card("address-the-food"), make_card("waterski")]
        cand_elig = g_order.ais[0]._ordered_candidate_indices_with_costs(g_order, 0, 4)
        self.assertEqual(cand_elig[0][0], 0)

    def test_reserve_score_fast_and_slow_parity_on_changed_cards(self):
        _ensure_osmium_caches()
        self.assertEqual(_OSMIUM_SPEC_BY_SLUG["istanbul"].net_e_pos, 2.0)
        changed_slugs = [
            "istanbul",
            "colouring",
            "butter-tarts",
            "play-with-the-kids",
            "nos-book",
            "fancy-craft",
            "assorted-cutlery",
            "last-years-shirt",
            "couch-tube",
            "rouladen",
            "boat-ride",
            "ski-on-shoulders",
            "paddleboat",
            "bougie-coffee-machine",
            "7-wonders",
            "after-dinner-entertainment",
            "cities-and-knights",
            "dates-first-nos",
            "epic-prank",
            "fit-to-print",
            "leftovers",
            "nos-shirt",
            "paddleboard",
            "solo",
            "sweet-lawn-chair",
            "waterski",
        ]
        game = _osmium_game(seed=107)
        tom_sets = (
            [make_card("photo-shoot"), make_card("zumba")],
            [make_card("epic-prank"), make_card("dates-first-nos")],
            [make_card("wildlife-spotting")],
        )
        for dl in (1, 3):
            for split in (4, 8, 12, len(changed_slugs)):
                hand = [make_card(s) for s in changed_slugs[:split]]
                for tom in tom_sets:
                    game.players[0].hand = hand
                    sorted_specs = sorted(
                        (_OSMIUM_SPEC_BY_SLUG[c.definition.slug] for c in hand),
                        key=lambda sp: sp.slug_rank,
                    )
                    tom_specs = [_OSMIUM_SPEC_BY_SLUG[c.definition.slug] for c in tom]
                    slow_v = _osmium_reserve_score(game, 0, tom, dl)
                    fast_v = _osmium_reserve_score_plain_specs(
                        sorted_specs, tom_specs, dl
                    )
                    self.assertAlmostEqual(slow_v, fast_v, places=11)

    def test_state_signature_distinguishes_optional_discard_markers(self):
        ai = OsmiumAI(random.Random(108))
        # Canonical cards test OsmiumAI's fast signature path; dummy cards test fallback path.
        for canon in (True, False):
            g1 = _osmium_game(seed=109)
            g1.day = 3
            g1.players[0].hand = [make_card("carcassonne"), make_card("movie")]
            g1.players[0].played_today = [make_card("paddleboard")]
            if canon:
                g1.suitcase = [make_card("movie"), make_card("zumba"), make_card("nap"), make_card("solo")]
                g1.trunk = [make_card("calzones") for _ in range(12)]
            else:
                g1.suitcase = [_dummy_card(10 + i, f"S{i}", 2, 1) for i in range(4)]
                g1.trunk = [_dummy_card(50 + i, f"T{i}", 2, 1) for i in range(12)]
            g1._is_setup = True

            s1 = ai._prepare_observable_game(g1, 0)
            self.assertEqual(bool(s1._all_canonical_defs), canon)
            s2 = _hydrogen_copy_game(s1)
            s2.players[0].played_today[0].markers["energy_cubes"] = 3
            s2._recompute_player_flags(0)
            self.assertNotEqual(ai._state_signature(s1, 0), ai._state_signature(s2, 0))

            g1.players[0].played_today = [make_card("epic-prank")]
            ep1 = ai._prepare_observable_game(g1, 0)
            ep1.players[0].played_today[0].markers["target_player_index"] = 1
            ep1._recompute_player_flags(0)
            ep2 = _hydrogen_copy_game(ep1)
            ep2.players[0].played_today[0].markers["target_player_index"] = 2
            ep2._recompute_player_flags(0)
            self.assertNotEqual(ai._state_signature(ep1, 0), ai._state_signature(ep2, 0))

    def test_next_day_starter_value_uses_fewest_cards_and_clockwise_tiebreak(self):
        game = _osmium_game(seed=110)
        game.day = 3
        game.starting_player = 0
        game.suitcase = [make_card("istanbul"), make_card("movie"), make_card("nap"), make_card("solo")]
        game._is_setup = True
        ai = game.ais[1]

        # All opponents asleep with 2 cards; seat 0 beats seat 1 on tie (starter=0).
        for idx in (0, 2, 3):
            game.players[idx].asleep = True
            game.players[idx].hand = [make_card("nap"), make_card("solo")]

        game.players[1].hand = [make_card("nap"), make_card("solo")]
        self.assertEqual(ai._first_to_bed_value(game, 1), 0.0)

        # Dropping to 1 card strictly beats all asleep opponents -> full pick1 bonus.
        game.players[1].hand = [make_card("nap")]
        expected = ai._suitcase_pick1_bonus(game, 1)
        self.assertAlmostEqual(ai._first_to_bed_value(game, 1), expected)

        # Chuck a Frisbee returning to hand pushes projected end-of-day hand to 2 -> loses tie to seat 0.
        game.players[1].played_today = [make_card("chuck-a-frisbee")]
        self.assertEqual(ai._first_to_bed_value(game, 1), 0.0)

        # If seat 1 was previous starter (starting_player=1), a 2-card tie wins!
        game.starting_player = 1
        self.assertAlmostEqual(ai._first_to_bed_value(game, 1), expected)

    def test_evaluate_terminal_day_keeps_awake_opponent_chuck_unplayable_today(self):
        game = _osmium_game(seed=111)
        game.day = 3
        game.starting_player = 0
        game.suitcase = [make_card("istanbul"), make_card("movie"), make_card("nap"), make_card("solo")]
        game._is_setup = True
        ai = game.ais[0]

        # Seat 0 plays Chuck a Frisbee (triggers needs_full_end_day=True), ending with 1 card.
        # Opp 1 is awake with empty hand + 5 energy + Chuck a Frisbee (guaranteed 1 card at end_day).
        # Opps 2 & 3 are asleep with 2 cards. Seat 0 wins tie on 1 card (p_win == 1.0).
        game.players[0].hand = []
        game.players[0].played_today = [make_card("chuck-a-frisbee")]
        game.players[1].asleep = False
        game.players[1].hand = []
        game.players[1].energy = 5
        game.players[1].played_today = [make_card("chuck-a-frisbee")]
        for idx in (2, 3):
            game.players[idx].asleep = True
            game.players[idx].hand = [make_card("nap"), make_card("solo")]

        val_with_chuck = ai._evaluate_terminal_day(game, 0)
        # Compare against equivalent state with 1 card already in hand (no needs_full_end_day).
        game_ref = _hydrogen_copy_game(game)
        game_ref.players[0].played_today = []
        game_ref.players[0].hand = [make_card("chuck-a-frisbee")]
        game_ref.players[0].fun = 3  # match Chuck a Frisbee's 3 Fun
        val_ref = ai._evaluate_terminal_day(game_ref, 0)
        self.assertAlmostEqual(val_with_chuck, val_ref, places=6)

    def test_projected_end_hand_size_respects_contested_scout_order(self):
        from thenos.ais.osmium import _projected_end_hand_size

        game = _osmium_game(seed=112)
        game.day = 3
        game.starting_player = 2
        game.suitcase = [make_card("nap"), make_card("solo"), make_card("movie"), make_card("azul")]
        game.trunk = [make_card("calzones") for _ in range(8)]
        game.players[2].energy = 7
        game.players[0].energy = 7
        game.players[2].hand = [make_card("scout-other-cottages")]
        game.players[0].hand = [make_card("wedding-anniversary")]
        game.play_card(2, 0)
        game.play_card(0, 0)

        # Seat 2 resolves first in player_order() and claims all 4 Suitcase cards;
        # Seat 0's copy claims 0 remaining cards.
        self.assertEqual(_projected_end_hand_size(game, 2), 4)
        self.assertEqual(_projected_end_hand_size(game, 0), 0)

    def test_fit_to_print_no_double_count_in_projected_fun_and_hand_size(self):
        from thenos.ais.carbon import _projected_opponent_day_fun
        from thenos.ais.osmium import _projected_end_hand_size

        game = _osmium_game(seed=113)
        game.day = 3
        game._is_setup = True
        game.players[1].played_today = [make_card("fit-to-print")]
        self.assertEqual(_projected_opponent_day_fun(game, 1), 5.0)

        game2 = _osmium_game(seed=114)
        game2.day = 3
        game2._is_setup = True
        game2.players[0].played_today = [make_card("nos-shirt"), make_card("fit-to-print")]
        # 0 + (5 + 2) = 7 <= 7, so Nos Shirt does not return to hand.
        self.assertEqual(_projected_end_hand_size(game2, 0), 0)

        game2.players[0].played_today = [make_card("nos-shirt"), make_card("waterski")]
        # 0 + (6 + 2) = 8 > 7, so Nos Shirt returns to hand.
        self.assertEqual(_projected_end_hand_size(game2, 0), 1)

    def test_solo_awake_opp_discount_requires_actual_solo_rescore(self):
        game_solo = _osmium_game(seed=115)
        game_solo.day = 3
        game_solo.starting_player = 0
        game_solo.suitcase = [make_card("istanbul"), make_card("movie"), make_card("nap"), make_card("solo")]
        game_solo.trunk = [make_card("calzones") for _ in range(8)]
        game_solo._is_setup = True
        ai = game_solo.ais[0]

        game_solo.players[0].hand = []
        game_solo.players[0].played_today = [make_card("solo"), make_card("fit-to-print")]
        game_solo.players[1].asleep = True
        game_solo.players[1].hand = []
        game_solo.players[1].played_today = [_dummy_card(910, "Ten Fun", 0, 10)]
        game_solo.players[2].asleep = False
        game_solo.players[2].energy = 5
        game_solo.players[2].hand = [make_card("nap")]
        game_solo.players[3].asleep = True
        game_solo.players[3].hand = [make_card("nap")]

        game_ref = _hydrogen_copy_game(game_solo)
        game_ref.players[0].played_today[0] = _dummy_card(
            911, "NonSolo Three", 3, 3, tags=("Board Game",)
        )

        # Solo (8 today) loses to asleep Opp 1 (10 today) and never rescores,
        # so awake Opp 2 must not trigger a phantom Solo discount.
        self.assertAlmostEqual(
            ai._evaluate_terminal_day(game_solo, 0),
            ai._evaluate_terminal_day(game_ref, 0),
            places=6,
        )

    def test_beryllium_sim_captures_on_play_fun_for_fun_scored_today(self):
        from thenos.ais.beryllium import _beryllium_copy_game

        game = _osmium_game(seed=116)
        game.day = 3
        game._is_setup = True
        game.players[0].fun = 10
        game.players[0].energy = 15
        game.players[0].hand = [
            make_card("wingspan"),
            make_card("chalk-art"),
            make_card("dates-first-nos"),
        ]
        sim = _beryllium_copy_game(game)

        for _ in range(3):
            game.play_card(0, 0)
            sim.play_card(0, 0)
        self.assertEqual(sim._fun_at_start_of_day, (10, 0, 0, 0))
        self.assertEqual(sim._fun_at_start_of_day, game._fun_at_start_of_day)
        game.end_day()
        sim.end_day()

        self.assertEqual(sim.players[0].fun, game.players[0].fun)
        self.assertEqual(
            sim.fun_scored_today(sim.players[0]),
            game.fun_scored_today(game.players[0]),
        )
        self.assertEqual(
            [c.definition.slug for c in sim.players[0].tomorrow_cards],
            ["dates-first-nos"],
        )
        self.assertEqual(
            [c.definition.slug for c in sim.players[0].tomorrow_cards],
            [c.definition.slug for c in game.players[0].tomorrow_cards],
        )

    def test_argon_and_beryllium_epic_prank_choose_lowest_projected_fun_opponent(self):
        from thenos.ais.argonbrain import ArgonBrain
        from thenos.ais.beryllium import BerylliumBrain

        game = _osmium_game(seed=117)
        game.day = 3
        game._is_setup = True
        game.players[0].played_today = [make_card("epic-prank")]
        game.players[1].played_today = [_dummy_card(920, "High NonBG", 1, 6, tags=("Outdoors",))]
        game.players[2].played_today = [_dummy_card(921, "Low BG", 1, 1, tags=("Board Game",))]
        game.players[3].played_today = [_dummy_card(922, "Mid Food", 1, 4, tags=("Food",))]

        self.assertEqual(ArgonBrain().choose_player(game, 0, [1, 2, 3]), 2)
        self.assertEqual(BerylliumBrain().choose_player(game, 0, [1, 2, 3]), 2)

    def test_choose_suitcase_card_while_asleep_clears_suitcase_markers(self):
        game = _osmium_game(seed=118)
        game.day = 3
        game.starting_player = 0
        game._is_setup = True
        game.players[0].asleep = True

        marked_nap = make_card("nap")
        marked_nap.markers["_puerto_rico_target"] = 1
        marked_nap.markers["energy_cube"] = 10
        game.suitcase = [
            marked_nap,
            make_card("bring-a-friend"),
            make_card("solo"),
            make_card("movie"),
        ]

        choice = game.ais[0].choose_suitcase_card(game, 0, game.suitcase)
        self.assertEqual(choice, 1)
        self.assertEqual(
            marked_nap.markers,
            {"_puerto_rico_target": 1, "energy_cube": 10},
        )

    def test_nos_shirt_played_before_followup_for_plus_two_and_returns_to_hand(self):
        game = _osmium_game(seed=119)
        game.day = 3
        p = game.players[0]
        p.energy = 7
        p.hand = [make_card("waterski"), make_card("nos-shirt")]
        game.suitcase = [make_card("nap") for _ in range(4)]
        game.trunk = [_dummy_card(930 + i, f"Trunk {i}", 2, 1) for i in range(12)]
        game._is_setup = True

        play_order: list[str] = []
        while game.playable_hand_indices(0):
            playable = tuple(game.playable_hand_indices(0))
            if game.ais[0].choose_to_go_to_bed(game, 0, playable):
                break
            idx = game.ais[0].choose_card_to_play(game, 0, playable)
            play_order.append(p.hand[idx].title)
            game.play_card(0, idx)
        self.assertEqual(play_order, ["Nos Shirt", "Waterski"])
        self.assertEqual(sum(game.card_fun(0, c) for c in p.played_today), 8)

        # With Epic Prank active Tomorrow, Waterski must still be played FIRST (doubled 6->12),
        # before Nos Shirt -> Nap.
        g_ep = _osmium_game(seed=120)
        g_ep.day = 3
        p_ep = g_ep.players[0]
        p_ep.energy = 7
        ep = make_card("epic-prank")
        ep.is_tomorrow = True
        p_ep.tomorrow_cards = [ep]
        p_ep.hand = [make_card("nos-shirt"), make_card("waterski"), make_card("nap")]
        g_ep.suitcase = [make_card("movie") for _ in range(4)]
        g_ep.trunk = [_dummy_card(960 + i, f"Trunk {i}", 2, 1) for i in range(12)]
        g_ep._is_setup = True

        first_idx = g_ep.ais[0].choose_card_to_play(g_ep, 0, g_ep.playable_hand_indices(0))
        self.assertEqual(p_ep.hand[first_idx].title, "Waterski")

    def test_play_with_the_kids_terminal_eval_discounts_awake_opponents_by_played_count(
        self,
    ):
        game = _osmium_game(seed=121)
        game.day = 6
        p = game.players[0]
        p.played_today = [make_card("play-with-the-kids")]
        for idx in (1, 2, 3):
            game.players[idx].hand = [make_card("nap") for _ in range(4)]
            game.players[idx].energy = 7
            game.players[idx].played_today = [make_card("nap")]

        # Tied on 1 played card with awake opponents -> discounted.
        v_awake = game.ais[0]._evaluate_terminal_day(game, 0)
        for idx in (1, 2, 3):
            game.players[idx].asleep = True
        v_asleep = game.ais[0]._evaluate_terminal_day(game, 0)
        self.assertGreater(v_asleep, v_awake + 1.0)

        # With 5+ cards played today and small hand, awake opponents trigger no phantom discount.
        g_five = _osmium_game(seed=122)
        g_five.day = 6
        p5 = g_five.players[0]
        p5.hand = []
        p5.played_today = [make_card("play-with-the-kids")] + [
            make_card("nap") for _ in range(4)
        ]
        for idx in (1, 2, 3):
            g_five.players[idx].hand = [make_card("nap")]
            g_five.players[idx].energy = 7
        v_five_awake = g_five.ais[0]._evaluate_terminal_day(g_five, 0)
        for idx in (1, 2, 3):
            g_five.players[idx].asleep = True
        v_five_asleep = g_five.ais[0]._evaluate_terminal_day(g_five, 0)
        self.assertAlmostEqual(v_five_awake, v_five_asleep, places=9)

    def test_canonical_fast_path_2_next_tag_parity(self):
        cases = (
            ("water-volleyball", "waterski", 12),
            ("cheap-red", "beaver-burgers", 9),
            ("johnny-appleseed", "beaver-burgers", 8),
            ("canoe", "ponyback", 6),
            ("fancy-craft", "ponyback", 7),
        )
        for first_slug, second_slug, expected in cases:
            with self.subTest(pair=(first_slug, second_slug)):
                g = _osmium_game(seed=123)
                g.players[0].energy = 20
                g.players[0].hand = [make_card(first_slug), make_card(second_slug)]
                g.suitcase = [make_card("nap") for _ in range(4)]
                g.trunk = [make_card("nap") for _ in range(12)]
                g._is_setup = True

                sim_live = g.ais[0]._prepare_observable_game(g, 0)
                self.assertTrue(sim_live._all_canonical_defs)
                g.play_card(0, 0)
                g.play_card(0, 0)
                sim_live.play_card(0, 0)
                sim_live.play_card(0, 0)

                exact = sum(g.card_fun(0, c) for c in g.players[0].played_today)
                self.assertEqual(exact, expected)
                self.assertEqual(sim_live.score_played_today_fast(0)[0], exact)

                sim_post = g.ais[0]._prepare_observable_game(g, 0)
                self.assertTrue(sim_post._all_canonical_defs)
                self.assertEqual(sim_post.score_played_today_fast(0)[0], exact)

    def test_hydrogen_reserve_plain_vs_fallback_parity_butter_tarts(self):
        import dataclasses
        from thenos.ais.hydrogen import HydrogenAI
        from thenos.models import CardInstance

        hai = HydrogenAI()

        g_plain = _osmium_game(seed=124)
        g_plain.day = 5
        g_plain.players[0].hand = [
            make_card("adventure-race"),
            make_card("butter-tarts"),
        ]
        g_plain.suitcase = [make_card("nap") for _ in range(4)]
        g_plain.trunk = [make_card("nap") for _ in range(12)]
        g_plain._is_setup = True

        sim_plain = hai._prepare_observable_game(g_plain, 0)
        self.assertTrue(sim_plain._all_canonical_defs)
        self.assertFalse(sim_plain._has_nonplain_hand)
        plain_val = hai._evaluate_tomorrow_and_reserve_fast(
            sim_plain, 0, (), days_left=1
        )

        g_fb = _osmium_game(seed=124)
        g_fb.day = 5
        adv = make_card("adventure-race")
        adv_noncanon = CardInstance(
            instance_id=adv.instance_id,
            definition=dataclasses.replace(adv.definition),
        )
        g_fb.players[0].hand = [adv_noncanon, make_card("butter-tarts")]
        g_fb.suitcase = [make_card("nap") for _ in range(4)]
        g_fb.trunk = [make_card("nap") for _ in range(12)]
        g_fb._is_setup = True

        sim_fb = hai._prepare_observable_game(g_fb, 0)
        self.assertFalse(sim_fb._all_canonical_defs)
        fallback_val = hai._evaluate_tomorrow_and_reserve_fast(
            sim_fb, 0, (), days_left=1
        )

        self.assertAlmostEqual(plain_val, fallback_val, places=9)

    def test_last_years_shirt_sets_trunk_top_and_plays_before_cutlery(self):
        # After Last Year's Shirt with Assorted Cutlery in hand, place high-payoff Adventure Race on Trunk.
        cutlery_game = _osmium_game(seed=125)
        cutlery_game.day = 3
        cp = cutlery_game.players[0]
        cp.energy = 1
        cp.played_today = [make_card("epic-duels"), make_card("last-years-shirt")]
        cp.hand = [
            make_card("assorted-cutlery"),
            make_card("adventure-race"),
            make_card("nap"),
        ]
        c_idx = cutlery_game.ais[0].choose_card_to_discard(
            cutlery_game, 0, tuple(cp.hand)
        )
        self.assertEqual(cp.hand[c_idx].definition.slug, "adventure-race")

        # After Last Year's Shirt with Bougie Coffee Machine in hand, place Rouladen on Trunk.
        bcm_game = _osmium_game(seed=126)
        bcm_game.day = 3
        bp = bcm_game.players[0]
        bp.energy = 5
        bp.played_today = [make_card("epic-duels"), make_card("last-years-shirt")]
        bp.hand = [
            make_card("bougie-coffee-machine"),
            make_card("rouladen"),
            make_card("zumba"),
        ]
        b_idx = bcm_game.ais[0].choose_card_to_discard(
            bcm_game, 0, tuple(bp.hand)
        )
        self.assertEqual(bp.hand[b_idx].definition.slug, "rouladen")

        # Live play ordering: play Last Year's Shirt before Assorted Cutlery to set up Adventure Race.
        play_game = _osmium_game(seed=127)
        play_game.day = 3
        pp = play_game.players[0]
        pp.energy = 2
        pp.played_today = [make_card("epic-duels")]
        pp.hand = [
            make_card("last-years-shirt"),
            make_card("assorted-cutlery"),
            make_card("adventure-race"),
        ]
        play_game.suitcase = [make_card("nap") for _ in range(4)]
        play_game.trunk = [make_card("nap") for _ in range(12)]
        play_game._is_setup = True
        play_idx = play_game.ais[0].choose_card_to_play(
            play_game, 0, play_game.playable_hand_indices(0)
        )
        self.assertEqual(play_idx, 0)

    def test_inline_end_day_matches_full_end_day_for_simple_cards(self):
        from thenos.ais.hydrogen import _PF_HAS_COMPLEX_END_DAY

        # Test chuck-a-frisbee, nos-shirt (>7 and <=7), dates-first-nos (<2 and >=2),
        # epic-prank (beating target and losing to target), solo (tied/winning and losing),
        # and copied nos-shirt / epic-prank.
        for case_name in (
            "chuck_and_shirt_above_7",
            "shirt_le_7_dates_low",
            "dates_ge_2_epic_win_solo_tie",
            "epic_lose_solo_lose",
            "copied_nos_shirt_with_ski_on_shoulders",
            "scout_other_cottages",
        ):
            with self.subTest(case=case_name):
                g = _osmium_game(seed=201)
                g.day = 3
                g.starting_player = 0
                g.suitcase = [
                    make_card("movie"),
                    make_card("zumba"),
                    make_card("nap"),
                    make_card("solo"),
                ]
                g.trunk = [make_card("calzones") for _ in range(12)]
                g._is_setup = True
                p = g.players[0]
                p.hand = [make_card("nap")]
                p.fun = 5
                g.players[1].asleep = True
                g.players[2].asleep = True
                g.players[3].asleep = True
                if case_name == "chuck_and_shirt_above_7":
                    p.played_today = [
                        make_card("chuck-a-frisbee"),
                        make_card("nos-shirt"),
                        make_card("waterski"),
                    ]
                    g.players[1].played_today = [make_card("movie")]
                elif case_name == "shirt_le_7_dates_low":
                    p.played_today = [
                        make_card("nos-shirt"),
                        make_card("dates-first-nos"),
                    ]
                elif case_name == "dates_ge_2_epic_win_solo_tie":
                    ep = make_card("epic-prank")
                    ep.markers["target_player_index"] = 1
                    p.played_today = [
                        make_card("dates-first-nos"),
                        ep,
                        make_card("solo"),
                        make_card("tres-fute"),
                    ]
                    g.players[1].played_today = [make_card("waterski")]  # 6 == 6 tie!
                elif case_name == "epic_lose_solo_lose":
                    ep = make_card("epic-prank")
                    ep.markers["target_player_index"] = 1
                    p.played_today = [ep, make_card("solo")]
                    g.players[1].played_today = [make_card("waterski")]  # 3 < 6
                elif case_name == "scout_other_cottages":
                    scout = make_card("scout-other-cottages")
                    owner_marker = f"_scout_energy_cube_{scout.instance_id}"
                    g.suitcase[0].markers["energy_cube"] = True
                    g.suitcase[0].markers[owner_marker] = True
                    g.suitcase[1].markers["energy_cube"] = True
                    g.suitcase[1].markers[owner_marker] = True
                    p.played_today = [scout, make_card("movie")]
                else:
                    # Copied Nos Shirt (leftovers-like marker) + Ski on Shoulders (fun_value reads start_of_scoring)
                    g.players[1].fun = 20
                    copied_ns = make_card("leftovers")
                    copied_ns.markers["_copied_definition"] = make_card("nos-shirt").definition
                    copied_ns.markers["_copying_effect"] = True
                    p.played_today = [
                        copied_ns,
                        make_card("waterski"),
                        make_card("ski-on-shoulders"),
                    ]

                ai = g.ais[0]
                sim_inline = ai._prepare_observable_game(g, 0)
                self.assertFalse(sim_inline._p_flags[0] & _PF_HAS_COMPLEX_END_DAY)
                hand_len_before = len(sim_inline.players[0].hand)
                inline_val = ai._evaluate_terminal_day(sim_inline, 0)
                self.assertEqual(len(sim_inline.players[0].hand), hand_len_before)

                sim_full = ai._prepare_observable_game(g, 0)
                sim_full._p_flags[0] |= _PF_HAS_COMPLEX_END_DAY
                full_val = ai._evaluate_terminal_day(sim_full, 0)
                self.assertAlmostEqual(inline_val, full_val, places=6)

                if case_name == "scout_other_cottages":
                    g_opp = _osmium_game(seed=201)
                    g_opp.day = 3
                    g_opp.starting_player = 1
                    g_opp.suitcase = [
                        make_card("movie"),
                        make_card("zumba"),
                        make_card("nap"),
                        make_card("solo"),
                    ]
                    g_opp.trunk = [make_card("calzones") for _ in range(12)]
                    g_opp._is_setup = True
                    p0 = g_opp.players[0]
                    p0.hand = [make_card("nap")]
                    p0.fun = 5
                    sc0 = make_card("scout-other-cottages")
                    om0 = f"_scout_energy_cube_{sc0.instance_id}"
                    sc1 = make_card("scout-other-cottages")
                    om1 = f"_scout_energy_cube_{sc1.instance_id}"
                    g_opp.suitcase[0].markers["energy_cube"] = True
                    g_opp.suitcase[0].markers[om0] = True
                    g_opp.suitcase[0].markers[om1] = True
                    g_opp.suitcase[1].markers[om0] = True
                    p0.played_today = [sc0, make_card("movie")]
                    g_opp.players[1].played_today = [sc1]
                    for idx_o in (1, 2, 3):
                        g_opp.players[idx_o].asleep = True
                    sim_c = ai._prepare_observable_game(g_opp, 0)
                    c_val = ai._evaluate_terminal_day(sim_c, 0)
                    sim_cf = ai._prepare_observable_game(g_opp, 0)
                    sim_cf._p_flags[0] |= _PF_HAS_COMPLEX_END_DAY
                    cf_val = ai._evaluate_terminal_day(sim_cf, 0)
                    self.assertAlmostEqual(c_val, cf_val, places=6)

    def test_state_signature_respects_markers_and_order(self):
        ai = OsmiumAI(random.Random(202))
        g = _osmium_game(seed=202)
        g.day = 3
        g.suitcase = [make_card("nap") for _ in range(4)]
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True

        # 1. Positional order in played_today is preserved by _state_signature
        o1 = ai._prepare_observable_game(g, 0)
        o1.players[0].played_today = [make_card("nap"), make_card("azul")]
        o1._recompute_player_flags(0)
        o2 = ai._prepare_observable_game(g, 0)
        o2.players[0].played_today = [make_card("azul"), make_card("nap")]
        o2._recompute_player_flags(0)
        self.assertNotEqual(ai._state_signature(o1, 0), ai._state_signature(o2, 0))

        # 2. _SIG_MARKER_KEYS force detailed marker distinction even with same slugs
        m1 = ai._prepare_observable_game(g, 0)
        q1 = make_card("quacks")
        q1.markers["energy_cubes"] = 1
        m1.players[0].played_today = [q1]
        m1._recompute_player_flags(0)
        m2 = ai._prepare_observable_game(g, 0)
        q2 = make_card("quacks")
        q2.markers["energy_cubes"] = 2
        m2.players[0].played_today = [q2]
        m2._recompute_player_flags(0)
        self.assertNotEqual(ai._state_signature(m1, 0), ai._state_signature(m2, 0))

        # 3. Falsy non-sig marker (e.g. _gave_energy only) uses packed scalar fast path
        g1 = ai._prepare_observable_game(g, 0)
        n1 = make_card("nap")
        n1.markers["_gave_energy"] = True
        g1.players[0].played_today = [n1]
        g1._recompute_player_flags(0)
        g2 = ai._prepare_observable_game(g, 0)
        g2.players[0].played_today = [make_card("nap")]
        g2._recompute_player_flags(0)
        self.assertEqual(ai._state_signature(g1, 0), ai._state_signature(g2, 0))
        self.assertIsInstance(ai._state_signature(g1, 0)[2], int)

    def test_search_pure_commutative_credits_all_first_actions_in_live_play(self):
        ai = OsmiumAI(random.Random(303))
        g = _osmium_game(seed=303)
        g.day = 6
        g.suitcase = [make_card("nap") for _ in range(4)]
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True
        p = g.players[0]
        p.energy = 7
        p.hand = [
            make_card("splendor"),
            make_card("azul"),
            make_card("cheap-white"),
        ]
        sim = ai._prepare_observable_game(g, 0)
        _, best_val, table = ai._search_best_day_plan_with_table(sim, 0)
        self.assertAlmostEqual(best_val, 9.0, places=6)
        self.assertAlmostEqual(table[0][0], best_val, places=6)
        self.assertAlmostEqual(table[1][0], best_val, places=6)
        self.assertAlmostEqual(table[2][0], best_val, places=6)

    def test_search_best_day_plan_subclass_without_need_table(self):
        class _SubHydrogen(HydrogenAI):
            def _search_best_day_plan_with_table(
                self,
                base_sim,
                player_index,
                *,
                beam_width=12,
                max_plays=None,
                allow_bed_now=True,
                allowed_first_indices=None,
                base_turns_offset=0,
                root_skipped_override=None,
                start_extra_active=False,
            ):
                return super()._search_best_day_plan_with_table(
                    base_sim,
                    player_index,
                    beam_width=beam_width,
                    max_plays=max_plays,
                    allow_bed_now=allow_bed_now,
                    allowed_first_indices=allowed_first_indices,
                    base_turns_offset=base_turns_offset,
                    root_skipped_override=root_skipped_override,
                    start_extra_active=start_extra_active,
                )

        class _SubOsmium(OsmiumAI):
            def _search_best_day_plan_with_table(
                self,
                base_sim,
                player_index,
                *,
                beam_width=12,
                max_plays=None,
                allow_bed_now=True,
                allowed_first_indices=None,
                base_turns_offset=0,
                root_skipped_override=None,
                start_extra_active=False,
            ):
                return super()._search_best_day_plan_with_table(
                    base_sim,
                    player_index,
                    beam_width=beam_width,
                    max_plays=max_plays,
                    allow_bed_now=allow_bed_now,
                    allowed_first_indices=allowed_first_indices,
                    base_turns_offset=base_turns_offset,
                    root_skipped_override=root_skipped_override,
                    start_extra_active=start_extra_active,
                )

        for ai in (_SubHydrogen(random.Random(305)), _SubOsmium(random.Random(306))):
            g = _osmium_game(seed=307)
            g.day = 6
            g.suitcase = [make_card("nap") for _ in range(4)]
            g.trunk = [make_card("calzones") for _ in range(12)]
            g._is_setup = True
            g.players[0].energy = 7
            g.players[0].hand = [make_card("waterski"), make_card("movie")]
            sim = ai._prepare_observable_game(g, 0)
            act, val = ai._search_best_day_plan(sim, 0)
            self.assertIsNotNone(act)
            self.assertGreater(val, 0.0)

    def test_awake_opp_new_cards_adjustment_cached_parity(self):
        ai = OsmiumAI(random.Random(308))
        g = _osmium_game(seed=308)
        g.day = 3
        g.suitcase = [make_card("nap") for _ in range(4)]
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True
        g._fun_at_start_of_day = [0, 0, 0, 0]

        ep = make_card("epic-prank")
        ep.markers["target_player_index"] = 2
        g.players[0].energy = 0
        g.players[0].hand = [make_card("splendor"), make_card("movie")]
        g.players[0].played_today = [
            make_card("7-wonders"),
            make_card("azul"),
            make_card("solo"),
            ep,
        ]
        g.players[1].played_today = [make_card("fit-to-print")]
        g.players[2].played_today = [make_card("nap")]
        g.players[3].played_today = [make_card("nap")]

        sim = ai._prepare_observable_game(g, 0)
        next_tom = [ep]
        p0 = sim.players[0]
        scored0, _ = sim.score_played_today_fast(0)
        start_f = sim._fun_at_start_of_day[0]
        my_df = float((p0.fun - start_f) + scored0)

        uncached = _awake_opp_new_cards_adjustment(sim, 0, next_tom)
        opp_cache: dict = {}
        cached_1 = _awake_opp_new_cards_adjustment(
            sim,
            0,
            next_tom,
            _my_day_fun=my_df,
            _opp_cache=opp_cache,
            _known_has_target=True,
        )
        cached_2 = _awake_opp_new_cards_adjustment(
            sim,
            0,
            next_tom,
            _my_day_fun=my_df,
            _opp_cache=opp_cache,
            _known_has_target=True,
        )
        self.assertAlmostEqual(cached_1, uncached, places=9)
        self.assertAlmostEqual(cached_2, uncached, places=9)
        self.assertNotIn(("odf", 1), opp_cache)
        self.assertIn(("odf", 2), opp_cache)
        self.assertIn(("opr", 2), opp_cache)

    def test_suitcase_pick1_edge_mixed_sc_plain_parity(self):
        ai = OsmiumAI(random.Random(309))
        g = _osmium_game(seed=309)
        g.day = 3
        sc_marked = make_card("azul")
        sc_marked.markers["_all_tags"] = True
        g.suitcase = [
            make_card("nap"),
            make_card("istanbul"),
            sc_marked,
            make_card("splendor"),
        ]
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True
        p0 = g.players[0]
        p0.hand = [make_card("splendor"), make_card("movie")]
        p0.played_today = [make_card("waterski")]
        p0.tomorrow_cards = [make_card("azul")]

        sim = ai._prepare_observable_game(g, 0)
        sc = sim.suitcase
        sc_metas = tuple(_HYDROGEN_META_BY_SLUG[c.definition.slug] for c in sc)
        sc_costs = tuple(sim.cards_energy_costs(0, sc))
        plain_list = []
        for sc_c, m, c in zip(sc, sc_metas, sc_costs):
            cm = sc_c.markers
            if (
                not cm
                or (
                    ("_all_tags" not in cm or not cm["_all_tags"])
                    and ("_extra_tags" not in cm or not cm["_extra_tags"])
                )
            ) and not m.q_early_flags and not m.q_late_flags and m.h_qcode == _HQ_NONE:
                plain_list.append(
                    (
                        (m.q_fun_term - c * 0.78)
                        + m.q_imm_e_term
                        + (
                            0.65
                            if (c == 0 and m.base_fun >= 0)
                            else (0.25 if (c == 1 and m.base_fun >= 0) else 0.0)
                        ),
                        c <= 1,
                        m.must_play_first,
                        m.tag_mask,
                        m.sorted_tag_bits,
                    )
                )
            else:
                plain_list.append(None)
        mixed_plain = tuple(plain_list)
        self.assertIsNone(mixed_plain[1])
        self.assertIsNone(mixed_plain[2])
        self.assertIsNotNone(mixed_plain[0])
        self.assertIsNotNone(mixed_plain[3])

        ref = _suitcase_pick1_edge(sim, 0)
        mixed = _suitcase_pick1_edge(
            sim,
            0,
            _sc_costs=sc_costs,
            _sc_metas=sc_metas,
            _sc_plain=mixed_plain,
        )
        self.assertAlmostEqual(mixed, ref, places=9)

    def test_mid_day_root_pure_comm_credits_all_first_actions(self):
        ai = OsmiumAI(random.Random(310))
        g = _osmium_game(seed=310)
        g.day = 6
        g.suitcase = [make_card("nap") for _ in range(4)]
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True
        p = g.players[0]
        p.energy = 7
        p.played_today = [make_card("nap")]
        p.hand = [
            make_card("splendor"),
            make_card("azul"),
            make_card("cheap-white"),
        ]
        sim = ai._prepare_observable_game(g, 0)
        _, best_val, table = ai._search_best_day_plan_with_table(sim, 0)
        self.assertAlmostEqual(
            best_val, 9.0 + p.played_today[0].definition.base_fun, places=6
        )
        self.assertAlmostEqual(table[0][0], best_val, places=6)
        self.assertAlmostEqual(table[1][0], best_val, places=6)
        self.assertAlmostEqual(table[2][0], best_val, places=6)

        g_wc = _osmium_game(seed=311)
        g_wc.day = 6
        g_wc.suitcase = [make_card("nap") for _ in range(4)]
        g_wc.trunk = [make_card("calzones") for _ in range(12)]
        g_wc._is_setup = True
        pw = g_wc.players[0]
        pw.energy = 7
        pw.played_today = [make_card("work-call")]
        pw.hand = [make_card("splendor"), make_card("azul")]
        sim_wc = ai._prepare_observable_game(g_wc, 0)
        act_wc, _, table_wc = ai._search_best_day_plan_with_table(sim_wc, 0)
        self.assertEqual(act_wc, 1)
        self.assertGreater(table_wc[1][0], table_wc[0][0] + 1.0)

        g_bv = _osmium_game(seed=312)
        g_bv.day = 6
        g_bv.suitcase = [make_card("nap") for _ in range(4)]
        g_bv.trunk = [make_card("calzones") for _ in range(12)]
        g_bv._is_setup = True
        pb = g_bv.players[0]
        pb.energy = 7
        pb.played_today = [make_card("beach-volleyball")]
        pb.hand = [make_card("nap"), make_card("splendor")]
        sim_bv = ai._prepare_observable_game(g_bv, 0)
        _, _, table_bv = ai._search_best_day_plan_with_table(sim_bv, 0)
        _, v0, _ = ai._search_best_day_plan_with_table(
            sim_bv, 0, allowed_first_indices=[0]
        )
        _, v1, _ = ai._search_best_day_plan_with_table(
            sim_bv, 0, allowed_first_indices=[1]
        )
        self.assertAlmostEqual(table_bv[0][0], v0, places=6)
        self.assertAlmostEqual(table_bv[1][0], v1, places=6)

    def test_continuation_potential_cached_day_fun_and_opp_cache_parity(self):
        ai = OsmiumAI(random.Random(313))
        g = _osmium_game(seed=313)
        g.day = 3
        g.suitcase = [make_card("nap") for _ in range(4)]
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True
        g._fun_at_start_of_day = [0, 0, 0, 0]

        ep_uncached = make_card("epic-prank")
        ep_uncached.markers["target_player_index"] = 1
        ep_cached = make_card("epic-prank")
        ep_cached.markers["target_player_index"] = 2

        p0 = g.players[0]
        p0.energy = 3
        p0.played_today = [make_card("dates-first-nos"), ep_uncached]
        p0.hand = [make_card("splendor"), make_card("waterski")]
        g.players[1].played_today = [make_card("fit-to-print")]
        g.players[2].played_today = [make_card("azul")]

        sim = ai._prepare_observable_game(g, 0)
        scored0, _ = sim.score_played_today_fast(0)
        cur_df = float((sim.players[0].fun - sim._fun_at_start_of_day[0]) + scored0)
        ref = ai._continuation_potential(sim, 0)
        opp_cache: dict = {}
        fast1 = ai._continuation_potential(
            sim, 0, _cur_day_fun=cur_df, _opp_cache=opp_cache
        )
        self.assertAlmostEqual(fast1, ref, places=9)
        self.assertNotIn(("opr", 1), opp_cache)

        p0.played_today = [make_card("dates-first-nos"), ep_cached]
        sim2 = ai._prepare_observable_game(g, 0)
        scored2, _ = sim2.score_played_today_fast(0)
        cur_df2 = float((sim2.players[0].fun - sim2._fun_at_start_of_day[0]) + scored2)
        ref2 = ai._continuation_potential(sim2, 0)
        fast2 = ai._continuation_potential(
            sim2, 0, _cur_day_fun=cur_df2, _opp_cache=opp_cache
        )
        fast3 = ai._continuation_potential(
            sim2, 0, _cur_day_fun=cur_df2, _opp_cache=opp_cache
        )
        self.assertAlmostEqual(fast2, ref2, places=9)
        self.assertAlmostEqual(fast3, ref2, places=9)
        self.assertIn(("opr", 2), opp_cache)

    def test_order_cards_for_trunk_fast_ctx_parity(self):
        ai = OsmiumAI(random.Random(314))
        trunk_slugs = ("nap", "splendor", "waterski", "calzones", "azul")
        for hand_slugs in (
            ("waterski",),
            ("assorted-cutlery", "waterski"),
            ("bougie-coffee-machine", "azul"),
            ("quacks", "waterski"),
            ("assorted-cutlery", "quacks"),
        ):
            g = _osmium_game(seed=314)
            g.day = 3
            g.suitcase = [make_card("nap") for _ in range(4)]
            g.trunk = [make_card("calzones") for _ in range(12)]
            g._is_setup = True
            p0 = g.players[0]
            p0.energy = 7
            p0.hand = [make_card(s) for s in hand_slugs]
            p0.played_today = [make_card("nap")]
            cards = [make_card(s) for s in trunk_slugs]
            sim = ai._prepare_observable_game(g, 0)
            # Clear known_trunk_state side-effect differences cleanly on each fresh call.
            ref_order = ai.order_cards_for_trunk(g, 0, cards)
            fast_order = ai.order_cards_for_trunk(sim, 0, cards)
            self.assertEqual(fast_order, ref_order)

    def test_score_played_today_fast_epic_prank_tomorrow_parity(self):
        ai = OsmiumAI(random.Random(315))
        tom_combos = (
            ("epic-prank",),
            ("epic-prank", "dates-first-nos"),
            ("dates-first-nos", "epic-prank"),
            ("epic-prank", "wildlife-spotting", "fancy-floatie"),
            ("wildlife-spotting", "fancy-floatie", "epic-prank"),
        )
        played_boards = (
            # Path 1: zero fun mods
            ("waterski", "azul"),
            # Path 2 single card
            ("waterski",),
            # Path 2 additive with before/after modifiers (Campfire before/after must not be doubled by Epic Prank)
            ("waterski", "campfire", "calzones"),
            # Path 3 non-additive today_mods
            ("work-call", "waterski", "bring-a-friend", "azul"),
        )
        for char in (None, next(c for c in CHARACTERS if c.zero_fun_tag == "Food")):
            for tom_slugs in tom_combos:
                for p_slugs in played_boards:
                    g = _osmium_game(seed=315)
                    g.day = 4
                    g.suitcase = [make_card("nap") for _ in range(4)]
                    g.trunk = [make_card("calzones") for _ in range(12)]
                    g._is_setup = True
                    p0 = g.players[0]
                    p0.character = char
                    p0.tomorrow_cards = [make_card(s) for s in tom_slugs]
                    p0.played_today = [make_card(s) for s in p_slugs]
                    sim = ai._prepare_observable_game(g, 0)
                    fast_score, fast_tom = sim.score_played_today_fast(0)
                    ref_score = sum(
                        sim.card_fun(0, pc) for pc in sim.players[0].played_today
                    )
                    self.assertEqual(fast_score, ref_score)
                    ref_tom = [
                        pc
                        for pc in sim.players[0].played_today
                        if pc.effective_behavior.has_tomorrow_action
                    ]
                    self.assertEqual(fast_tom, ref_tom)

    def test_score_played_today_fast_bubly_and_cheesy_phone_game_parity(self):
        ai = OsmiumAI(random.Random(316))
        # Test Bubly across no-energy-before, energy-before, energy-after, copied, and additive/non-add boards
        for p_slugs, ge_positions in (
            (("bubly", "fajitas"), (1,)),
            (("fajitas", "bubly"), (0,)),
            (("fajitas", "bubly", "campfire"), (0,)),
            (("work-call", "fajitas", "bubly"), (1,)),
            (("bubly",), ()),
        ):
            g = _osmium_game(seed=316)
            g.day = 3
            g.suitcase = [make_card("nap") for _ in range(4)]
            g.trunk = [make_card("calzones") for _ in range(12)]
            g._is_setup = True
            p0 = g.players[0]
            p0.played_today = [make_card(s) for s in p_slugs]
            for pos in ge_positions:
                p0.played_today[pos].markers["_gave_energy"] = True
            sim = ai._prepare_observable_game(g, 0)
            fast_score, _ = sim.score_played_today_fast(0)
            ref_score = sum(
                sim.card_fun(0, pc) for pc in sim.players[0].played_today
            )
            self.assertEqual(fast_score, ref_score)

        # Test Cheesy Phone Game with 0/1/2 Relax opponents (including _all_tags marker) and copied Leftovers
        for relax_cnt in (0, 1, 2):
            g = _osmium_game(seed=317)
            g.day = 3
            g.suitcase = [make_card("nap") for _ in range(4)]
            g.trunk = [make_card("calzones") for _ in range(12)]
            g._is_setup = True
            cpg = make_card("cheesy-phone-game")
            g.players[0].played_today = [cpg, make_card("campfire")]
            for o_i in range(1, 1 + relax_cnt):
                if o_i == 2:
                    all_tag_c = make_card("nap")
                    all_tag_c.markers["_all_tags"] = True
                    g.players[o_i].played_today = [all_tag_c]
                else:
                    g.players[o_i].played_today = [make_card("movie")]
            sim = ai._prepare_observable_game(g, 0)
            fast_score, _ = sim.score_played_today_fast(0)
            ref_score = sum(
                sim.card_fun(0, pc) for pc in sim.players[0].played_today
            )
            self.assertEqual(fast_score, ref_score)

    def test_safe_prefix_seen_pre_pure_parity_and_boundaries(self):
        ai = OsmiumAI(random.Random(318))
        # Safe prefix after Fajitas collapses pure-comm tails while preserving exact value and credits
        g = _osmium_game(seed=318)
        g.day = 6
        g.suitcase = [make_card("nap") for _ in range(4)]
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True
        p = g.players[0]
        p.energy = 7
        p.played_today = [make_card("fajitas")]
        p.hand = [make_card("splendor"), make_card("azul"), make_card("cheap-white")]
        sim = ai._prepare_observable_game(g, 0)
        _, best_val, table = ai._search_best_day_plan_with_table(sim, 0)
        self.assertAlmostEqual(best_val, 9.0 + make_card("fajitas").definition.base_fun, places=6)
        self.assertAlmostEqual(table[0][0], best_val, places=6)
        self.assertAlmostEqual(table[1][0], best_val, places=6)
        self.assertAlmostEqual(table[2][0], best_val, places=6)

        # Boundary: Swimming in hand checks previous card tags -> must NOT collapse [A, B] vs [B, A]
        g_swim = _osmium_game(seed=319)
        g_swim.day = 6
        g_swim.suitcase = [make_card("nap") for _ in range(4)]
        g_swim.trunk = [make_card("calzones") for _ in range(12)]
        g_swim._is_setup = True
        ps = g_swim.players[0]
        ps.energy = 6
        ps.hand = [make_card("movie"), make_card("waterski"), make_card("swimming")]
        sim_s = ai._prepare_observable_game(g_swim, 0)
        _, v_sw, tab_sw = ai._search_best_day_plan_with_table(sim_s, 0)
        _, v_sw_0, _ = ai._search_best_day_plan_with_table(sim_s, 0, allowed_first_indices=[0])
        _, v_sw_1, _ = ai._search_best_day_plan_with_table(sim_s, 0, allowed_first_indices=[1])
        self.assertAlmostEqual(tab_sw[0][0], v_sw_0, places=6)
        self.assertAlmostEqual(tab_sw[1][0], v_sw_1, places=6)
        self.assertGreater(v_sw, 0.0)

    def test_expanded_sc_plain_specs_all_eligible_cards_parity(self):
        ai = OsmiumAI(random.Random(320))
        _ensure_osmium_caches()
        g = _osmium_game(seed=320)
        g.day = 3
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True
        p = g.players[0]
        p.energy = 2
        p.hand = [make_card("sweet-lawn-chair"), make_card("azul"), make_card("movie")]
        p.played_today = [make_card("campfire")]
        p.tomorrow_cards = [make_card("fancy-floatie")]
        sim = ai._prepare_observable_game(g, 0)
        wanted, ctx = _build_wanted_and_ctx(sim.players[0], True)
        self.assertTrue(wanted)

        eligible_cnt = 0
        for slug in sorted(CARD_REGISTRY):
            m = _HYDROGEN_META_BY_SLUG[slug]
            if not (
                not m.has_own_cost_mod
                and m.title != "Istanbul"
                and m.quick_sleep_or_late != 2
                and m.cost < 8
                and m.prof.conditional_energy_gain <= 0
                and m.c_qcode in _SC_PLAIN_CQ_CODES
                and (
                    m.quick_syn_kind == _QSYN_NONE
                    or m.quick_syn_kind == _QSYN_NEG_SMB
                )
                and (
                    m.quick_global_kind == _QGLOB_NONE
                    or m.quick_global_kind == _QGLOB_DOCK
                    or m.quick_global_kind == _QGLOB_COUCH_TUBE
                )
                and m.h_qcode == _HQ_NONE
            ):
                continue
            eligible_cnt += 1
            sc_c = make_card(slug)
            c = sim.energy_cost(0, sc_c)
            base_v = (m.q_fun_term - c * 0.78) + m.q_imm_e_term
            if m.has_quick_draw:
                base_v += m.quick_draw_cheap if c <= 2 else m.quick_draw_exp
            if m.quick_sleep_or_late == 1:
                base_v -= 1.6
            if m.c_qcode == _CQ_GROUP_DINNER:
                base_v -= 1.85
            elif m.c_qcode == _CQ_BOUGIE_COFFEE_MACHINE:
                base_v += 1.35
            elif m.c_qcode == _CQ_CHEESY_PHONE_GAME:
                base_v += 1.05
            elif m.conditional_fun_bonus > 0 and not m.is_multi_scaler:
                cfb = m.conditional_fun_bonus
                base_v += (cfb if cfb < 5 else 5) * 0.55
            if m.quick_penalty != 0.0:
                base_v -= m.quick_penalty
            if m.quick_syn_kind == _QSYN_NEG_SMB:
                base_v += m.quick_neg_smb_pen
            if m.quick_global_kind == _QGLOB_DOCK:
                base_v += 3.5
            elif m.quick_global_kind == _QGLOB_COUCH_TUBE:
                base_v += 3.1
            if m.goes_to_tomorrow:
                base_v += m.quick_tom_base
                if m.quick_tom_extra != 0.0:
                    base_v += m.quick_tom_extra
            if c <= 1 and m.base_fun >= 0:
                base_v += 0.65 if c == 0 else 0.25
            sp = (base_v, c <= 1, m.must_play_first, m.tag_mask, m.sorted_tag_bits)
            fast_v = _eval_plain_sc_spec(sp, ctx.w_arr, ctx.w_mask)
            ref_v = _hydrogen_quick_card_desirability(
                sim, 0, sc_c, wanted, cost=c, _ctx=ctx, _in_hand=False, _meta=m
            )
            self.assertAlmostEqual(fast_v, ref_v, places=12, msg=f"slug={slug}")

        self.assertGreaterEqual(eligible_cnt, 80)

    def test_hydrogen_eval_ai_choose_card_to_discard_fast_paths(self):
        eval_ai = _HydrogenEvalAI()
        ai = OsmiumAI(random.Random(321))
        g = _osmium_game(seed=321)
        g.day = 3
        g.suitcase = [make_card("nap") for _ in range(4)]
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True
        p0 = g.players[0]
        p0.energy = 3

        # 1. Single option returns 0
        self.assertEqual(eval_ai.choose_card_to_discard(g, 0, [make_card("waterski")]), 0)

        # 2. Tell a Story without Cutlery/BCM skips trunk setup and selects lowest keep value
        p0.played_today = [make_card("tell-a-story")]
        p0.hand = [make_card("waterski"), make_card("nap"), make_card("classic-book")]
        sim = ai._prepare_observable_game(g, 0)
        sim_p = sim.players[0]
        self.assertEqual(
            eval_ai.choose_card_to_discard(sim, 0, sim_p.hand),
            eval_ai.choose_card_to_discard(g, 0, p0.hand),
        )

    def test_q_const_eligible_all_catalog_cards_bit_exact_parity(self):
        ai = OsmiumAI(random.Random(322))
        eval_ai = _HydrogenEvalAI()
        _ensure_osmium_caches()

        for day in (2, 6):
            for with_wanted in (False, True):
                g = _osmium_game(seed=322)
                g.day = day
                g.suitcase = [make_card("nap") for _ in range(4)]
                g.trunk = [make_card("calzones") for _ in range(12)]
                g._is_setup = True
                p0 = g.players[0]
                p0.energy = 3
                if with_wanted:
                    p0.hand = [make_card("sweet-lawn-chair"), make_card("azul")]
                    p0.played_today = [make_card("campfire")]
                    p0.tomorrow_cards = [make_card("fancy-floatie")]
                else:
                    p0.hand = [make_card("nap")]
                sim = ai._prepare_observable_game(g, 0)
                wanted_dict, ctx = _build_wanted_and_ctx(
                    sim.players[0], True, _build_dict=True
                )
                _, ctx_nodict = _build_wanted_and_ctx(
                    sim.players[0], True, _build_dict=False
                )
                eligible_cnt = 0
                for slug in sorted(CARD_REGISTRY):
                    m = _HYDROGEN_META_BY_SLUG[slug]
                    if not m.q_const_eligible:
                        continue
                    eligible_cnt += 1
                    c_fast = make_card(slug)
                    fast_arr = _eval_q_const_with_arr(
                        m, ctx_nodict.w_arr, ctx_nodict.w_mask, day >= 6
                    )
                    fast_api = _hydrogen_quick_card_desirability(
                        sim,
                        0,
                        c_fast,
                        wanted_dict,
                        cost=m.cost,
                        _ctx=ctx,
                        _in_hand=False,
                        _meta=m,
                    )
                    # Distinct CardSynergyProfile identity forces unoptimized slow path
                    slow_ref = _hydrogen_quick_card_desirability(
                        sim,
                        0,
                        c_fast,
                        wanted_dict,
                        cost=m.cost,
                        prof=replace(m.prof),
                        _in_hand=False,
                    )
                    self.assertEqual(fast_arr, slow_ref, msg=f"{slug} d={day}")
                    self.assertEqual(fast_api, slow_ref, msg=f"{slug} d={day}")
                self.assertGreaterEqual(eligible_cnt, 80)

        # Also check _HydrogenEvalAI choose_suitcase_card & Day-6 choose_card_to_discard
        for day, wt_slug, energy in ((3, None, 2), (6, None, 2), (6, "wrestle-the-kids", 1)):
            g2 = _osmium_game(seed=323)
            g2.day = day
            g2.trunk = [make_card("calzones") for _ in range(12)]
            g2._is_setup = True
            p0 = g2.players[0]
            p0.energy = energy
            hand_slugs = ["floating", "johnny-appleseed", "azul", "morning-run"]
            if wt_slug:
                hand_slugs.append(wt_slug)
            p0.hand = [make_card(s) for s in hand_slugs]
            p0.played_today = [make_card("campfire")]
            g2.suitcase = [
                make_card("floating"),
                make_card("doxology"),
                make_card("azul"),
                make_card("waterski"),
            ]
            sim2 = ai._prepare_observable_game(g2, 0)
            self.assertEqual(
                eval_ai.choose_suitcase_card(sim2, 0, sim2.suitcase),
                eval_ai.choose_suitcase_card(g2, 0, g2.suitcase),
            )
            self.assertEqual(
                eval_ai.choose_card_to_discard(sim2, 0, sim2.players[0].hand),
                eval_ai.choose_card_to_discard(g2, 0, p0.hand),
            )

    def test_istanbul_cache_tomorrow_gave_energy_parity(self):
        ai = OsmiumAI(random.Random(324))
        _HYDROGEN_ISTANBUL_CACHE.clear()
        g = _osmium_game(seed=324)
        g.day = 3
        g.suitcase = [make_card("nap") for _ in range(4)]
        g.trunk = [make_card("calzones") for _ in range(12)]
        g._is_setup = True
        p0 = g.players[0]
        p0.energy = 2
        mb = make_card("morning-bike")
        mb.markers["_gave_energy"] = True
        fl = make_card("floating")
        fl.markers["_gave_energy"] = True
        p0.tomorrow_cards = [mb, fl]
        p0.played_today = [make_card("istanbul")]
        p0.hand = [make_card("nap"), make_card("waterski"), make_card("azul")]

        sim = ai._prepare_observable_game(g, 0)
        res1 = _evaluate_istanbul_discards_fast(sim, 0, sim.players[0].hand)
        self.assertGreater(len(_HYDROGEN_ISTANBUL_CACHE), 0)
        cache_len = len(_HYDROGEN_ISTANBUL_CACHE)
        # Second call with simple tomorrow markers hits cache cleanly and matches vanilla fallback
        res2 = _evaluate_istanbul_discards_fast(sim, 0, sim.players[0].hand)
        self.assertEqual(len(_HYDROGEN_ISTANBUL_CACHE), cache_len)
        self.assertEqual(res1, res2)

        # Non-simple marker on tomorrow card bypasses cache cleanly
        ep = make_card("epic-prank")
        ep.markers["target_player_index"] = 1
        p0.tomorrow_cards = [ep]
        sim_ns = ai._prepare_observable_game(g, 0)
        res_ns = _evaluate_istanbul_discards_fast(sim_ns, 0, sim_ns.players[0].hand)
        self.assertEqual(len(_HYDROGEN_ISTANBUL_CACHE), cache_len)
        self.assertTrue(res_ns)

    def test_cities_and_knights_inline_end_day_parity(self):
        ai = OsmiumAI(random.Random(325))
        cases = (
            # (day, p0_played_slugs, p1_played_slugs)
            (3, ("cities-and-knights", "waterski"), ()),
            (3, ("cities-and-knights", "nap", "movie", "waterski"), ()),
            (3, ("cities-and-knights", "azul", "splendor", "campfire", "waterski"), ()),
            (6, ("cities-and-knights", "azul", "splendor", "campfire", "waterski"), ()),
            (3, ("scout-other-cottages",), ("cities-and-knights", "azul", "splendor", "campfire", "waterski")),
        )
        for day, p0_slugs, p1_slugs in cases:
            g = _osmium_game(seed=325)
            g.day = day
            g.suitcase = [make_card("calzones"), make_card("nap"), make_card("waterski"), make_card("movie")]
            g.trunk = [make_card("azul") for _ in range(12)]
            g._is_setup = True
            g.players[0].played_today = [make_card(s) for s in p0_slugs]
            g.players[1].played_today = [make_card(s) for s in p1_slugs]
            sim = ai._prepare_observable_game(g, 0)
            sim._capture_day_start_fun()
            if "scout-other-cottages" in p0_slugs:
                sc_c = sim.suitcase[0]
                sc_c.markers[f"_scout_energy_cube_{sim.players[0].played_today[0].instance_id}"] = True
            fast_v = ai._evaluate_terminal_day(sim, 0)

            # Force full end_day reference using _has_card_ref / non-hsim Game clone
            ref_sim = Game(
                create_default_deck(),
                [OsmiumAI(random.Random(900 + i)) for i in range(4)],
                random.Random(901),
            )
            ref_sim.day = day
            ref_sim._is_setup = True
            ref_sim.suitcase = [CardInstance(c.instance_id, c.definition, c.is_tomorrow, dict(c.markers)) for c in sim.suitcase]
            ref_sim.trunk = [CardInstance(c.instance_id, c.definition) for c in sim.trunk]
            for i in range(4):
                ref_sim.players[i].played_today = [
                    CardInstance(c.instance_id, c.definition, c.is_tomorrow, dict(c.markers))
                    for c in sim.players[i].played_today
                ]
            ref_sim._capture_day_start_fun()
            ref_v = ai._evaluate_terminal_day(ref_sim, 0)
            self.assertAlmostEqual(fast_v, ref_v, places=9, msg=f"case={(day, p0_slugs, p1_slugs)}")

    def test_euchre_awards_ceremony_fast_on_card_play_parity(self):
        for follow_slug, all_tags in (("calzones", False), ("azul", False), ("waterski", False), ("nap", True)):
            g_van = _osmium_game(seed=326)
            g_van.day = 3
            g_van._is_setup = True
            g_van.suitcase = [make_card("splendor"), make_card("movie"), make_card("campfire"), make_card("nap")]
            g_van.trunk = [make_card("calzones") for _ in range(10)]
            g_van.players[0].energy = 7
            g_van.players[0].played_today = [make_card("euchre-awards-ceremony")]
            follow = make_card(follow_slug)
            if all_tags:
                follow.markers["_all_tags"] = True
            g_van.players[0].hand = [follow]

            ai = OsmiumAI(random.Random(326))
            sim = ai._prepare_observable_game(g_van, 0)
            g_van.play_card(0, 0)
            sim.play_card(0, 0)

            self.assertEqual(
                [c.definition.slug for c in sim.players[0].hand],
                [c.definition.slug for c in g_van.players[0].hand],
            )
            self.assertEqual(
                [c.definition.slug for c in sim.suitcase],
                [c.definition.slug for c in g_van.suitcase],
            )

    def test_inline_classic_book_sing_song_and_continuation_fast_path(self):
        ai = OsmiumAI(random.Random(327))
        # Classic Book & Sing Song cost parity (with Classic Book in hand vs played, and _extra_tags/_all_tags active)
        for played_slugs, opp_slugs, use_extra_tags in (
            (("movie", "campfire"), ("campfire",), False),
            (("classic-book", "movie"), ("waterski",), False),
            (("colouring", "waterski"), ("waterski",), True),
        ):
            g = _osmium_game(seed=327)
            g.day = 3
            g._is_setup = True
            g.suitcase = [make_card("nap") for _ in range(4)]
            g.trunk = [make_card("calzones") for _ in range(10)]
            g.players[0].energy = 7
            cb = make_card("classic-book")
            ss = make_card("sing-song")
            g.players[0].hand = [cb, ss]
            g.players[0].played_today = [make_card(s) for s in played_slugs]
            g.players[1].played_today = [make_card(s) for s in opp_slugs]
            if use_extra_tags:
                g._sync_played_today_tags(g.players[0])
                g.players[1].played_today[0].markers["_all_tags"] = True
            sim = ai._prepare_observable_game(g, 0)

            fast_costs = sim.cards_energy_costs(0, sim.players[0].hand)
            ref_costs = [g.energy_cost(0, c) for c in g.players[0].hand]
            self.assertEqual(fast_costs, ref_costs)

            fast_playable = sim.hand_playable_with_costs(0)
            ref_playable = [
                (idx, g.energy_cost(0, g.players[0].hand[idx]))
                for idx in g.playable_hand_indices(0)
            ]
            self.assertEqual(fast_playable, ref_playable)
            if use_extra_tags:
                self.assertEqual(fast_costs[0], 0)
                self.assertEqual(fast_costs[1], make_card("sing-song").definition.cost - 1)

        # Inline _continuation_potential 1-line fast path matches _continuation_potential bit-for-bit
        g_c = _osmium_game(seed=328)
        g_c.day = 3
        g_c._is_setup = True
        g_c.suitcase = [make_card("nap") for _ in range(4)]
        g_c.trunk = [make_card("calzones") for _ in range(10)]
        g_c.players[0].energy = 5
        g_c.players[0].played_today = [make_card("waterski")]
        g_c.players[0].hand = [make_card("azul"), make_card("movie")]
        sim_c = ai._prepare_observable_game(g_c, 0)
        self.assertEqual(
            ai._continuation_potential(sim_c, 0, _has_before_scaler_in_hand=False),
            5 * 0.72,
        )

    def test_wave7_comm_metadata_and_blocked_tag_mask(self):
        _ensure_osmium_caches()
        comm_count = sum(
            1 for slug in CARD_REGISTRY if _HYDROGEN_META_BY_SLUG[slug].comm_kind != _COMM_NONE
        )
        self.assertEqual(comm_count, 54)
        self.assertTrue(_HYDROGEN_META_BY_SLUG["carcassonne"].is_adj_next_tail)
        for non_comm in (
            "doritos",
            "ice-cream-sandwich",
            "new-nos-book-entry",
            "paint-rocks",
            "bubly",
            "sleep-in",
            "carcassonne",
        ):
            self.assertEqual(
                _HYDROGEN_META_BY_SLUG[non_comm].comm_kind,
                _COMM_NONE,
                msg=non_comm,
            )

        # Non-catalog definition reusing a catalog slug must stay non-commutative.
        custom_nap = CardInstance(
            9999,
            CardDefinition(
                slug="nap",
                title="Custom Nap",
                tags=frozenset({"Relax"}),
                cost=0,
                base_fun=1,
                behavior=CARD_REGISTRY["nap"].behavior,
            ),
        )
        self.assertEqual(_get_card_meta(custom_nap).comm_kind, _COMM_NONE)
        custom_7w = CardInstance(
            9998,
            CardDefinition(
                slug="7-wonders",
                title="Custom 7W",
                tags=frozenset({"Board Game"}),
                cost=2,
                base_fun=1,
                behavior=CARD_REGISTRY["7-wonders"].behavior,
            ),
        )
        self.assertEqual(_get_card_meta(custom_7w).comm_kind, _COMM_NONE)

        ai = OsmiumAI(random.Random(701))

        def blocked_for(played_slugs, tom_slugs=()):
            g = _osmium_game(seed=701)
            g.day = 3
            g._is_setup = True
            g.suitcase = [make_card("waterski") for _ in range(4)]
            g.trunk = [make_card("azul") for _ in range(8)]
            g.players[0].played_today = [make_card(s) for s in played_slugs]
            g.players[0].tomorrow_cards = [make_card(s) for s in tom_slugs]
            sim = ai._prepare_observable_game(g, 0)
            return _compute_blocked_tag_mask(
                sim.players[0], sim._p_flags[0], _HYDROGEN_META_BY_SLUG
            )

        self.assertEqual(blocked_for(["waterski", "movie"]), 0)
        self.assertEqual(
            blocked_for(["euchre-awards-ceremony"]),
            _MASK_FOOD | _MASK_BOARD_GAME,
        )
        self.assertEqual(
            blocked_for(["movie"], ["photo-shoot"]),
            _MASK_SOCIAL | _MASK_RELAX,
        )
        self.assertEqual(blocked_for(["water-volleyball"]), _MASK_EXERCISE)
        self.assertEqual(blocked_for(["water-volleyball", "waterski"]), 0)
        self.assertEqual(blocked_for(["schwank"]), _MASK_EXERCISE)
        self.assertEqual(blocked_for(["schwank", "waterski"]), _MASK_EXERCISE)
        self.assertEqual(blocked_for(["schwank", "waterski", "waterski"]), 0)
        self.assertEqual(blocked_for(["after-dinner-entertainment"]), 0)
        self.assertEqual(
            blocked_for(["calzones", "after-dinner-entertainment"]),
            _MASK_EXERCISE,
        )
        self.assertEqual(
            blocked_for(["calzones", "after-dinner-entertainment", "waterski"]),
            0,
        )
        self.assertEqual(blocked_for(["christmas-name-draw"]), 0xFF)
        self.assertEqual(blocked_for(["colouring", "waterski"]), 0)
        self.assertEqual(blocked_for(["colouring"], ["photo-shoot"]), 0xFF)

    def test_wave7_search_dedup_exact_boundaries(self):
        ai = OsmiumAI(random.Random(702))

        def run_day6_search(hand_slugs, energy, suitcase_slugs=None):
            g = _osmium_game(seed=702)
            g.day = 6
            g._is_setup = True
            g.suitcase = [
                make_card(s)
                for s in (suitcase_slugs or ("movie", "nap", "calzones", "azul"))
            ]
            g.trunk = [make_card("splendor") for _ in range(10)]
            g.players[0].energy = energy
            g.players[0].hand = [make_card(s) for s in hand_slugs]
            sim = ai._prepare_observable_game(g, 0)
            sim._capture_day_start_fun()
            return ai._search_best_day_plan_with_table(sim, 0, allow_bed_now=False)

        # Sleep In suppresses Nap energy gain: best plan pays for Waterski (6 Fun), not Spikeball.
        act_si, val_si, _ = run_day6_search(
            ("sleep-in", "nap", "waterski", "spikeball"), energy=3
        )
        self.assertEqual(act_si, 0)
        self.assertAlmostEqual(val_si, 6.0, places=9)

        # Carcassonne must not falsely commute onto a Board Game tail; must sit in the middle (+4).
        act_carc, val_carc, _ = run_day6_search(
            ("carcassonne", "azul", "splendor"), energy=6
        )
        self.assertIn(act_carc, (1, 2))
        self.assertAlmostEqual(val_carc, 11.0, places=9)

        # Nap must play before Bubly so Bubly scores +2, unlocking Waterski (6+2=8 Fun).
        act_bub, val_bub, _ = run_day6_search(
            ("nap", "bubly", "waterski"), energy=4
        )
        self.assertEqual(act_bub, 0)
        self.assertAlmostEqual(val_bub, 8.0, places=9)

        # Euchre Awards Ceremony triggers Suitcase pick when Food follows, beating Calzones-first.
        act_euc, val_euc, _ = run_day6_search(
            ("euchre-awards-ceremony", "calzones", "movie"),
            energy=7,
            suitcase_slugs=("splendor", "nap", "nap", "nap"),
        )
        self.assertEqual(act_euc, 0)
        self.assertAlmostEqual(val_euc, 8.0, places=9)

        # Water Volleyball before Spikeball must preserve the +4 Exercise bonus.
        act_wv10, val_wv10, _ = run_day6_search(
            ("water-volleyball", "spikeball", "movie"), energy=10
        )
        self.assertEqual(act_wv10, 0)
        self.assertAlmostEqual(val_wv10, 13.0, places=9)

        act_wv11, val_wv11, _ = run_day6_search(
            ("water-volleyball", "spikeball", "movie"), energy=11
        )
        self.assertIn(act_wv11, (0, 2))
        self.assertAlmostEqual(val_wv11, 14.0, places=9)

    def test_wave8_inplace_deferred_search_bit_exact_parity(self):
        class _MaterializedRefOsmiumAI(OsmiumAI):
            def _simulate_play_step(
                self, state, player_index, hand_idx, known_cost=None
            ):
                return super()._simulate_play_step(
                    state, player_index, hand_idx, known_cost
                )

        boards = [
            (
                3,
                7,
                ("nap", "waterski", "movie", "outdoor-movie", "azul", "calzones"),
                ("fajitas", "group-dinner"),
                ("waterski", "movie", "splendor", "azul"),
            ),
            (
                6,
                7,
                (
                    "euchre-awards-ceremony",
                    "calzones",
                    "nap",
                    "bubly",
                    "movie",
                    "outdoor-movie",
                    "carcassonne",
                    "azul",
                    "splendor",
                ),
                ("beach-volleyball", "water-volleyball"),
                ("splendor", "nap", "movie", "azul"),
            ),
            (
                4,
                7,
                ("group-dinner", "waterski", "movie", "nap", "swimming"),
                ("hold-the-baby", "cliff-climbing"),
                ("calzones", "splendor", "doxology", "floating"),
            ),
        ]
        for b_idx, (day, energy, hand_slugs, tom_slugs, suit_slugs) in enumerate(boards):
            g_a = _osmium_game(seed=800 + b_idx)
            g_a.day = day
            g_a._is_setup = True
            g_a.suitcase = [make_card(s) for s in suit_slugs]
            g_a.trunk = [make_card("splendor") for _ in range(16)]
            g_a.players[0].energy = energy
            g_a.players[0].hand = [make_card(s) for s in hand_slugs]
            g_a.players[0].tomorrow_cards = [make_card(s) for s in tom_slugs]

            ai_fast = OsmiumAI(random.Random(810 + b_idx))
            ai_ref = _MaterializedRefOsmiumAI(random.Random(810 + b_idx))
            sim_a = ai_fast._prepare_observable_game(g_a, 0)
            sim_a._capture_day_start_fun()
            sim_b = ai_ref._prepare_observable_game(g_a, 0)
            sim_b._capture_day_start_fun()

            snap_hand = [c.definition.slug for c in sim_a.players[0].hand]
            snap_e = sim_a.players[0].energy
            snap_pf = sim_a._p_flags[0]

            act_a, val_a, tbl_a = ai_fast._search_best_day_plan_with_table(sim_a, 0)
            act_b, val_b, tbl_b = ai_ref._search_best_day_plan_with_table(sim_b, 0)

            self.assertEqual([c.definition.slug for c in sim_a.players[0].hand], snap_hand)
            self.assertEqual(sim_a.players[0].played_today, [])
            self.assertEqual(sim_a.players[0].energy, snap_e)
            self.assertEqual(sim_a._p_flags[0], snap_pf)
            self.assertTrue(all(not c.markers for c in sim_a.players[0].hand))

            self.assertEqual(act_a, act_b, msg=f"board {b_idx}")
            self.assertAlmostEqual(val_a, val_b, places=12, msg=f"board {b_idx}")
            self.assertEqual(set(tbl_a), set(tbl_b), msg=f"board {b_idx}")
            for k in tbl_a:
                self.assertAlmostEqual(tbl_a[k][0], tbl_b[k][0], places=12)
                self.assertIsInstance(tbl_a[k][1], Game)
                self.assertEqual(
                    [c.definition.slug for c in tbl_a[k][1].players[0].played_today],
                    [c.definition.slug for c in tbl_b[k][1].players[0].played_today],
                )
                self.assertEqual(
                    tbl_a[k][1].players[0].energy,
                    tbl_b[k][1].players[0].energy,
                )

        # Whole-game parity: fast in-place search must match materialized search bit-for-bit.
        g_fast = Game(
            create_default_deck(),
            [OsmiumAI(random.Random(850 + i)) for i in range(4)],
            random.Random(860),
        )
        g_ref = Game(
            create_default_deck(),
            [_MaterializedRefOsmiumAI(random.Random(850 + i)) for i in range(4)],
            random.Random(860),
        )
        self.assertEqual(g_fast.run().scores, g_ref.run().scores)

    def test_wave8_w_arr_only_and_q_const_suitcase_parity(self):
        ai = OsmiumAI(random.Random(870))
        eval_ai = _HydrogenEvalAI()
        for day, played_slugs in ((3, ()), (6, ("calzones", "waterski", "movie"))):
            g = _osmium_game(seed=870 + day)
            g.day = day
            g._is_setup = True
            g.suitcase = [make_card(s) for s in ("floating", "evening-chat", "sunrise", "paint")]
            g.trunk = [make_card("azul") for _ in range(8)]
            p0 = g.players[0]
            p0.energy = 4
            p0.tomorrow_cards = [make_card("wildlife-spotting")]
            p0.played_today = [make_card(s) for s in played_slugs]
            p0.hand = [make_card("azul"), make_card("hold-the-baby"), make_card("waterski")]

            sim = ai._prepare_observable_game(g, 0)
            w_arr_fast, w_mask_fast = _build_w_arr_only(sim.players[0])
            _, ctx_ref = _build_wanted_and_ctx(sim.players[0], True)
            self.assertEqual(w_arr_fast, ctx_ref.w_arr)
            self.assertEqual(w_mask_fast, ctx_ref.w_mask)

            self.assertEqual(
                eval_ai.choose_suitcase_card(sim, 0, sim.suitcase),
                eval_ai.choose_suitcase_card(g, 0, g.suitcase),
            )

            metas = tuple(_HYDROGEN_META_BY_SLUG[c.definition.slug] for c in sim.suitcase)
            self.assertTrue(all(m.q_const_eligible for m in metas))
            costs = tuple(sim.cards_energy_costs(0, sim.suitcase))
            sc_plain = tuple(("Q", m) for m in metas)
            self.assertEqual(
                _suitcase_pick1_edge(
                    sim, 0, _sc_costs=costs, _sc_metas=metas, _sc_plain=sc_plain
                ),
                _suitcase_pick1_edge(sim, 0),
            )

            # Mixed suitcase (plain 5-tuple / ("Q", meta) / None fallback) must also match.
            g.suitcase = [make_card(s) for s in ("waterski", "floating", "doxology", "evening-chat")]
            sim_m = ai._prepare_observable_game(g, 0)
            metas_m = tuple(_HYDROGEN_META_BY_SLUG[c.definition.slug] for c in sim_m.suitcase)
            costs_m = tuple(sim_m.cards_energy_costs(0, sim_m.suitcase))
            sc_plain_m = (
                None,
                ("Q", metas_m[1]),
                None,
                ("Q", metas_m[3]),
            )
            self.assertEqual(
                _suitcase_pick1_edge(
                    sim_m, 0, _sc_costs=costs_m, _sc_metas=metas_m, _sc_plain=sc_plain_m
                ),
                _suitcase_pick1_edge(sim_m, 0),
            )

    def test_wave9_inplace_on_play_and_exact_optimizations(self):
        class _MaterializedRefOsmiumAI(OsmiumAI):
            def _simulate_play_step(
                self, state, player_index, hand_idx, known_cost=None
            ):
                return super()._simulate_play_step(
                    state, player_index, hand_idx, known_cost
                )

        # 1. Bit-exact search parity across all 16 new in-place on_play ops + 7-wonders/solo.
        boards = [
            # evening-chat, early-bedtime, morning-coffee, paddleboard (hand > 0)
            (4, 7, ("evening-chat", "early-bedtime", "morning-coffee", "paddleboard", "azul"), (), (), ()),
            # paddleboard with empty remaining hand (hand == 0 after play)
            (6, 7, ("paddleboard",), (), (), ()),
            # sleep-in + nap + waterski (sleep-in blocks subsequent energy gain)
            (5, 3, ("sleep-in", "nap", "waterski", "spikeball"), (), (), ()),
            # butter-tarts with remaining energy > 0 and == 0 (cost=4 -> e=7 vs e=4)
            (5, 7, ("butter-tarts", "waterski", "calzones"), (), (), ()),
            (5, 4, ("butter-tarts", "calzones"), (), (), ()),
            # wingspan alone and with fun modifier (bug-spray / epic-playlist) on board
            (4, 7, ("wingspan", "bug-spray", "epic-playlist", "azul", "splendor"), (), (), ()),
            # the-crew untriggered (< 2 opp board games) and triggered (>= 2 opp board games)
            (5, 6, ("the-crew", "azul", "splendor"), (), (), ((), (), ())),
            (5, 6, ("the-crew", "azul", "splendor"), (), (), (("azul",), ("splendor",), ())),
            # wurstchen, camp-crossroads-dessert, keeper, potato-pancakes
            (4, 7, ("wurstchen", "camp-crossroads-dessert", "keeper", "potato-pancakes", "waterski", "nap", "movie"), (), (), ()),
            # ice-cream-sandwich & doritos (with and without prior Food)
            (5, 7, ("ice-cream-sandwich", "doritos", "calzones", "waterski"), (), (), ()),
            # steak (with and without cost >= 5), beach-conversation (immediately after Outdoors)
            (4, 7, ("waterski", "steak", "beach-conversation", "splendor", "azul"), (), (), ()),
            # 7-wonders, solo ("C" dedup) and epic-prank (falls back cleanly)
            (4, 7, ("7-wonders", "solo", "epic-prank", "azul", "movie"), (), (), (("azul",), (), ())),
        ]
        for b_idx, (day, energy, hand_s, played_s, tom_s, opp_played_s) in enumerate(boards):
            g_a = _osmium_game(seed=900 + b_idx)
            g_a.day = day
            g_a._is_setup = True
            g_a.suitcase = [make_card(s) for s in ("waterski", "movie", "splendor", "azul")]
            g_a.trunk = [make_card("splendor") for _ in range(12)]
            g_a.players[0].energy = energy
            g_a.players[0].hand = [make_card(s) for s in hand_s]
            g_a.players[0].played_today = [make_card(s) for s in played_s]
            g_a.players[0].tomorrow_cards = [make_card(s) for s in tom_s]
            for opp_i, ops in enumerate(opp_played_s, start=1):
                g_a.players[opp_i].played_today = [make_card(s) for s in ops]

            ai_fast = OsmiumAI(random.Random(920 + b_idx))
            ai_ref = _MaterializedRefOsmiumAI(random.Random(920 + b_idx))
            sim_a = ai_fast._prepare_observable_game(g_a, 0)
            sim_a._capture_day_start_fun()
            sim_b = ai_ref._prepare_observable_game(g_a, 0)
            sim_b._capture_day_start_fun()

            snap_hand = [c.definition.slug for c in sim_a.players[0].hand]
            snap_e = sim_a.players[0].energy
            snap_fun = sim_a.players[0].fun
            snap_pf = sim_a._p_flags[0]

            act_a, val_a, tbl_a = ai_fast._search_best_day_plan_with_table(sim_a, 0)
            act_b, val_b, tbl_b = ai_ref._search_best_day_plan_with_table(sim_b, 0)

            self.assertEqual([c.definition.slug for c in sim_a.players[0].hand], snap_hand)
            self.assertEqual([c.definition.slug for c in sim_a.players[0].played_today], list(played_s))
            self.assertEqual(sim_a.players[0].energy, snap_e)
            self.assertEqual(sim_a.players[0].fun, snap_fun)
            self.assertEqual(sim_a.players[0].asleep, False)
            self.assertEqual(sim_a._p_flags[0], snap_pf)
            self.assertTrue(all(not c.markers for c in sim_a.players[0].hand))

            self.assertEqual(act_a, act_b, msg=f"wave9 board {b_idx}")
            self.assertAlmostEqual(val_a, val_b, places=12, msg=f"wave9 board {b_idx}")
            self.assertEqual(set(tbl_a), set(tbl_b), msg=f"wave9 board {b_idx}")
            for k in tbl_a:
                self.assertAlmostEqual(tbl_a[k][0], tbl_b[k][0], places=12)
                st_a, st_b = tbl_a[k][1], tbl_b[k][1]
                self.assertIsInstance(st_a, Game)
                self.assertEqual(st_a.players[0].asleep, st_b.players[0].asleep)
                self.assertEqual(st_a.players[0].fun, st_b.players[0].fun)
                self.assertEqual(st_a.players[0].energy, st_b.players[0].energy)
                self.assertEqual(
                    [c.markers for c in st_a.players[0].played_today],
                    [c.markers for c in st_b.players[0].played_today],
                )

        # Verify early-bedtime / evening-chat materialize asleep=True in table,
        # and paddleboard sets _PF_HAS_SIG_MARKER when rem_hand > 0 only.
        g_sleep = _osmium_game(seed=950)
        g_sleep.day = 3
        g_sleep._is_setup = True
        g_sleep.suitcase = [make_card("azul") for _ in range(4)]
        g_sleep.trunk = [make_card("splendor") for _ in range(8)]
        g_sleep.players[0].energy = 7
        g_sleep.players[0].hand = [make_card("early-bedtime"), make_card("evening-chat")]
        ai_s = OsmiumAI(random.Random(951))
        sim_s = ai_s._prepare_observable_game(g_sleep, 0)
        sim_s._capture_day_start_fun()
        _, _, tbl_s = ai_s._search_best_day_plan_with_table(sim_s, 0)
        self.assertTrue(tbl_s[0][1].players[0].asleep)
        self.assertTrue(tbl_s[1][1].players[0].asleep)

        g_pad = _osmium_game(seed=952)
        g_pad.day = 6
        g_pad._is_setup = True
        g_pad.suitcase = [make_card("azul") for _ in range(4)]
        g_pad.trunk = [make_card("splendor") for _ in range(8)]
        g_pad.players[0].energy = 7
        g_pad.players[0].hand = [make_card("paddleboard"), make_card("azul")]
        sim_pad1 = ai_s._prepare_observable_game(g_pad, 0)
        sim_pad1._capture_day_start_fun()
        _, _, tbl_pad1 = ai_s._search_best_day_plan_with_table(sim_pad1, 0)
        self.assertTrue(bool(tbl_pad1[0][1]._p_flags[0] & _PF_HAS_SIG_MARKER))
        g_pad.players[0].hand = [make_card("paddleboard")]
        sim_pad0 = ai_s._prepare_observable_game(g_pad, 0)
        sim_pad0._capture_day_start_fun()
        _, _, tbl_pad0 = ai_s._search_best_day_plan_with_table(sim_pad0, 0)
        self.assertFalse(bool(tbl_pad0[0][1]._p_flags[0] & _PF_HAS_SIG_MARKER))

        # 2. _PF_HAS_FUN_VAL + score_played_today_fast parity across plain, fun_val, and tom/copied boards.
        ai_f = OsmiumAI(random.Random(960))
        for slug_list, is_tom_idx, expected_fv in (
            (["azul", "movie", "splendor"], -1, False),
            (["azul", "bubly", "decaf"], -1, True),
            (["azul", "7-wonders"], -1, True),
            (["azul", "movie"], 0, False),
            (["azul", "evening-chat"], -1, False),
        ):
            g_sc = _osmium_game(seed=961)
            g_sc.day = 4
            g_sc._is_setup = True
            g_sc.players[0].energy = 3
            g_sc.players[0].played_today = [make_card(s) for s in slug_list]
            if is_tom_idx >= 0:
                g_sc.players[0].played_today[is_tom_idx].is_tomorrow = True
            sim_sc = ai_f._prepare_observable_game(g_sc, 0)
            self.assertEqual(bool(sim_sc._p_flags[0] & _PF_HAS_FUN_VAL), expected_fv)
            fast_score, fast_tom = sim_sc.score_played_today_fast(0)
            ref_score = sum(g_sc.card_fun(0, c) for c in g_sc.players[0].played_today)
            self.assertEqual(fast_score, ref_score)
            self.assertEqual(
                [c.definition.slug for c in fast_tom],
                [
                    c.definition.slug
                    for c in g_sc.players[0].played_today
                    if c.effective_behavior.has_tomorrow_action
                ],
            )

        # 3. _NEUTRAL_Q_OK_D0 / _NEUTRAL_Q_OK_D1 parity across all 161 registry cards.
        _ensure_osmium_caches()
        g_nq = _osmium_game(seed=970)
        g_nq.players[0].hand = [make_card("movie"), make_card("canoe")]
        g_nq.players[0].played_today = [make_card("nap")]
        g_nq.players[0].tomorrow_cards = [make_card("photo-shoot")]
        sim_nq = ai_f._prepare_observable_game(g_nq, 0)
        sim_ctx_nq = _build_quick_zone_ctx(sim_nq.players[0], True)
        for day, tbl in ((1, _NEUTRAL_Q_OK_D1), (6, _NEUTRAL_Q_OK_D0)):
            sim_nq.day = day
            for slug, defn in CARD_REGISTRY.items():
                meta = _HYDROGEN_META_BY_SLUG[slug]
                if not meta.q_const_eligible or meta.has_own_cost_mod:
                    continue
                q_ref = _hydrogen_quick_card_desirability(
                    sim_nq,
                    0,
                    CardInstance(1000, defn),
                    {},
                    _ctx=sim_ctx_nq,
                    _in_hand=False,
                    _meta=meta,
                )
                self.assertEqual(tbl[slug], 0.2 <= q_ref <= 1.6, msg=f"day={day} {slug}")

        # 4. _continuation_potential single-bit aa_mask + played fast-skip parity and min_bs_cost=3/4 boards.
        for cont_played, cont_hand, cont_e in (
            (("waterski", "azul", "movie", "calzones"), ("outdoor-movie", "azul", "splendor"), 5),
            (("waterski", "after-dinner-entertainment", "calzones", "stretch"), ("ice-wine", "nap", "movie"), 2),
            (("hold-the-baby", "bug-spray"), ("nos-book", "calzones", "waterski"), 3),
        ):
            g_c = _osmium_game(seed=980)
            g_c.day = 3
            g_c._is_setup = True
            g_c.suitcase = [make_card("azul") for _ in range(4)]
            g_c.trunk = [make_card("splendor") for _ in range(8)]
            g_c.players[0].energy = cont_e
            g_c.players[0].played_today = [make_card(s) for s in cont_played]
            g_c.players[0].hand = [make_card(s) for s in cont_hand]
            sim_c = ai_f._prepare_observable_game(g_c, 0)
            self.assertAlmostEqual(
                ai_f._continuation_potential(sim_c, 0),
                ai_f._continuation_potential(g_c, 0),
                places=12,
            )
            sim_c._capture_day_start_fun()
            g_c_ref = _hydrogen_copy_game(g_c)
            sim_cr = _MaterializedRefOsmiumAI(random.Random(981))._prepare_observable_game(
                g_c_ref, 0
            )
            sim_cr._capture_day_start_fun()
            _, val_cf, _ = ai_f._search_best_day_plan_with_table(sim_c, 0)
            _, val_cr, _ = _MaterializedRefOsmiumAI(
                random.Random(981)
            )._search_best_day_plan_with_table(sim_cr, 0)
            self.assertAlmostEqual(val_cf, val_cr, places=12)

    def test_wave10_photo_shoot_ade_nt1_and_unseen_parity(self):
        class _MaterializedRefOsmiumAI(OsmiumAI):
            def _simulate_play_step(
                self, state, player_index, hand_idx, known_cost=None
            ):
                return super()._simulate_play_step(
                    state, player_index, hand_idx, known_cost
                )

        # 1. Photo Shoot tomorrow: non-Social/non-Relax play in-place, Social/Relax trigger Suitcase pickup.
        g_ps = _osmium_game(seed=1001)
        g_ps.day = 4
        g_ps._is_setup = True
        g_ps.suitcase = [make_card(s) for s in ("waterski", "movie", "splendor", "azul")]
        g_ps.trunk = [make_card("calzones") for _ in range(10)]
        g_ps.players[0].energy = 7
        g_ps.players[0].tomorrow_cards = [make_card("photo-shoot")]
        g_ps.players[0].hand = [
            make_card(s)
            for s in ("waterski", "azul", "calzones", "wingspan", "movie", "nap")
        ]
        ai_fast = OsmiumAI(random.Random(1002))
        ai_ref = _MaterializedRefOsmiumAI(random.Random(1002))
        sim_ps_a = ai_fast._prepare_observable_game(g_ps, 0)
        sim_ps_a._capture_day_start_fun()
        self.assertTrue(bool(sim_ps_a._p_flags[0] & _PF_HAS_PHOTO_SHOOT))
        sim_ps_b = ai_ref._prepare_observable_game(g_ps, 0)
        sim_ps_b._capture_day_start_fun()
        act_a, val_a, tbl_a = ai_fast._search_best_day_plan_with_table(sim_ps_a, 0)
        act_b, val_b, tbl_b = ai_ref._search_best_day_plan_with_table(sim_ps_b, 0)
        self.assertEqual(act_a, act_b)
        self.assertAlmostEqual(val_a, val_b, places=12)
        self.assertEqual(set(tbl_a), set(tbl_b))
        for k in tbl_a:
            self.assertAlmostEqual(tbl_a[k][0], tbl_b[k][0], places=12)
            self.assertEqual(
                len(tbl_a[k][1].players[0].hand),
                len(tbl_b[k][1].players[0].hand),
            )
        # Non-Social/Relax followed directly by Social/Relax triggers pick identically in fast sim.
        sim_step = ai_fast._prepare_observable_game(g_ps, 0)
        sim_step.play_card(0, 1)  # azul (Board Game) -> no pick
        self.assertEqual(len(sim_step.players[0].hand), 5)
        sim_step.play_card(0, 3)  # movie (Social) -> triggers Photo Shoot pick (+1 card -1 played = 5)
        self.assertEqual(len(sim_step.players[0].hand), 5)

        # 2. _MF_ADE in Fast Path 3, Fast Path 3b (copied/all_tags), Slow Path (custom card), and blocked mask.
        self.assertEqual(_HYDROGEN_META_BY_SLUG["after-dinner-entertainment"].mod_fun_kind, _MF_ADE)
        ade_boards = [
            # Fast Path 3: with earlier Food (doubles only first Exercise), and without Food
            (("calzones", "after-dinner-entertainment", "waterski", "spikeball", "water-trampoline"), False),
            (("after-dinner-entertainment", "calzones", "waterski", "water-trampoline"), False),
            # Fast Path 3b: with copied card / _PF_ALL_TAGS
            (("calzones", "after-dinner-entertainment", "waterski", "bring-a-friend"), False),
            # Slow Path: custom non-canonical card alongside ADE
            (("calzones", "after-dinner-entertainment", "waterski"), True),
        ]
        for slugs, add_custom in ade_boards:
            g_ade = _osmium_game(seed=1010)
            g_ade.day = 4
            g_ade._is_setup = True
            g_ade.players[0].played_today = [make_card(s) for s in slugs]
            if add_custom:
                g_ade.players[0].played_today.append(
                    _dummy_card(99001, "Custom Ex", 1, 2, frozenset({"Exercise"}))
                )
            sim_ade = ai_fast._prepare_observable_game(g_ade, 0)
            fast_s, _ = sim_ade.score_played_today_fast(0)
            ref_s = sum(g_ade.card_fun(0, c) for c in g_ade.players[0].played_today)
            self.assertEqual(fast_s, ref_s)

        g_blk = _osmium_game(seed=1011)
        g_blk.day = 4
        g_blk._is_setup = True
        g_blk.players[0].played_today = [make_card("calzones"), make_card("after-dinner-entertainment")]
        sim_blk = ai_fast._prepare_observable_game(g_blk, 0)
        self.assertEqual(
            _compute_blocked_tag_mask(sim_blk.players[0], sim_blk._p_flags[0], _HYDROGEN_META_BY_SLUG),
            _MASK_EXERCISE,
        )
        g_blk.players[0].played_today.append(make_card("waterski"))
        sim_blk2 = ai_fast._prepare_observable_game(g_blk, 0)
        self.assertEqual(
            _compute_blocked_tag_mask(sim_blk2.players[0], sim_blk2._p_flags[0], _HYDROGEN_META_BY_SLUG),
            0,
        )

        # 3. _evaluate_terminal_day inline nt == 1 reserve cache parity.
        for tom_s, hand_s in (
            (("evening-chat",), ()),
            (("photo-shoot",), ("azul", "movie")),
            (("wildlife-spotting",), ("azul", "movie", "waterski", "calzones", "nap")),
            (("colouring",), ("azul", "movie")),
        ):
            g_nt = _osmium_game(seed=1020)
            g_nt.day = 3
            g_nt._is_setup = True
            g_nt.suitcase = [make_card("azul") for _ in range(4)]
            g_nt.trunk = [make_card("splendor") for _ in range(8)]
            g_nt.players[0].energy = 2
            g_nt.players[0].hand = [make_card(s) for s in hand_s]
            g_nt.players[0].played_today = [make_card(s) for s in tom_s]
            if tom_s[0] == "colouring":
                g_nt.players[0].played_today[0].markers["chosen_tag"] = "Outdoors"
            sim_nt = ai_fast._prepare_observable_game(g_nt, 0)
            # First call populates cache; second hits inline nt == 1 cache in _evaluate_terminal_day.
            val_first = ai_fast._evaluate_terminal_day(sim_nt, 0, turns_from_root=1, root_skipped_turns=0)
            val_second = ai_fast._evaluate_terminal_day(sim_nt, 0, turns_from_root=1, root_skipped_turns=0)
            ai_uncached = OsmiumAI(random.Random(1020))
            ai_uncached._reserve_eval_cache = {}
            val_ref = ai_uncached._evaluate_terminal_day(sim_nt, 0, turns_from_root=1, root_skipped_turns=0)
            self.assertAlmostEqual(val_first, val_ref, places=12)
            self.assertAlmostEqual(val_second, val_ref, places=12)

        # 4. _HydrogenEvalAI.choose_suitcase_card no-cost-mod precomputed costs parity.
        eval_ai = _HydrogenEvalAI()
        for sc_slugs, hand_slugs, played_slugs in (
            (("classic-book", "sing-song", "azul", "movie"), ("nap", "floating"), ()),
            (("floating", "evening-chat", "sunrise", "paint"), ("azul",), ()),
            (("classic-book", "waterski", "movie", "splendor"), ("nap",), ("beaver-burgers",)),
        ):
            g_ev = _osmium_game(seed=1030)
            g_ev.day = 4
            g_ev._is_setup = True
            g_ev.suitcase = [make_card(s) for s in sc_slugs]
            g_ev.players[0].energy = 4
            g_ev.players[0].hand = [make_card(s) for s in hand_slugs]
            g_ev.players[0].played_today = [make_card(s) for s in played_slugs]
            sim_ev = ai_fast._prepare_observable_game(g_ev, 0)
            self.assertEqual(
                eval_ai.choose_suitcase_card(sim_ev, 0, sim_ev.suitcase),
                eval_ai.choose_suitcase_card(g_ev, 0, g_ev.suitcase),
            )

        # 5. _unseen_tag_counts_and_total zero-alloc path parity.
        g_un = _osmium_game(seed=1040)
        g_un.suitcase = [make_card("waterski"), make_card("azul")]
        g_un.discard = [make_card("calzones")]
        g_un.trunk = [make_card("splendor") for _ in range(20)]
        g_un.players[0].hand = [make_card("movie"), make_card("nap")]
        g_un.players[1].played_today = [make_card("beaver-burgers")]
        g_un.players[2].tomorrow_cards = [make_card("photo-shoot")]

        def ref_unseen(game_obj, pi, ktop):
            vis = list(game_obj.suitcase) + list(game_obj.discard) + list(game_obj.players[pi].hand)
            for p in game_obj.players:
                vis.extend(p.played_today)
                vis.extend(p.tomorrow_cards)
            seen_ids = {c.instance_id for c in vis}
            for kc in ktop:
                if kc.instance_id not in seen_ids:
                    vis.append(kc)
                    seen_ids.add(kc.instance_id)
            tot = sum(1 for _ in CARD_REGISTRY) - len(vis)
            cnts = dict(_CATALOG_TAG_COUNTS)
            for c in vis:
                for t in c.definition.tags:
                    if t in cnts:
                        cnts[t] = max(0, cnts[t] - 1)
            return cnts, tot

        c_fast, t_fast = _unseen_tag_counts_and_total(g_un, 0, ())
        c_ref, t_ref = ref_unseen(g_un, 0, ())
        self.assertEqual((c_fast, t_fast), (c_ref, t_ref))
        kt = (make_card("canoe"), g_un.discard[0])
        self.assertEqual(
            _unseen_tag_counts_and_total(g_un, 0, kt),
            ref_unseen(g_un, 0, kt),
        )
        g_un.trunk = []
        self.assertEqual(_unseen_tag_counts_and_total(g_un, 0, ()), _unseen_tag_counts_and_total(g_un, 0, kt))

    def test_adversarial_review_regressions(self):
        import thenos.ais.carbon as carbon_mod
        import thenos.ais.hydrogen as hydrogen_mod

        class _NoCanonRefOsmiumAI(OsmiumAI):
            def _state_signature(self, sim, player_index):
                return super()._state_signature(sim, player_index)

            def _simulate_play_step(
                self, state, player_index, hand_idx, known_cost=None
            ):
                return super()._simulate_play_step(
                    state, player_index, hand_idx, known_cost
                )

        # 1. Copied Beach Volleyball (Wedding Anniversary) must not take the in-place fast path.
        g_bv = _osmium_game(seed=1101)
        g_bv.day = 6
        g_bv._is_setup = True
        g_bv.suitcase = [make_card("splendor") for _ in range(4)]
        g_bv.trunk = [make_card("splendor") for _ in range(8)]
        g_bv.players[1].played_today = [make_card("beach-volleyball")]
        g_bv.players[0].energy = 7
        g_bv.players[0].hand = [
            make_card("wedding-anniversary"),
            make_card("movie"),
            make_card("kneeboard"),
        ]
        ai_bv = OsmiumAI(random.Random(1102))
        ai_bv_ref = _NoCanonRefOsmiumAI(random.Random(1102))
        sim_bv = ai_bv._prepare_observable_game(g_bv, 0)
        sim_bv._capture_day_start_fun()
        act_bv, val_bv, tbl_bv = ai_bv._search_best_day_plan_with_table(sim_bv, 0)
        act_bv_ref, val_bv_ref, tbl_bv_ref = ai_bv_ref._search_best_day_plan_with_table(
            ai_bv_ref._prepare_observable_game(g_bv, 0), 0
        )
        self.assertEqual(act_bv, 0)
        self.assertEqual(act_bv, act_bv_ref)
        self.assertAlmostEqual(val_bv, val_bv_ref, places=12)
        self.assertAlmostEqual(tbl_bv[0][0], tbl_bv_ref[0][0], places=12)

        # 2. _search_best_day_plan allowed_first_indices fallback must respect cost <= energy & can_play.
        g_unaff = _osmium_game(seed=1103)
        g_unaff.day = 6
        g_unaff._is_setup = True
        g_unaff.suitcase = [make_card("splendor") for _ in range(4)]
        g_unaff.players[0].energy = 1
        g_unaff.players[0].hand = [make_card("adventure-race")]
        ai_chk = OsmiumAI(random.Random(1104))
        act_unaff, val_unaff = ai_chk._search_best_day_plan(
            ai_chk._prepare_observable_game(g_unaff, 0),
            0,
            max_plays=5,
            allowed_first_indices=[0],
        )
        self.assertIsNone(act_unaff)
        self.assertAlmostEqual(val_unaff, 0.0, places=12)
        g_cant = _osmium_game(seed=1105)
        g_cant.day = 6
        g_cant._is_setup = True
        g_cant.suitcase = [make_card("splendor") for _ in range(4)]
        g_cant.players[0].energy = 5
        g_cant.players[0].played_today = [make_card("nap")]
        g_cant.players[0].hand = [make_card("morning-coffee")]
        act_cant, val_cant = ai_chk._search_best_day_plan(
            ai_chk._prepare_observable_game(g_cant, 0),
            0,
            max_plays=5,
            allowed_first_indices=[0],
        )
        self.assertIsNone(act_cant)
        self.assertAlmostEqual(val_cant, 0.0, places=12)

        # 3. Canonical "C" signature must not collide order-dependent Solo / Bubly lines.
        g_solo = _osmium_game(seed=1106)
        g_solo.day = 6
        g_solo._is_setup = True
        g_solo.suitcase = [make_card("azul") for _ in range(4)]
        g_solo.players[1].played_today = [make_card("splendor") for _ in range(4)]  # 12 Fun
        g_solo.players[0].energy = 15
        g_solo.players[0].hand = [
            make_card("nos-shirt"),
            make_card("dumbed-down-settlers"),
            make_card("solo"),
            make_card("fishing-evening"),
        ]
        ai_s = OsmiumAI(random.Random(1107))
        ai_s_ref = _NoCanonRefOsmiumAI(random.Random(1107))
        act_s, val_s, tbl_s = ai_s._search_best_day_plan_with_table(
            ai_s._prepare_observable_game(g_solo, 0), 0
        )
        act_sr, val_sr, tbl_sr = ai_s_ref._search_best_day_plan_with_table(
            ai_s_ref._prepare_observable_game(g_solo, 0), 0
        )
        self.assertEqual(act_s, act_sr)
        self.assertAlmostEqual(val_s, val_sr, places=12)
        self.assertEqual(set(tbl_s), set(tbl_sr))
        for k in tbl_s:
            self.assertAlmostEqual(tbl_s[k][0], tbl_sr[k][0], places=12)

        g_bub = _osmium_game(seed=1108)
        g_bub.day = 6
        g_bub._is_setup = True
        g_bub.suitcase = [make_card("azul") for _ in range(4)]
        g_bub.players[0].energy = 12
        g_bub.players[0].hand = [
            make_card("nos-shirt"),
            make_card("nap"),
            make_card("solo"),
            make_card("bubly"),
        ]
        act_b, val_b, tbl_b = ai_s._search_best_day_plan_with_table(
            ai_s._prepare_observable_game(g_bub, 0), 0
        )
        act_br, val_br, tbl_br = ai_s_ref._search_best_day_plan_with_table(
            ai_s_ref._prepare_observable_game(g_bub, 0), 0
        )
        self.assertEqual(act_b, act_br)
        self.assertAlmostEqual(val_b, val_br, places=12)
        for k in tbl_b:
            self.assertAlmostEqual(tbl_b[k][0], tbl_br[k][0], places=12)

        # 4. Inline end-day Solo must not reuse stale opp_today when opp has Fit to Print.
        g_ftp = _osmium_game(seed=1109)
        g_ftp.day = 6
        g_ftp._is_setup = True
        g_ftp.suitcase = [make_card("azul") for _ in range(4)]
        g_ftp.players[1].played_today = [make_card("fit-to-print"), make_card("splendor")]
        g_ftp.players[0].energy = 10
        g_ftp.players[0].hand = [
            make_card("solo"),
            make_card("movie"),
            make_card("movie"),
            make_card("movie"),
        ]
        ai_ftp = OsmiumAI(random.Random(1110))
        ai_ftp_ref = _NoCanonRefOsmiumAI(random.Random(1110))
        sim_ftp = ai_ftp._prepare_observable_game(g_ftp, 0)
        sim_ftp._capture_day_start_fun()
        act_f, val_f, tbl_f = ai_ftp._search_best_day_plan_with_table(sim_ftp, 0)
        sim_ftp_r = ai_ftp_ref._prepare_observable_game(g_ftp, 0)
        sim_ftp_r._capture_day_start_fun()
        act_fr, val_fr, tbl_fr = ai_ftp_ref._search_best_day_plan_with_table(sim_ftp_r, 0)
        self.assertEqual(act_f, act_fr)
        self.assertAlmostEqual(val_f, val_fr, places=12)
        self.assertEqual(set(tbl_f), set(tbl_fr))
        for k in tbl_f:
            self.assertAlmostEqual(tbl_f[k][0], tbl_fr[k][0], places=12)

        # 5. Copied Photo Shoot in tomorrow_cards must block Social/Relax in _compute_blocked_tag_mask.
        g_cps = _osmium_game(seed=1111)
        g_cps.day = 6
        g_cps._is_setup = True
        g_cps.suitcase = [
            make_card("kneeboard"),
            make_card("movie"),
            make_card("dumbed-down-settlers"),
            make_card("fishing-evening"),
        ]
        g_cps.trunk = [make_card("m-ms") for _ in range(10)]
        copied_ps = CardInstance(
            9901,
            CARD_REGISTRY["wedding-anniversary"],
            is_tomorrow=True,
            markers={"_copied_definition": CARD_REGISTRY["photo-shoot"]},
        )
        g_cps.players[0].tomorrow_cards = [copied_ps]
        g_cps.players[0].played_today = [make_card("kneeboard")]
        g_cps.players[0].energy = 3
        g_cps.players[0].hand = [make_card("campfire"), make_card("nap")]
        ai_cps = OsmiumAI(random.Random(1112))
        ai_cps_ref = _NoCanonRefOsmiumAI(random.Random(1112))
        sim_cps = ai_cps._prepare_observable_game(g_cps, 0)
        sim_cps._capture_day_start_fun()
        self.assertEqual(
            _compute_blocked_tag_mask(
                sim_cps.players[0], sim_cps._p_flags[0], _HYDROGEN_META_BY_SLUG
            ),
            0xFF,
        )
        act_cp, val_cp, tbl_cp = ai_cps._search_best_day_plan_with_table(sim_cps, 0)
        sim_cps_r = ai_cps_ref._prepare_observable_game(g_cps, 0)
        sim_cps_r._capture_day_start_fun()
        act_cpr, val_cpr, tbl_cpr = ai_cps_ref._search_best_day_plan_with_table(sim_cps_r, 0)
        self.assertEqual(act_cp, 0)
        self.assertEqual(act_cp, act_cpr)
        self.assertAlmostEqual(val_cp, val_cpr, places=12)
        for k in tbl_cp:
            self.assertAlmostEqual(tbl_cp[k][0], tbl_cpr[k][0], places=12)

        # 6. Subclass overrides without private kwargs must not raise TypeError.
        class _OverrideSubOsmiumAI(OsmiumAI):
            def _continuation_potential(self, sim, player_index):
                return super()._continuation_potential(sim, player_index)

            def _suitcase_pick1_bonus(self, sim, player_index):
                return super()._suitcase_pick1_bonus(sim, player_index)

            def _extra_continuation_bonus(
                self,
                sim,
                player_index,
                energy_budget,
                has_unconsumed_ade,
                has_dates_fn,
                has_ck,
                epic_prank_target,
                has_unconsumed_nos_shirt=False,
            ):
                return super()._extra_continuation_bonus(
                    sim,
                    player_index,
                    energy_budget,
                    has_unconsumed_ade,
                    has_dates_fn,
                    has_ck,
                    epic_prank_target,
                    has_unconsumed_nos_shirt,
                )

        g_sub = _osmium_game(seed=1113)
        g_sub.day = 3
        g_sub._is_setup = True
        g_sub.suitcase = [make_card("azul") for _ in range(4)]
        g_sub.trunk = [make_card("splendor") for _ in range(8)]
        g_sub.players[0].energy = 6
        g_sub.players[0].played_today = [
            make_card("m-ms"),
            make_card("after-dinner-entertainment"),
        ]
        g_sub.players[0].hand = [make_card("waterski"), make_card("movie")]
        sub_ai = _OverrideSubOsmiumAI(random.Random(1114))
        sub_choice = sub_ai.choose_card_to_play(
            g_sub, 0, list(range(len(g_sub.players[0].hand)))
        )
        self.assertIn(sub_choice, (None, 0, 1))

        # 7. Shared-opponent CoW marker safety and carbon bitmask parity.
        self.assertEqual(
            carbon_mod._CARBON_TAG_SYNC_FLAGS,
            hydrogen_mod._PF_ALL_TAGS | hydrogen_mod._PF_MAY_HAVE_TAG_MARKERS,
        )
        g_cow = _osmium_game(seed=1115)
        g_cow.day = 4
        g_cow._is_setup = True
        g_cow.suitcase = [make_card("azul") for _ in range(4)]
        c_col = make_card("colouring")
        c_col.markers["chosen_tag"] = "Food"
        c_mov = make_card("movie")
        g_cow.players[1].played_today = [c_col, c_mov]
        c_stale = make_card("movie")
        c_stale.markers["_all_tags"] = True
        c_stale.markers["_extra_tags"] = frozenset({"Food"})
        g_cow.players[2].played_today = [c_stale]
        sim_root = ai_s._prepare_observable_game(g_cow, 0)
        sim_cow = hydrogen_mod._hydrogen_copy_game_for_play(sim_root, 0)
        self.assertNotEqual(sim_cow._shared_opp_mask & (1 << 1), 0)
        self.assertNotEqual(sim_cow._shared_opp_mask & (1 << 2), 0)
        _ = sim_cow.card_fun(1, sim_cow.players[1].played_today[1])
        _ = sim_cow.score_played_today_fast(1)
        _ = sim_cow.card_fun(2, sim_cow.players[2].played_today[0])
        _ = sim_cow.score_played_today_fast(2)
        self.assertNotIn("_extra_tags", sim_root.players[1].played_today[1].markers)
        self.assertNotIn("_all_tags", sim_root.players[1].played_today[1].markers)
        self.assertTrue(sim_root.players[2].played_today[0].markers.get("_all_tags"))
        self.assertEqual(
            sim_root.players[2].played_today[0].markers.get("_extra_tags"),
            frozenset({"Food"}),
        )

    def test_trunk_ordering_and_live_known_trunk_reshuffle_invalidation(self):
        game = _osmium_game(seed=1201)
        game.day = 3
        game._is_setup = True
        game.suitcase = [make_card("nap") for _ in range(4)]
        game.trunk = [make_card("splendor") for _ in range(12)]
        ai = game.ais[0]

        # 1. With no consumer in hand, bury stronger cards (Zumba last, Splendor on top).
        game.players[0].energy = 7
        game.players[0].hand = [make_card("movie")]
        cards_no_cons = [make_card("zumba"), make_card("splendor"), make_card("waterski")]
        order_no_cons = list(ai.order_cards_for_trunk(game, 0, cards_no_cons))
        self.assertEqual(cards_no_cons[order_no_cons[0]].definition.slug, "splendor")
        self.assertEqual(cards_no_cons[order_no_cons[-1]].definition.slug, "zumba")
        sim_nc = ai._prepare_observable_game(game, 0)
        self.assertEqual(list(ai.order_cards_for_trunk(sim_nc, 0, cards_no_cons)), order_no_cons)

        # 2. Assorted Cutlery affordable via Nap places Adventure Race on top.
        cards_cutlery = [make_card("adventure-race"), make_card("nap"), make_card("splendor")]
        game.players[0].energy = 0
        game.players[0].hand = [make_card("assorted-cutlery"), make_card("nap")]
        order_ac = list(ai.order_cards_for_trunk(game, 0, cards_cutlery))
        self.assertEqual(cards_cutlery[order_ac[0]].definition.slug, "adventure-race")

        # 3. Bougie Coffee Machine orders Food synergies (High-End Red before Ice Wine/Beaver Burgers) in top 3.
        game.players[0].energy = 5
        game.players[0].hand = [make_card("bougie-coffee-machine")]
        bcm_cards = [
            make_card("ice-wine"),
            make_card("high-end-red"),
            make_card("beaver-burgers"),
            make_card("splendor"),
        ]
        order_bcm = list(ai.order_cards_for_trunk(game, 0, bcm_cards))
        top3_slugs = [bcm_cards[i].definition.slug for i in order_bcm[:3]]
        self.assertEqual(top3_slugs[0], "high-end-red")
        self.assertEqual(set(top3_slugs), {"high-end-red", "ice-wine", "beaver-burgers"})

        # 4. Treat Store Run (with Suitcase Food) & Fancy Floatie (with Suitcase Relax).
        game.suitcase = [make_card("calzones"), make_card("nap"), make_card("movie"), make_card("azul")]
        game.players[0].hand = [make_card("treat-store-run")]
        sweep_cards = [make_card("doritos"), make_card("floating"), make_card("splendor")]
        order_tsr = list(ai.order_cards_for_trunk(game, 0, sweep_cards))
        self.assertEqual(sweep_cards[order_tsr[0]].definition.slug, "doritos")
        game.suitcase = [make_card("floating"), make_card("nap"), make_card("movie"), make_card("azul")]
        game.players[0].hand = [make_card("fancy-floatie")]
        order_ff = list(ai.order_cards_for_trunk(game, 0, sweep_cards))
        self.assertEqual(sweep_cards[order_ff[0]].definition.slug, "floating")

        # 5. Quacks orders non-Outdoors ascending ahead of Outdoors.
        game.players[0].hand = [make_card("quacks")]
        q_cards = [make_card("adventure-race"), make_card("splendor"), make_card("nap")]
        order_q = list(ai.order_cards_for_trunk(game, 0, q_cards))
        self.assertEqual([q_cards[i].definition.slug for i in order_q], ["splendor", "nap", "adventure-race"])

        # 6. Poker orders lowest-value tagged card on top.
        game.players[0].hand = [make_card("poker")]
        p_cards = [make_card("waterski"), make_card("splendor"), make_card("zumba")]
        order_p = list(ai.order_cards_for_trunk(game, 0, p_cards))
        self.assertEqual(p_cards[order_p[0]].definition.slug, "splendor")

        # 7. _get_live_known_trunk_top invalidates on same-day discard reshuffle.
        r_game = _osmium_game(seed=1202)
        r_game.day = 3
        r_game._is_setup = True
        r_game.trunk = [make_card("movie") for _ in range(3)]
        r_game.discard = [make_card("azul") for _ in range(5)]
        r_ai = r_game.ais[0]
        r_ai.order_cards_for_trunk(r_game, 0, p_cards)
        r_game.trunk.extend(reversed([p_cards[i] for i in order_p]))
        self.assertEqual(len(r_ai._get_live_known_trunk_top(r_game, 0)), 3)
        # Simulate reshuffle dropping len(discard) to 0 while trunk length appears similar.
        r_game.trunk = [make_card("calzones") for _ in range(6)]
        r_game.discard = []
        self.assertEqual(r_ai._get_live_known_trunk_top(r_game, 0), ())
        self.assertIsNone(r_ai._known_trunk_state)

    def test_trunk_live_discard_setup_with_populated_trunk(self):
        # 1. Tell a Story with 1 normal awake opp sets up Adventure Race for Assorted Cutlery.
        g = _osmium_game(seed=1210)
        g.day = 3
        g._is_setup = True
        g.suitcase = [make_card("nap") for _ in range(4)]
        g.trunk = [make_card("azul") for _ in range(16)]
        g.players[0].energy = 2
        g.players[0].played_today = [make_card("tell-a-story")]
        g.players[0].hand = [
            make_card("assorted-cutlery"),
            make_card("adventure-race"),
            make_card("nap"),
        ]
        g.players[1].energy = 5
        g.players[2].asleep = True
        g.players[3].asleep = True
        idx = g.ais[0].choose_card_to_discard(g, 0, tuple(g.players[0].hand))
        self.assertEqual(g.players[0].hand[idx].definition.slug, "adventure-race")

        # 2. Avoid feeding a cheap high-value bomb into awake opponent with Photo Shoot tomorrow.
        g_ps = _osmium_game(seed=1210)
        g_ps.day = 3
        g_ps._is_setup = True
        g_ps.suitcase = [make_card("nap") for _ in range(4)]
        g_ps.trunk = [make_card("azul") for _ in range(16)]
        g_ps.players[0].energy = 4
        g_ps.players[0].played_today = [make_card("tell-a-story")]
        g_ps.players[0].hand = [
            make_card("assorted-cutlery"),
            make_card("waterski"),
            make_card("zumba"),
        ]
        g_ps.players[1].energy = 5
        g_ps.players[2].asleep = True
        g_ps.players[3].asleep = True
        idx_norm = g_ps.ais[0].choose_card_to_discard(g_ps, 0, tuple(g_ps.players[0].hand))
        self.assertEqual(g_ps.players[0].hand[idx_norm].definition.slug, "waterski")
        g_ps.players[1].tomorrow_cards = [make_card("photo-shoot")]
        g_ps.players[2].asleep = False
        g_ps.players[2].energy = 5
        g_ps.players[2].hand = [make_card("nap") for _ in range(3)]
        g_ps.players[2].tomorrow_cards = [make_card("photo-shoot")]
        idx_ps = g_ps.ais[0].choose_card_to_discard(g_ps, 0, tuple(g_ps.players[0].hand))
        self.assertNotEqual(g_ps.players[0].hand[idx_ps].definition.slug, "waterski")

        # 3. Leftovers (3E) + played BCM (after High-End Red) places Ice Wine on top via Last Year's Shirt.
        g_left = _osmium_game(seed=1211)
        g_left.day = 4
        g_left._is_setup = True
        g_left.suitcase = [make_card("nap") for _ in range(4)]
        g_left.trunk = [make_card("azul") for _ in range(16)]
        g_left.players[0].energy = 3
        g_left.players[0].played_today = [
            make_card("high-end-red"),
            make_card("bougie-coffee-machine"),
            make_card("epic-duels"),
            make_card("last-years-shirt"),
        ]
        g_left.players[0].hand = [
            make_card("leftovers"),
            make_card("ice-wine"),
            make_card("splendor"),
        ]
        idx_l = g_left.ais[0].choose_card_to_discard(
            g_left, 0, tuple(g_left.players[0].hand)
        )
        self.assertEqual(g_left.players[0].hand[idx_l].definition.slug, "ice-wine")

        # 4. Copied Tell a Story via Bring a Friend sets up Adventure Race for Cutlery.
        g_baf = _osmium_game(seed=1212)
        g_baf.day = 4
        g_baf._is_setup = True
        g_baf.suitcase = [make_card("nap") for _ in range(4)]
        g_baf.trunk = [make_card("azul") for _ in range(16)]
        copied_tas = make_card("bring-a-friend")
        copied_tas.markers["_copied_definition"] = CARD_REGISTRY["tell-a-story"]
        g_baf.players[0].energy = 2
        g_baf.players[0].played_today = [make_card("epic-duels"), copied_tas]
        g_baf.players[0].hand = [
            make_card("assorted-cutlery"),
            make_card("adventure-race"),
            make_card("nap"),
        ]
        idx_b = g_baf.ais[0].choose_card_to_discard(
            g_baf, 0, tuple(g_baf.players[0].hand)
        )
        self.assertEqual(g_baf.players[0].hand[idx_b].definition.slug, "adventure-race")

        # 5. Quacks with Last Year's Shirt places non-Outdoors rather than Outdoors.
        g_q = _osmium_game(seed=1213)
        g_q.day = 4
        g_q._is_setup = True
        g_q.suitcase = [make_card("nap") for _ in range(4)]
        g_q.trunk = [make_card("azul") for _ in range(16)]
        g_q.players[0].energy = 2
        g_q.players[0].played_today = [make_card("epic-duels"), make_card("last-years-shirt")]
        g_q.players[0].hand = [
            make_card("quacks"),
            make_card("movie"),
            make_card("waterski"),
        ]
        idx_q = g_q.ais[0].choose_card_to_discard(g_q, 0, tuple(g_q.players[0].hand))
        self.assertEqual(g_q.players[0].hand[idx_q].definition.slug, "movie")

    def test_trunk_beam_search_and_quacks_preservation(self):
        # 1. With known Adventure Race on trunk, play Assorted Cutlery Turn 1 even with awake opps.
        g1 = _osmium_game(seed=1220)
        g1.day = 3
        g1._is_setup = True
        g1.suitcase = [make_card("nap") for _ in range(4)]
        adv = make_card("adventure-race")
        g1.trunk = [make_card("azul") for _ in range(14)] + [adv]
        g1.players[0].energy = 3
        g1.players[0].hand = [make_card("assorted-cutlery"), make_card("splendor")]
        g1.ais[0]._ensure_live_game(g1)
        g1.ais[0]._known_trunk_state = (id(g1), 3, len(g1.trunk), (adv,), (0, 0, 0, 0))
        g1.ais[0]._known_trunk_discard_len = 0
        self.assertEqual(g1.ais[0].choose_card_to_play(g1, 0, [0, 1]), 0)

        # 2. Sequence Epic Duels -> Tell a Story -> Assorted Cutlery instead of burning Cutlery blind.
        g2 = _osmium_game(seed=1221)
        g2.day = 4
        g2._is_setup = True
        g2.suitcase = [make_card("nap") for _ in range(4)]
        g2.trunk = [make_card("azul") for _ in range(14)]
        g2.players[0].energy = 6
        g2.players[0].hand = [
            make_card("assorted-cutlery"),
            make_card("tell-a-story"),
            make_card("epic-duels"),
            make_card("adventure-race"),
        ]
        act2 = g2.ais[0].choose_card_to_play(g2, 0, [0, 1, 2, 3])
        self.assertIn(act2, (1, 2))

        # 3. Prefer Read the Radar before Assorted Cutlery rather than blind Cutlery.
        g3 = _osmium_game(seed=1222)
        g3.day = 4
        g3._is_setup = True
        g3.suitcase = [make_card("nap") for _ in range(4)]
        g3.trunk = [make_card("adventure-race") for _ in range(7)] + [
            make_card("waterski") for _ in range(7)
        ]
        g3.players[0].energy = 3
        g3.players[0].hand = [make_card("assorted-cutlery"), make_card("read-the-radar")]
        self.assertEqual(g3.ais[0].choose_card_to_play(g3, 0, [0, 1]), 1)

        # 4. Day 3 with 1E left holding Cutlery + Tell a Story + Adventure Race goes to bed rather than blind Cutlery.
        g4 = _osmium_game(seed=1223)
        g4.day = 3
        g4._is_setup = True
        g4.suitcase = [make_card("nap") for _ in range(4)]
        g4.trunk = [make_card("azul") for _ in range(14)]
        g4.players[0].energy = 1
        g4.players[0].hand = [
            make_card("assorted-cutlery"),
            make_card("tell-a-story"),
            make_card("adventure-race"),
        ]
        self.assertTrue(g4.ais[0].choose_to_go_to_bed(g4, 0, [0]))

        # 5. Late Day 6 with 1E and no setup card plays Assorted Cutlery blind instead of passing.
        g5 = _osmium_game(seed=1224)
        g5.day = 6
        g5._is_setup = True
        g5.suitcase = [make_card("nap") for _ in range(4)]
        g5.trunk = [make_card("azul") for _ in range(14)]
        g5.players[0].energy = 1
        g5.players[0].hand = [make_card("assorted-cutlery")]
        self.assertFalse(g5.ais[0].choose_to_go_to_bed(g5, 0, [0]))
        self.assertEqual(g5.ais[0].choose_card_to_play(g5, 0, [0]), 0)

        # 6. Quacks stops before trashing known Civilization when holding affordable Assorted Cutlery.
        g6 = _osmium_game(seed=1225)
        g6.day = 4
        g6._is_setup = True
        civ = make_card("civilization-the-boardgame")
        g6.trunk = [make_card("azul") for _ in range(10)] + [civ]
        g6.players[0].energy = 2
        g6.players[0].played_today = [make_card("quacks")]
        g6.players[0].hand = [make_card("assorted-cutlery")]
        g6.ais[0]._ensure_live_game(g6)
        g6.ais[0]._known_trunk_state = (id(g6), 4, len(g6.trunk), (civ,), (0, 0, 0, 0))
        g6.ais[0]._known_trunk_discard_len = 0
        self.assertFalse(
            g6.ais[0].choose_optional_action(g6, 0, "reveal another card for Quacks")
        )

    def test_trunk_combo_reserve_parity_and_synergies(self):
        _ensure_osmium_caches()
        trunk_slugs = [
            "read-the-radar",
            "fishing-boat",
            "ski-boat",
            "grocery-store-run",
            "tell-a-story",
            "last-years-shirt",
            "assorted-cutlery",
            "bougie-coffee-machine",
            "quacks",
            "poker",
            "treat-store-run",
            "fancy-floatie",
            "epic-duels",
            "adventure-race",
            "rouladen",
        ]
        game = _osmium_game(seed=1230)
        tom_sets = (
            [],
            [make_card("photo-shoot")],
            [make_card("epic-prank"), make_card("dates-first-nos")],
        )
        for dl in (1, 3):
            for end in (5, 10, len(trunk_slugs)):
                hand = [make_card(s) for s in trunk_slugs[:end]]
                game.players[0].hand = hand
                sorted_specs = sorted(
                    (_OSMIUM_SPEC_BY_SLUG[c.definition.slug] for c in hand),
                    key=lambda sp: sp.slug_rank,
                )
                for tom in tom_sets:
                    tom_specs = [_OSMIUM_SPEC_BY_SLUG[c.definition.slug] for c in tom]
                    slow_v = _osmium_reserve_score(game, 0, tom, dl)
                    fast_v = _osmium_reserve_score_plain_specs(
                        sorted_specs, tom_specs, dl
                    )
                    self.assertAlmostEqual(slow_v, fast_v, places=11)

        def res_for(slugs: list[str], dl: int = 2) -> float:
            game.players[0].hand = [make_card(s) for s in slugs]
            return _osmium_reserve_score(game, 0, (), dl)

        # Positive reserve combo synergies over unrelated same-slot replacement (`azul`).
        self.assertGreater(
            res_for(["assorted-cutlery", "tell-a-story", "adventure-race"])
            - res_for(["assorted-cutlery", "azul", "adventure-race"]),
            res_for(["azul", "tell-a-story", "adventure-race"])
            - res_for(["azul", "azul", "adventure-race"]),
        )
        self.assertGreater(
            res_for(["assorted-cutlery", "read-the-radar"])
            - res_for(["assorted-cutlery", "azul"]),
            res_for(["azul", "read-the-radar"]) - res_for(["azul", "azul"]),
        )
        self.assertGreater(
            res_for(["bougie-coffee-machine", "read-the-radar"])
            - res_for(["bougie-coffee-machine", "azul"]),
            res_for(["azul", "read-the-radar"]) - res_for(["azul", "azul"]),
        )
        self.assertGreater(
            res_for(["quacks", "fishing-boat"]) - res_for(["quacks", "azul"]),
            res_for(["azul", "fishing-boat"]) - res_for(["azul", "azul"]),
        )
        self.assertGreater(
            res_for(["poker", "read-the-radar"]) - res_for(["poker", "azul"]),
            res_for(["azul", "read-the-radar"]) - res_for(["azul", "azul"]),
        )

    def test_live_trunk_tracking_reshuffle_and_bohnanza_invalidation(self):
        def _setup_game(seed: int, trunk_slugs: list[str], discard_slugs: list[str]) -> Game:
            g = Game(
                [],
                [
                    OsmiumAI(random.Random(seed)),
                    RandomAI(random.Random(seed + 1)),
                    RandomAI(random.Random(seed + 2)),
                    RandomAI(random.Random(seed + 3)),
                ],
                random.Random(seed + 100),
            )
            g.day = 3
            g._is_setup = True
            g.suitcase = [make_card("nap") for _ in range(4)]
            g.trunk = [make_card(s) for s in trunk_slugs]
            g.discard = [make_card(s) for s in discard_slugs]
            return g

        # 1. Reshuffle where discard refills back to the same length (`1 == prev_dlen`).
        g1 = _setup_game(
            1301,
            ["rouladen", "solo", "movie", "nap", "fajitas", "calzones"],
            ["zumba"],
        )
        g1.players[0].hand = [make_card("read-the-radar")]
        g1.players[0].energy = 2
        g1.play_card(0, 0)
        self.assertEqual(len(g1.ais[0]._get_live_known_trunk_top(g1, 0)), 5)

        g1.players[1].hand = [make_card("euchre-tournament")]
        g1.players[1].energy = 20
        g1.play_card(1, 0)
        g1.players[2].hand = [make_card("booby-prize") for _ in range(3)]
        g1.players[2].energy = 20
        for _ in range(3):
            g1.play_card(2, 0)
        g1.players[3].hand = [make_card("pudding-chomeur"), make_card("azul")]
        g1.players[3].energy = 20
        g1.play_card(3, 0)
        self.assertEqual(len(g1.discard), 1)
        self.assertEqual(g1.ais[0]._get_live_known_trunk_top(g1, 0), ())
        self.assertIsNone(g1.ais[0]._known_trunk_state)
        sim1 = g1.ais[0]._prepare_observable_game(g1, 0)
        self.assertIsNone(sim1._known_trunk_state)

        # 2. Reshuffle starting from empty discard (`prev_dlen == 0`).
        g2 = _setup_game(
            1302,
            ["rouladen", "solo", "movie", "nap", "fajitas", "calzones"],
            [],
        )
        g2.players[0].hand = [make_card("read-the-radar")]
        g2.players[0].energy = 2
        g2.play_card(0, 0)
        g2.players[1].hand = [make_card("euchre-tournament")]
        g2.players[1].energy = 20
        g2.play_card(1, 0)
        g2.players[2].hand = [make_card("booby-prize") for _ in range(3)]
        g2.players[2].energy = 20
        for _ in range(3):
            g2.play_card(2, 0)
        self.assertEqual(g2.ais[0]._get_live_known_trunk_top(g2, 0), ())
        self.assertIsNone(g2.ais[0]._known_trunk_state)

        # 3. Exhausted tracker (`draws_since >= len(ordered_cards)`) clears state.
        g3 = _setup_game(
            1303,
            ["rouladen", "solo", "movie", "nap", "fajitas"],
            ["zumba", "azul"],
        )
        g3.players[0].hand = [make_card("read-the-radar")]
        g3.players[0].energy = 2
        g3.play_card(0, 0)
        g3.players[1].hand = [make_card("euchre-tournament")]
        g3.players[1].energy = 20
        g3.play_card(1, 0)
        g3.players[2].hand = [make_card("booby-prize")]
        g3.players[2].energy = 20
        g3.play_card(2, 0)
        self.assertEqual(g3.ais[0]._get_live_known_trunk_top(g3, 0), ())
        self.assertIsNone(g3.ais[0]._known_trunk_state)
        g3.trunk = [make_card("nap") for _ in range(3)]
        self.assertEqual(g3.ais[0]._get_live_known_trunk_top(g3, 0), ())

        class _TrashAllRandomAI(RandomAI):
            def choose_cards_to_discard(self, game, player_index, options):
                return list(range(len(options)))

        # 4a. Bohnanza shrinking opp.played_today before Read the Radar.
        g4 = _setup_game(
            1304,
            ["rouladen", "solo", "movie", "nap", "fajitas", "calzones"],
            [],
        )
        g4.ais[1] = _TrashAllRandomAI(random.Random(1305))
        g4.players[1].played_today = [make_card("nap"), make_card("nap")]
        g4.players[0].hand = [make_card("read-the-radar")]
        g4.players[0].energy = 2
        g4.play_card(0, 0)
        g4.players[1].hand = [make_card("bohnanza"), make_card("read-the-radar")]
        g4.players[1].energy = 20
        g4.play_card(1, 0)
        g4.play_card(1, 0)
        self.assertEqual(len(g4.players[1].played_today), 2)
        self.assertEqual(g4.ais[0]._get_live_known_trunk_top(g4, 0), ())
        self.assertIsNone(g4.ais[0]._known_trunk_state)

        # 4b. Read the Radar played by opponent then trashed by Bohnanza.
        g5 = _setup_game(
            1305,
            ["rouladen", "solo", "movie", "nap", "fajitas", "calzones"],
            [],
        )
        g5.ais[1] = _TrashAllRandomAI(random.Random(1306))
        g5.players[1].played_today = [make_card("nap")]
        g5.players[0].hand = [make_card("read-the-radar")]
        g5.players[0].energy = 2
        g5.play_card(0, 0)
        g5.players[1].hand = [make_card("read-the-radar"), make_card("bohnanza")]
        g5.players[1].energy = 20
        g5.play_card(1, 0)
        g5.play_card(1, 0)
        self.assertEqual(len(g5.players[1].played_today), 1)
        self.assertTrue(any(c.definition.slug == "read-the-radar" for c in g5.discard))
        self.assertEqual(g5.ais[0]._get_live_known_trunk_top(g5, 0), ())
        self.assertIsNone(g5.ais[0]._known_trunk_state)

        # 5. Subclass overriding _suitcase_pick1_bonus bypasses fast bm_cache.
        class CustomPick1Osmium(OsmiumAI):
            def _suitcase_pick1_bonus(self, game: Game, player_index: int) -> float:
                return 99.0 if game.players[player_index].energy >= 1 else 0.0

        g6 = _setup_game(1306, ["nap"] * 10, [])
        g6.players[0].hand = [make_card("nap")]
        g6.players[0].energy = 3
        for oi in range(1, 4):
            g6.players[oi].hand = [make_card("nap") for _ in range(3)]
        c_ai = CustomPick1Osmium(random.Random(1306))
        sim6 = c_ai._prepare_observable_game(g6, 0)
        bm_d: dict[tuple[int, int], float] = {}
        ftb_ctx6 = (
            1.0,
            tuple(0 for _ in range(4)),
            tuple(0 for _ in range(4)),
            tuple(0 for _ in range(4)),
            0,
            {},
            sim6.suitcase,
            tuple(c.instance_id for c in sim6.suitcase),
            (),
            tuple(sim6.cards_energy_costs(0, sim6.suitcase)),
            tuple(_get_card_meta(c) for c in sim6.suitcase),
            None,
            {},
            bm_d,
            sim6.players[0].tomorrow_cards,
        )
        v1 = c_ai._evaluate_terminal_day(
            sim6, 0, turns_from_root=1, _ftb_ctx=ftb_ctx6, _ctx_ok=True, _sc_ok=True
        )
        self.assertEqual(bm_d, {})
        ftb_ctx6[5].clear()
        sim6.players[0].energy = 0
        v2 = c_ai._evaluate_terminal_day(
            sim6, 0, turns_from_root=1, _ftb_ctx=ftb_ctx6, _ctx_ok=True, _sc_ok=True
        )
        self.assertEqual(bm_d, {})
        self.assertNotAlmostEqual(v1, v2, places=3)


if __name__ == "__main__":
    unittest.main()
