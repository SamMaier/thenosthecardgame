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
from thenos.ais.hydrogen import _HYDROGEN_RESERVE_EVAL_CACHE, _hydrogen_copy_game
from thenos.ais.osmium import (
    _OSMIUM_RESERVE_CACHE_MAX,
    _OSMIUM_RESERVE_EVAL_CACHE,
    _OSMIUM_SPEC_BY_SLUG,
    _ensure_osmium_caches,
    _osmium_reserve_score,
    _osmium_reserve_score_plain_specs,
)
from thenos.cards import make_card
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
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
        # Optimal: Sweet Lawn Chair (0) -> Colouring (1+2=3) -> Tres Fute (1+2=3) => 6 Fun.
        self.assertEqual(play_order, ["Sweet Lawn Chair", "Colouring", "Tres Fute"])
        self.assertEqual(sum(game.card_fun(0, c) for c in p.played_today), 6)

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

        # When able to play Date's First Nos AND Waterski (>= 3 Fun), Osmium plays them.
        p.energy = 12
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
        # 1 + 5 = 6 <= 7, so Nos Shirt does not return to hand.
        self.assertEqual(_projected_end_hand_size(game2, 0), 0)

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


if __name__ == "__main__":
    unittest.main()
