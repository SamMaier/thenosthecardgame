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
    BerylliumBrain,
    HydrogenAI,
    Osmium,
    OsmiumAI,
    OsmiumBrain,
    OsmiumbrainAI,
)
from thenos.ais.hydrogen import _HYDROGEN_RESERVE_EVAL_CACHE, _hydrogen_copy_game
from thenos.ais.osmium import (
    _OSMIUM_RESERVE_CACHE_MAX,
    _OSMIUM_RESERVE_EVAL_CACHE,
)
from thenos.cards import make_card
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
from thenos.characters import CHARACTERS
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
            g.players[0].played_today.append(make_card("cities-and-knights"))
            opt = ai.choose_optional_action(g, 0, "discard one card for Cities and Knights")
            disc = ai.choose_card_to_discard(g, 0, tuple(g.players[0].hand))
            return pick, play, disc, opt

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
        player.played_today.append(make_card("cities-and-knights"))
        _ = ai.choose_optional_action(game, 0, "discard one card for Cities and Knights")
        player.played_today.pop()

        self.assertEqual(game.rng.getstate(), rng_before)
        for c in player.hand:
            self.assertEqual(c.markers, {})
        for c in game.suitcase:
            self.assertEqual(c.markers, {})
        for c in game.trunk:
            self.assertEqual(c.markers, {})

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
        p.energy = 2
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

        # Child must not receive Colouring's after_all_tags scaler synergy bonus.
        p.hand = [make_card("carcassonne"), make_card("movie")]
        without_colouring = ai._evaluate_tomorrow_and_reserve_fast(
            game, 0, [], days_left=1
        )
        p.hand = [make_card("carcassonne"), make_card("movie"), make_card("colouring")]
        with_colouring = ai._evaluate_tomorrow_and_reserve_fast(
            game, 0, [], days_left=1
        )
        self.assertLessEqual(with_colouring, without_colouring)

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
            "val = ai._evaluate_tomorrow_and_reserve_fast(g, 0, [make_card('colouring')], days_left=3); "
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
                BerylliumBrain(random.Random(301)),
                BerylliumBrain(random.Random(302)),
                BerylliumBrain(random.Random(303)),
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
            Competitor("Beryllium", BerylliumBrain),
            Competitor("Osmium", _FastOsmiumAI),
            Competitor("Beryllium", BerylliumBrain),
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
        self.assertEqual(serial.ais["Beryllium"], parallel.ais["Beryllium"])


if __name__ == "__main__":
    unittest.main()
