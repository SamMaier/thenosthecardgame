import copy
import os
import random
import subprocess
import sys
import unittest

from thenos.ai import (
    Hydrogen as HydrogenFromAIModule,
    HydrogenAI as HydrogenAIFromAIModule,
    HydrogenBrain as HydrogenBrainFromAIModule,
    HydrogenbrainAI as HydrogenbrainAIFromAIModule,
)
from thenos.ais import (
    BerylliumBrain,
    CarbonAI,
    Hydrogen,
    HydrogenAI,
    HydrogenBrain,
    HydrogenbrainAI,
)
from thenos.ais.hydrogen import (
    _HYDROGEN_COPY_FOOD_TITLES,
    _HYDROGEN_RESERVE_EVAL_CACHE,
    _hydrogen_copy_game_for_play,
    _hydrogen_quick_card_desirability,
)
from thenos.ais.hydrogenbrain import HydrogenAI as HydrogenAlias
from thenos.cards import make_card
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
from thenos.game import Game
from thenos.simulation import Competitor, simulate_games


class _FastHydrogenAI(HydrogenAI):
    BEAM_WIDTH = 5
    DRAFT_BEAM_WIDTH = 5
    BRANCH_LIMIT = 4
    DRAFT_SAMPLES = 2
    PLAY_SAMPLES = 3


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


def _hydrogen_game(seed: int = 41) -> Game:
    return Game(
        [],
        [HydrogenAI(random.Random(seed + idx)) for idx in range(4)],
        random.Random(seed + 100),
    )


class HydrogenAITests(unittest.TestCase):
    def test_alias_exports(self):
        self.assertIs(Hydrogen, HydrogenAI)
        self.assertIs(HydrogenBrain, HydrogenAI)
        self.assertIs(HydrogenbrainAI, HydrogenAI)
        self.assertIs(HydrogenFromAIModule, HydrogenAI)
        self.assertIs(HydrogenAIFromAIModule, HydrogenAI)
        self.assertIs(HydrogenBrainFromAIModule, HydrogenAI)
        self.assertIs(HydrogenbrainAIFromAIModule, HydrogenAI)
        self.assertIs(HydrogenAlias, HydrogenAI)
        self.assertIsNot(HydrogenAI, CarbonAI)

    def test_copy_food_titles_excludes_morning_coffee(self):
        self.assertIsInstance(_HYDROGEN_COPY_FOOD_TITLES, frozenset)
        self.assertNotIn("Morning Coffee", _HYDROGEN_COPY_FOOD_TITLES)
        self.assertIn("Beaver Burgers", _HYDROGEN_COPY_FOOD_TITLES)

    def test_cards_csv_tag_synergies(self):
        game = _hydrogen_game()
        game.day = 3
        player = game.players[0]
        player.energy = 7
        ai = game.ais[0]

        cliff = make_card("cliff-climbing")
        player.hand = [cliff, make_card("scrabble")]
        cliff_out_q = _hydrogen_quick_card_desirability(game, 0, cliff, {})
        cliff_out_d = ai._draft_synergy_adjustment(game, 0, cliff, {})
        player.hand = [cliff, make_card("zumba")]
        cliff_ex_q = _hydrogen_quick_card_desirability(game, 0, cliff, {})
        cliff_ex_d = ai._draft_synergy_adjustment(game, 0, cliff, {})
        self.assertGreater(cliff_out_q, cliff_ex_q)
        self.assertGreater(cliff_out_d, cliff_ex_d)

        volleyball = make_card("water-volleyball")
        player.hand = [volleyball, make_card("zumba")]
        vb_ex_q = _hydrogen_quick_card_desirability(game, 0, volleyball, {})
        vb_ex_d = ai._draft_synergy_adjustment(game, 0, volleyball, {})
        player.hand = [volleyball, make_card("scrabble")]
        vb_out_q = _hydrogen_quick_card_desirability(game, 0, volleyball, {})
        vb_out_d = ai._draft_synergy_adjustment(game, 0, volleyball, {})
        self.assertGreater(vb_ex_q, vb_out_q)
        self.assertGreater(vb_ex_d, vb_out_d)

        bracelet = make_card("bracelet-making")
        player.hand = [bracelet, make_card("epic-prank")]
        br_soc_q = _hydrogen_quick_card_desirability(game, 0, bracelet, {})
        br_soc_d = ai._draft_synergy_adjustment(game, 0, bracelet, {})
        player.hand = [bracelet, make_card("cheesy-phone-game")]
        br_rel_q = _hydrogen_quick_card_desirability(game, 0, bracelet, {})
        br_rel_d = ai._draft_synergy_adjustment(game, 0, bracelet, {})
        self.assertGreater(br_soc_q, br_rel_q)
        self.assertGreater(br_soc_d, br_rel_d)

    def test_first_to_bed_value_conditions_and_turn_decay(self):
        game = Game(
            create_default_deck(daily_conditions=True),
            [HydrogenAI(random.Random(41 + idx)) for idx in range(4)],
            random.Random(141),
            daily_conditions=True,
        )
        game.setup()
        game.start_day()
        game.day = 1
        ai = game.ais[0]

        self.assertAlmostEqual(
            ai._first_to_bed_value(game, 0, turns_from_root=0),
            HydrogenAI.FIRST_TO_BED_BONUS,
        )
        decay_vals = [
            ai._first_to_bed_value(game, 0, turns_from_root=t) for t in range(4)
        ]
        for prev_v, next_v in zip(decay_vals, decay_vals[1:]):
            self.assertGreater(prev_v, next_v)
        self.assertGreater(decay_vals[-1], 0.0)

        game.players[1].asleep = True
        self.assertEqual(ai._first_to_bed_value(game, 0), 0.0)

        game.players[1].asleep = False
        game.day = 6
        self.assertEqual(ai._first_to_bed_value(game, 0), 0.0)

    def test_diminishing_reserve_weights_and_cache_subclass_isolation(self):
        for weights in (
            HydrogenAI.RESERVE_SLOT_WEIGHTS_COMBO,
            HydrogenAI.RESERVE_SLOT_WEIGHTS_PLAIN,
        ):
            self.assertEqual(len(weights), 4)
            for earlier, later in zip(weights, weights[1:]):
                self.assertGreater(earlier, later)
            self.assertGreater(weights[-1], HydrogenAI.RESERVE_SLOT_TAIL_WEIGHT)
        self.assertGreater(HydrogenAI.RESERVE_SLOT_TAIL_WEIGHT, 0.0)

        class _AltReserveHydrogen(HydrogenAI):
            RESERVE_TOMORROW_WEIGHT = 0.45
            RESERVE_SLOT_WEIGHTS_PLAIN = (0.7, 0.4, 0.2, 0.1)

        game = _hydrogen_game()
        game.day = 3
        game.players[0].hand = [
            make_card("waterski"),
            make_card("zumba"),
            make_card("epic-prank"),
        ]
        default_val = HydrogenAI()._evaluate_tomorrow_and_reserve_fast(
            game, 0, [], days_left=3
        )
        alt_val = _AltReserveHydrogen()._evaluate_tomorrow_and_reserve_fast(
            game, 0, [], days_left=3
        )
        self.assertGreater(default_val, alt_val)

        game.players[0].hand = [_dummy_card(901, "Synthetic Same Slug", 1, 2)]
        low_val = game.ais[0]._evaluate_tomorrow_and_reserve_fast(
            game, 0, [], days_left=3
        )
        game.players[0].hand = [_dummy_card(902, "Synthetic Same Slug", 1, 9)]
        high_val = game.ais[0]._evaluate_tomorrow_and_reserve_fast(
            game, 0, [], days_left=3
        )
        self.assertGreater(high_val, low_val + 2.0)

        self.assertNotIn("_reserve_eval_cache", game.ais[0].__dict__)
        self.assertIs(
            game.ais[0]._reserve_eval_cache, _HYDROGEN_RESERVE_EVAL_CACHE
        )
        self.assertIs(
            copy.deepcopy(game).ais[0]._reserve_eval_cache,
            _HYDROGEN_RESERVE_EVAL_CACHE,
        )
        self.assertIs(
            game.copy_for_simulation().ais[0]._reserve_eval_cache,
            _HYDROGEN_RESERVE_EVAL_CACHE,
        )

    def test_subsearch_bed_turn_accounting_and_trunk_reshuffle_guard(self):
        game = Game(
            create_default_deck(daily_conditions=True),
            [HydrogenAI(random.Random(61 + i)) for i in range(4)],
            random.Random(161),
            daily_conditions=True,
        )
        game.setup()
        game.start_day()
        game.day = 1
        ai = game.ais[0]

        child = _hydrogen_copy_game_for_play(game, 0)
        child.players[0].hand[0] = make_card("waterski")
        child.play_card(0, 0)
        _, _, table = ai._search_best_day_plan_with_table(
            child,
            0,
            beam_width=4,
            max_plays=1,
            allow_bed_now=True,
            base_turns_offset=1,
            start_extra_active=False,
        )
        self.assertAlmostEqual(
            table[None][0],
            ai._evaluate_terminal_day(
                child, 0, turns_from_root=1, root_skipped_turns=0
            ),
        )

        # Returning cards records known trunk top; reshuffling (increasing len) clears it permanently.
        expected_len_before = len(game.trunk)
        game.return_cards_to_trunk_top(0, [make_card("waterski")])
        self.assertEqual(len(ai._get_live_known_trunk_top(game, 0)), 1)
        extras = [make_card("zumba"), make_card("epic-prank")]
        game.trunk.extend(extras)
        self.assertEqual(ai._get_live_known_trunk_top(game, 0), ())
        for _ in extras:
            game.trunk.pop()
        self.assertGreaterEqual(len(game.trunk), expected_len_before)
        self.assertEqual(ai._get_live_known_trunk_top(game, 0), ())

    def test_observable_state_invariance_with_daily_conditions(self):
        game = Game(
            create_default_deck(daily_conditions=True),
            [HydrogenAI(random.Random(40 + i)) for i in range(4)],
            random.Random(99),
            daily_conditions=True,
        )
        game.setup()
        game.start_day()
        acting_index = 1
        self.assertTrue(
            all(c is None for c in game.known_daily_conditions(acting_index))
        )
        game.players[2].played_today = [
            make_card("waterski"),
            make_card("epic-prank"),
        ]
        orig_played = list(game.players[acting_index].played_today)
        playable = tuple(game.playable_hand_indices(acting_index))

        def capture_choices():
            game.ais[acting_index].rng = random.Random(777)
            sample = game.ais[acting_index]._prepare_observable_game(
                game, acting_index
            )
            pick = HydrogenAI(random.Random(555)).choose_suitcase_card(
                game, acting_index, game.suitcase
            )
            bed = HydrogenAI(random.Random(556)).choose_to_go_to_bed(
                game, acting_index, playable
            )
            play = HydrogenAI(random.Random(557)).choose_card_to_play(
                game, acting_index, playable
            )
            extra = HydrogenAI(random.Random(558)).choose_extra_card_to_play(
                game, acting_index, playable
            )
            try:
                game.players[acting_index].played_today = [
                    *orig_played,
                    make_card("istanbul"),
                ]
                disc = tuple(
                    HydrogenAI(random.Random(559)).choose_cards_to_discard(
                        game, acting_index, game.players[acting_index].hand
                    )
                )
                game.players[acting_index].played_today = [
                    *orig_played,
                    make_card("wedding-anniversary"),
                ]
                copy_choice = HydrogenAI(random.Random(560)).choose_card_to_copy(
                    game, acting_index, game.players[2].played_today
                )
                kayak = make_card("kayak")
                game.players[acting_index].played_today = [*orig_played, kayak]
                spend = HydrogenAI(random.Random(561)).choose_energy_to_spend(
                    game, acting_index, kayak, 3
                )
            finally:
                game.players[acting_index].played_today = list(orig_played)
            opt = HydrogenAI(random.Random(562)).choose_optional_action(
                game, acting_index, "Quacks: draw another card?"
            )
            return (
                [c.instance_id for c in sample.trunk],
                [
                    [c.instance_id for c in sample.players[i].hand]
                    for i in (0, 2, 3)
                ],
                pick,
                bed,
                play,
                extra,
                disc,
                copy_choice,
                spend,
                opt,
            )

        before = capture_choices()

        # Scramble strictly unobservable state: trunk order, opponent hand card swap,
        # unrevealed _condition_deck order, and live game.rng state.
        game.trunk.reverse()
        swapped_from_trunk = game.trunk.pop()
        swapped_from_opp = game.players[0].hand.pop()
        game.trunk.append(swapped_from_opp)
        game.players[0].hand.append(swapped_from_trunk)
        game._condition_deck.reverse()
        for _ in range(20):
            game.rng.random()

        after = capture_choices()
        self.assertEqual(before, after)

    def test_lookahead_never_mutates_live_card_markers(self):
        game = _hydrogen_game()
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

    def test_cow_cloning_and_cache_isolation(self):
        game = _hydrogen_game()
        game.day = 2
        ai = game.ais[0]

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
        game.players[1].played_today.append(make_card("chuck-a-frisbee"))

        base_sim = ai._prepare_observable_game(game, 0)
        orig_opp2_target = base_sim.players[2].played_today[0].markers["target_card"]
        orig_opp1_hand_len = len(base_sim.players[1].hand)

        branch = _hydrogen_copy_game_for_play(base_sim, 0)
        branch.go_to_bed(1)
        self.assertIs(
            base_sim.players[2].played_today[0].markers["target_card"],
            orig_opp2_target,
        )
        self.assertIs(
            base_sim.players[2].played_today[1].markers["target_card"],
            orig_opp2_target,
        )
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

        branch.end_day()
        self.assertTrue(base_sim.players[0].tomorrow_cards[0].is_tomorrow)
        self.assertEqual(len(base_sim.players[1].hand), orig_opp1_hand_len)
        self.assertFalse(branch._has_card_ref)

        game2 = _hydrogen_game()
        game2.day = 3
        game2.players[1].played_today = [make_card("nos-book")]
        game2.players[0].played_today = [
            make_card("waterski"),
            make_card("wedding-anniversary"),
        ]
        game2.players[0].energy = 10
        sim2 = ai._prepare_observable_game(game2, 0)
        sim2.copy_card_effect(
            0, sim2.players[1].played_today[0], sim2.players[0].played_today[1]
        )
        self.assertTrue(sim2._has_card_ref)
        sim2_child = _hydrogen_copy_game_for_play(sim2, 0)
        self.assertEqual(sim2_child.score_played_today_fast(0)[0], 9)

    def test_pythonhashseed_determinism(self):
        snippet = (
            "import random; "
            "from tests.test_hydrogen_ai import _FastHydrogenAI; "
            "from thenos.ais.hydrogen import _hydrogen_free_play_value, _hydrogen_quick_card_desirability; "
            "from thenos.cards import make_card; "
            "from thenos.cards.base import CardBehavior, CardDefinition, CardInstance; "
            "from thenos.cards.catalog import create_default_deck; "
            "from thenos.game import Game; "
            "g = Game(create_default_deck(daily_conditions=True), [_FastHydrogenAI(random.Random(10+i)) for i in range(4)], random.Random(77), daily_conditions=True, characters=True); "
            "g.setup(); g.day = 2; ai = g.ais[0]; "
            "c = make_card('evening-on-the-dock'); c.markers['_all_tags'] = True; "
            "w = {'Board Game': 1.3, 'Exercise': 0.7, 'Food': -0.4, 'Indoors': 1.1, 'Item': 0.9, 'Outdoors': 1.7, 'Relax': 2.1, 'Social': 0.8}; "
            "dual = lambda i: CardInstance(800+i, CardDefinition(f'd{i}', f'Dual {i}', frozenset({'Food', 'Outdoors'}), 1, 2, CardBehavior())); "
            "p0 = g.players[0]; orig_hand = list(p0.hand); orig_played = list(p0.played_today); "
            "p0.hand = [make_card('cliff-climbing'), make_card('ice-wine'), dual(1), dual(2)]; "
            "p0.played_today = [make_card('colouring')]; "
            "vals = ("
            "repr(_hydrogen_quick_card_desirability(g, 0, c, w)), "
            "repr(_hydrogen_free_play_value(g, 0, c, w)), "
            "repr(ai._draft_synergy_adjustment(g, 0, c, w)), "
            "repr(ai._evaluate_tomorrow_and_reserve_fast(g, 0, [make_card('colouring')], days_left=3))"
            "); "
            "p0.hand = orig_hand; p0.played_today = orig_played; ai._reserve_eval_cache.clear(); "
            "res = g.run(); "
            "print(vals, res.scores)"
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
                HydrogenAI(random.Random(300)),
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
            Competitor("Hydrogen", _FastHydrogenAI),
            Competitor("Beryllium", BerylliumBrain),
            Competitor("Hydrogen", _FastHydrogenAI),
            Competitor("Beryllium", BerylliumBrain),
        )
        serial = simulate_games(
            2,
            seed=20261299,
            competitors=competitors,
            rotate_seats=True,
            workers=1,
        )
        parallel = simulate_games(
            2,
            seed=20261299,
            competitors=competitors,
            rotate_seats=True,
            workers=2,
        )
        self.assertEqual(serial.rows(), parallel.rows())
        self.assertEqual(serial.ais["Hydrogen"], parallel.ais["Hydrogen"])
        self.assertEqual(serial.ais["Beryllium"], parallel.ais["Beryllium"])


if __name__ == "__main__":
    unittest.main()
