import random
import unittest

from thenos.ais import (
    Beryllium,
    BerylliumAI,
    BerylliumBrain,
    BerylliumbrainAI,
)
from thenos.ais.beryllium import (
    _beryllium_build_profile,
    _beryllium_free_play_value,
    _contextual_copy_value,
    _contextual_hand_keep_value,
    _effective_card_title,
    _select_istanbul_discards,
    _wanted_tags_for_player,
)
from thenos.cards import make_card
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
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


def _beryllium_game(seed: int = 23) -> Game:
    return Game(
        [],
        [BerylliumBrain(random.Random(seed + idx)) for idx in range(4)],
        random.Random(seed + 100),
    )


class BerylliumBrainAITests(unittest.TestCase):
    def test_alias_exports(self):
        self.assertIs(Beryllium, BerylliumBrain)
        self.assertIs(BerylliumAI, BerylliumBrain)
        self.assertIs(BerylliumbrainAI, BerylliumBrain)

    def test_observable_game_preserves_acting_hand_and_resamples_hidden_zones(self):
        game = Game(
            create_default_deck(),
            [BerylliumBrain(random.Random(40 + idx)) for idx in range(4)],
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

    def test_effective_definition_profile_avoids_copy_cache_collision(self):
        game = _beryllium_game()
        game.players[0].energy = 10
        sweet_lawn = make_card("sweet-lawn-chair")
        doxology = make_card("doxology")
        wa_relax = make_card("wedding-anniversary")
        wa_social = make_card("wedding-anniversary")
        game.players[0].played_today = [
            sweet_lawn,
            doxology,
            wa_relax,
            wa_social,
        ]
        game.copy_card_effect(0, sweet_lawn, wa_relax, pay_source_cost=True)
        game.copy_card_effect(0, doxology, wa_social, pay_source_cost=True)

        prof_relax = _beryllium_build_profile(wa_relax)
        prof_social = _beryllium_build_profile(wa_social)

        self.assertEqual(prof_relax.target_tags, frozenset({"Relax"}))
        self.assertEqual(prof_social.target_tags, frozenset({"Social"}))
        self.assertEqual(_effective_card_title(wa_relax), "Sweet Lawn Chair")
        self.assertEqual(_effective_card_title(wa_social), "Doxology")

    def test_leftovers_and_wakeboard_valuation(self):
        game = _beryllium_game()
        game.day = 3
        player = game.players[0]
        player.energy = 2
        ice_wine = make_card("ice-wine")
        doritos = make_card("doritos")
        fajitas = make_card("fajitas")
        leftovers = make_card("leftovers")
        player.played_today = [fajitas, doritos, ice_wine, leftovers]
        player.hand = [
            _dummy_card(1, "Snack", 1, 2, frozenset({"Food"})),
        ]

        # Leftovers is free (pay_cost=False) and Ice Wine scales off 3 Food cards played before Leftovers
        iw_val = _contextual_copy_value(
            game, 0, ice_wine, leftovers, pay_cost=False
        )
        doritos_val = _contextual_copy_value(
            game, 0, doritos, leftovers, pay_cost=False
        )
        self.assertGreater(iw_val, doritos_val)

        # Wakeboard free-play value is not penalized by prior cards' costs
        wakeboard = make_card("wakeboard")
        self.assertGreaterEqual(
            _beryllium_free_play_value(game, 0, wakeboard), 9.0
        )

    def test_bohnanza_preserves_ongoing_scaler_and_trashes_spent_card(self):
        game = _beryllium_game()
        game.day = 3
        player = game.players[0]
        player.energy = 4
        spent = _dummy_card(1, "Spent One", 1, 1)
        high_end_red = make_card("high-end-red")
        bohnanza = make_card("bohnanza")
        player.played_today = [spent, high_end_red, bohnanza]
        player.hand = [
            _dummy_card(2, "Food A", 1, 2, frozenset({"Food"})),
            _dummy_card(3, "Food B", 1, 2, frozenset({"Food"})),
            _dummy_card(4, "Food C", 1, 2, frozenset({"Food"})),
        ]

        discards = game.ais[0].choose_cards_to_discard(
            game, 0, [spent, high_end_red]
        )
        self.assertIn(0, discards)
        self.assertNotIn(1, discards)

    def test_lookahead_never_mutates_live_card_markers(self):
        game = _beryllium_game()
        game.day = 3
        player = game.players[0]
        player.energy = 7
        player.hand = [
            make_card("scout-the-other-cottages"),
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

    def test_serial_and_parallel_simulation_parity(self):
        competitors = (
            Competitor("Beryllium", BerylliumBrain),
            Competitor("Beryllium", BerylliumBrain),
            Competitor("Beryllium", BerylliumBrain),
            Competitor("Beryllium", BerylliumBrain),
        )
        serial = simulate_games(
            2,
            seed=20261109,
            competitors=competitors,
            rotate_seats=False,
            workers=1,
        )
        parallel = simulate_games(
            2,
            seed=20261109,
            competitors=competitors,
            rotate_seats=False,
            workers=2,
        )
        self.assertEqual(serial.rows(), parallel.rows())
        self.assertEqual(serial.ais["Beryllium"], parallel.ais["Beryllium"])

    def test_copied_tomorrow_reserve_and_sim_isolation(self):
        game = _beryllium_game()
        game.day = 2
        player = game.players[0]
        player.energy = 0
        photo_shoot = make_card("photo-shoot")
        wa_photo = make_card("wedding-anniversary")
        player.played_today = [photo_shoot, wa_photo]
        game.copy_card_effect(0, photo_shoot, wa_photo, pay_source_cost=False)
        player.hand = [
            make_card("paddleboat"),
            _dummy_card(1, "Cheap Relax", 1, 2, frozenset({"Relax"})),
        ]

        ai = game.ais[0]
        val_copied = ai._evaluate_tomorrow_and_reserve_fast(
            game, 0, [wa_photo], days_left=4
        )
        val_empty = ai._evaluate_tomorrow_and_reserve_fast(
            game, 0, [], days_left=4
        )
        self.assertGreater(val_copied, val_empty + 4.0)

        player.hand = []
        self.assertAlmostEqual(
            ai._evaluate_tomorrow_and_reserve_fast(
                game, 0, [wa_photo], days_left=4
            ),
            5.3,
        )

    def test_istanbul_avoids_wasted_discards_when_sufficient_or_blocked(self):
        # Already enough Energy today + tomorrow reserve card -> 0 discards
        game = _beryllium_game()
        game.day = 2
        player = game.players[0]
        player.energy = 4
        player.played_today = [make_card("istanbul")]
        player.hand = [
            make_card("splendor"),
            make_card("nos-shirt"),
            make_card("morning-walk"),
        ]
        self.assertEqual(
            tuple(game.ais[0].choose_cards_to_discard(game, 0, player.hand)),
            (),
        )

        # Sleep In blocks Energy gain -> 0 discards, and AI avoids playing Istanbul after Sleep In
        game2 = _beryllium_game()
        game2.day = 2
        p2 = game2.players[0]
        p2.energy = 3
        p2.played_today = [make_card("sleep-in"), make_card("istanbul")]
        p2.hand = [make_card("splendor"), make_card("treat-cereal")]
        self.assertEqual(
            tuple(game2.ais[0].choose_cards_to_discard(game2, 0, p2.hand)),
            (),
        )

    def test_istanbul_keeps_expensive_day6_payoff_and_scales_past_three_discards(self):
        # Day 6: discard 2 cheap cards (+6E -> 8E) to keep & play Wakeboard (cost 8, 10 Fun)
        game = _beryllium_game()
        game.day = 6
        player = game.players[0]
        player.energy = 2
        player.played_today = [make_card("istanbul")]
        player.hand = [
            make_card("wakeboard"),
            make_card("splendor"),
            make_card("non-fiction-book"),
        ]
        discards = tuple(
            game.ais[0].choose_cards_to_discard(game, 0, player.hand)
        )
        self.assertEqual(set(discards), {1, 2})

        # Day 6: discard 5 dead/weak cards (>3) for +15E to play BOTH Wakeboard (8) + Risk (7)
        game_big = _beryllium_game()
        game_big.day = 6
        p_big = game_big.players[0]
        p_big.energy = 0
        p_big.played_today = [make_card("istanbul")]
        p_big.hand = [
            make_card("wakeboard"),
            make_card("risk"),
            make_card("morning-run"),
            make_card("morning-bike"),
            make_card("morning-walk"),
            make_card("treat-cereal"),
            make_card("bug-spray"),
        ]
        big_discards = tuple(
            game_big.ais[0].choose_cards_to_discard(
                game_big, 0, p_big.hand
            )
        )
        self.assertEqual(set(big_discards), {2, 3, 4, 5, 6})

        # Day 6 knapsack pair: discard Risk (7E, 8F) to fund Waterski + Waterski (10E, 12F)
        game_pair = _beryllium_game()
        game_pair.day = 6
        p_pair = game_pair.players[0]
        p_pair.energy = 7
        p_pair.played_today = [make_card("istanbul")]
        p_pair.hand = [
            make_card("risk"),
            make_card("waterski"),
            make_card("waterski"),
        ]
        self.assertEqual(
            tuple(
                game_pair.ais[0].choose_cards_to_discard(
                    game_pair, 0, p_pair.hand
                )
            ),
            (0,),
        )

        # Bring a Friend on board: rollout evaluator sees 2x Fun on hand cards, not just 2x cost
        game_baf = _beryllium_game()
        game_baf.day = 3
        p_baf = game_baf.players[0]
        p_baf.energy = 0
        p_baf.played_today = [
            make_card("bring-a-friend"),
            make_card("istanbul"),
        ]
        p_baf.hand = [
            make_card("azul"),  # 6E cost, 8 Fun with Bring a Friend
            make_card("morning-walk"),
            make_card("morning-bike"),
        ]
        self.assertEqual(
            set(_select_istanbul_discards(game_baf, 0, p_baf.hand)),
            {1, 2},
        )

        # Wakesurf on board: can discard the final card in hand (0 cards at EOD = +5 Fun)
        game_ws = _beryllium_game()
        game_ws.day = 4
        p_ws = game_ws.players[0]
        p_ws.energy = 0
        p_ws.played_today = [make_card("wakesurf"), make_card("istanbul")]
        p_ws.hand = [make_card("morning-walk")]
        self.assertEqual(
            tuple(game_ws.ais[0].choose_cards_to_discard(game_ws, 0, p_ws.hand)),
            (0,),
        )

        # Day 5 reserve & Day 6 keep value recognize Istanbul + Wakeboard synergy
        game_res = _beryllium_game()
        game_res.day = 5
        p_res = game_res.players[0]
        p_res.hand = [make_card("istanbul"), make_card("wakeboard")]
        self.assertGreater(
            game_res.ais[0]._evaluate_tomorrow_and_reserve_fast(
                game_res, 0, [], days_left=1
            ),
            2.5,
        )
        game_d6_keep = _beryllium_game()
        game_d6_keep.day = 6
        p_d6 = game_d6_keep.players[0]
        p_d6.energy = 5
        wake_d6 = make_card("wakeboard")
        walk_d6 = make_card("morning-walk")
        p_d6.hand = [
            make_card("istanbul"),
            wake_d6,
            walk_d6,
            make_card("morning-bike"),
        ]
        wanted = _wanted_tags_for_player(p_d6)
        self.assertGreater(
            _contextual_hand_keep_value(game_d6_keep, 0, wake_d6, wanted),
            _contextual_hand_keep_value(game_d6_keep, 0, walk_d6, wanted),
        )

        # Full turn execution: chooses Istanbul first to ramp into Wakeboard
        game_turn = _beryllium_game()
        game_turn.day = 6
        game_turn._is_setup = True
        game_turn.trunk = [
            _dummy_card(200 + i, f"T{i}", 2, 2) for i in range(12)
        ]
        p_turn = game_turn.players[0]
        p_turn.energy = 5
        p_turn.hand = [
            make_card("wakeboard"),
            make_card("istanbul"),
            make_card("morning-run"),
            make_card("morning-bike"),
        ]
        for opp in game_turn.players[1:]:
            opp.asleep = True
        game_turn.playing_phase()
        self.assertEqual(
            [c.title for c in p_turn.played_today],
            ["Istanbul", "Wakeboard"],
        )


if __name__ == "__main__":
    unittest.main()
