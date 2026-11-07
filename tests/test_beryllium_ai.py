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
    _effective_card_title,
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


if __name__ == "__main__":
    unittest.main()
