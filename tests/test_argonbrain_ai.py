import random
import unittest

from thenos.ais import ArgonBrain, ArgonbrainAI
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


def _argon_game(seed: int = 17) -> Game:
    return Game(
        [],
        [ArgonBrain(random.Random(seed + idx)) for idx in range(4)],
        random.Random(seed + 100),
    )


class ArgonBrainAITests(unittest.TestCase):
    def test_alias_export(self):
        self.assertIs(ArgonbrainAI, ArgonBrain)

    def test_observable_game_preserves_acting_hand_and_resamples_hidden_zones(self):
        game = Game(
            create_default_deck(),
            [ArgonBrain(random.Random(40 + idx)) for idx in range(4)],
            random.Random(99),
        )
        game.setup()
        acting_index = 1
        live_hand_ids = [c.instance_id for c in game.players[acting_index].hand]
        live_trunk_ids = [c.instance_id for c in game.trunk]

        sample = game.ais[acting_index]._prepare_observable_game(game, acting_index)

        self.assertEqual(
            [c.instance_id for c in sample.players[acting_index].hand],
            live_hand_ids,
        )
        self.assertEqual([c.instance_id for c in game.trunk], live_trunk_ids)

        # Hidden trunk order and opponent hand allocation must not affect resampled game
        game.ais[acting_index].rng = random.Random(777)
        sample_a = game.ais[acting_index]._prepare_observable_game(game, acting_index)

        game.trunk.reverse()
        swapped_from_trunk = game.trunk.pop()
        swapped_from_opp = game.players[0].hand.pop()
        game.trunk.append(swapped_from_opp)
        game.players[0].hand.append(swapped_from_trunk)

        game.ais[acting_index].rng = random.Random(777)
        sample_b = game.ais[acting_index]._prepare_observable_game(game, acting_index)

        self.assertEqual(
            [c.instance_id for c in sample_a.trunk],
            [c.instance_id for c in sample_b.trunk],
        )
        for opp_idx in (0, 2, 3):
            self.assertEqual(
                [c.instance_id for c in sample_a.players[opp_idx].hand],
                [c.instance_id for c in sample_b.players[opp_idx].hand],
            )

    def test_drafts_synergy_card_matching_held_scaler_without_mutating_game(self):
        game = _argon_game()
        game.day = 2
        player = game.players[0]
        player.energy = 7
        player.hand = [make_card("sweet-lawn-chair")]
        game.suitcase = [
            _dummy_card(10, "Plain Two", 2, 2, frozenset()),
            _dummy_card(11, "Relax Two", 2, 2, frozenset({"Relax"})),
            _dummy_card(12, "Plain One", 1, 1, frozenset()),
            _dummy_card(13, "Plain Three", 3, 3, frozenset()),
        ]
        game.trunk = [
            _dummy_card(100 + i, f"Trunk {i}", 2, 2) for i in range(16)
        ]
        before_suitcase = tuple(game.suitcase)
        before_hand = tuple(player.hand)

        choice = game.ais[0].choose_suitcase_card(game, 0, game.suitcase)

        self.assertEqual(choice, 1)
        self.assertEqual(tuple(game.suitcase), before_suitcase)
        self.assertEqual(tuple(player.hand), before_hand)

    def test_holds_unpartnered_scaler_early_and_plays_assembled_synergy_combo(self):
        game = _argon_game()
        game.day = 1
        player = game.players[0]
        player.energy = 7
        player.hand = [make_card("high-end-red")]
        game.suitcase = [
            _dummy_card(10 + i, f"Suitcase {i}", 2, 2) for i in range(4)
        ]
        game.trunk = [
            _dummy_card(100 + i, f"Trunk {i}", 2, 2) for i in range(20)
        ]

        # Holding High-End Red alone on Day 1 beats playing it for 0 Fun
        self.assertTrue(game.ais[0].choose_to_go_to_bed(game, 0, (0,)))

        # With Food cards in hand, ArgonBrain plays High-End Red first to buff them
        player.hand = [
            _dummy_card(2, "Snack One", 1, 2, frozenset({"Food"})),
            make_card("high-end-red"),
            _dummy_card(3, "Snack Two", 1, 2, frozenset({"Food"})),
        ]
        self.assertFalse(game.ais[0].choose_to_go_to_bed(game, 0, (0, 1, 2)))
        first_play = game.ais[0].choose_card_to_play(game, 0, (0, 1, 2))
        self.assertEqual(first_play, 1)

    def test_ignores_tomorrow_only_card_on_final_day(self):
        game = _argon_game()
        game.day = 6
        player = game.players[0]
        player.energy = 7
        player.hand = [make_card("dates-first-nos")]
        game.suitcase = [
            _dummy_card(10 + i, f"Suitcase {i}", 2, 2) for i in range(4)
        ]
        game.trunk = [
            _dummy_card(100 + i, f"Trunk {i}", 2, 2) for i in range(20)
        ]

        self.assertTrue(game.ais[0].choose_to_go_to_bed(game, 0, (0,)))

    def test_lookahead_and_optional_actions_never_mutate_live_card_markers(self):
        game = _argon_game()
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

        _ = game.ais[0].choose_to_go_to_bed(game, 0, (0, 1, 2, 3))
        _ = game.ais[0].choose_card_to_play(game, 0, (0, 1, 2, 3))

        for c in player.hand:
            self.assertEqual(c.markers, {})
        for c in game.suitcase:
            self.assertEqual(c.markers, {})
        for c in game.trunk:
            self.assertEqual(c.markers, {})

        quacks = make_card("quacks")
        player.played_today = [quacks]
        self.assertTrue(
            game.ais[0].choose_optional_action(game, 0, "Quacks: draw another card?")
        )
        self.assertEqual(quacks.markers, {})

        from thenos.ais.argonbrain import _EVAL_AI_SINGLETON, _build_profile

        self.assertEqual(
            _build_profile(make_card("make-plans")).immediate_cards_gained, 2
        )
        self.assertEqual(
            _build_profile(make_card("pudding-chomeur")).immediate_energy_gain, 2
        )
        self.assertEqual(
            _build_profile(make_card("puerto-rico")).conditional_fun_bonus, 3
        )
        self.assertEqual(
            _build_profile(make_card("cities-and-knights")).conditional_fun_bonus, 3
        )
        self.assertEqual(
            tuple(_EVAL_AI_SINGLETON.order_daily_conditions(game, 0, ("a", "b", "c"))),
            (0, 1, 2),
        )

    def test_tell_a_story_and_assorted_cutlery_combo_plays_expensive_card(self):
        game = _argon_game()
        game.day = 3
        player = game.players[0]
        player.energy = 5
        player.hand = [
            make_card("assorted-cutlery"),
            make_card("tell-a-story"),
            make_card("adventure-race"),
            make_card("7-wonders"),
        ]
        game.suitcase = [
            _dummy_card(10 + i, f"Suitcase {i}", 2, 2) for i in range(4)
        ]
        game.trunk = [
            _dummy_card(100 + i, f"Trunk {i}", 2, 2) for i in range(16)
        ]

        # First play should be Tell a Story (index 1), placing Adventure Race on Trunk top
        self.assertFalse(game.ais[0].choose_to_go_to_bed(game, 0, (0, 1, 3)))
        first_choice = game.ais[0].choose_card_to_play(game, 0, (0, 1, 3))
        self.assertEqual(first_choice, 1)
        game.play_card(0, first_choice)
        self.assertEqual(game.trunk[-1].title, "Adventure Race")

        # Second play should be Assorted Cutlery (now index 0), playing Adventure Race for free
        playable = game.playable_hand_indices(0)
        self.assertFalse(game.ais[0].choose_to_go_to_bed(game, 0, playable))
        second_choice = game.ais[0].choose_card_to_play(game, 0, playable)
        self.assertEqual(player.hand[second_choice].title, "Assorted Cutlery")
        game.play_card(0, second_choice)
        self.assertEqual(
            [c.title for c in player.played_today],
            ["Tell a Story", "Assorted Cutlery", "Adventure Race"],
        )

    def test_read_the_radar_sets_up_quacks_and_poker_with_known_trunk_top(self):
        game = _argon_game()
        game.day = 2
        player = game.players[0]
        player.energy = 6
        player.hand = [
            make_card("read-the-radar"),
            make_card("quacks"),
            make_card("poker"),
        ]
        game.suitcase = [
            _dummy_card(10 + i, f"Suitcase {i}", 2, 2) for i in range(4)
        ]
        outdoors_card = _dummy_card(
            201, "Outdoors Trap", 2, 2, frozenset({"Outdoors"})
        )
        safe_one = _dummy_card(202, "Safe Social", 2, 2, frozenset({"Social"}))
        safe_two = _dummy_card(203, "Safe Food", 1, 1, frozenset({"Food"}))
        game.trunk = [
            _dummy_card(100 + i, f"Trunk {i}", 2, 2) for i in range(10)
        ] + [safe_one, outdoors_card, safe_two]

        # Playing Read the Radar (index 0) puts all 4 non-Outdoors cards above Outdoors Trap
        game.play_card(0, 0)
        for offset in range(1, 5):
            self.assertNotIn("Outdoors", game.trunk[-offset].tags)
        self.assertIn("Outdoors", game.trunk[-5].tags)

        # Playing Quacks (now index 0) reveals all 4 safe cards and stops before Outdoors Trap
        game.play_card(0, 0)
        quacks_card = player.played_today[-1]
        self.assertEqual(quacks_card.title, "Quacks")
        self.assertFalse(quacks_card.markers.get("_quacks_outdoors", False))
        self.assertEqual(quacks_card.markers.get("energy_cubes"), 4)
        self.assertEqual(game.trunk[-1].title, "Outdoors Trap")

        # Playing Poker next names "Outdoors" (known top card's tag) and succeeds
        poker_idx = next(
            i for i, c in enumerate(player.hand) if c.title == "Poker"
        )
        game.play_card(0, poker_idx)
        poker_card = player.played_today[-1]
        self.assertTrue(poker_card.markers.get("poker_success"))

    def test_nos_book_played_after_another_card_and_kept_by_bohnanza(self):
        from thenos.ais.argonbrain import _build_profile

        self.assertEqual(_build_profile(make_card("nos-book")).conditional_fun_bonus, 3)

        game = _argon_game()
        game.day = 2
        player = game.players[0]
        player.energy = 4
        player.hand = [
            make_card("nos-book"),
            make_card("7-wonders"),
        ]
        game.suitcase = [
            _dummy_card(10 + i, f"Suitcase {i}", 2, 2) for i in range(4)
        ]
        game.trunk = [
            _dummy_card(100 + i, f"Trunk {i}", 2, 2) for i in range(16)
        ]

        # Plays 7 Wonders first (index 1), then Nos Book (index 0)
        first = game.ais[0].choose_card_to_play(game, 0, (0, 1))
        self.assertEqual(first, 1)
        game.play_card(0, first)
        second = game.ais[0].choose_card_to_play(game, 0, (0,))
        self.assertEqual(second, 0)
        game.play_card(0, second)
        self.assertEqual(game.card_fun(0, player.played_today[0]), 6)

        # Bohnanza should not discard Nos Book while its target is in played_today
        bohnanza = make_card("bohnanza")
        player.played_today.append(bohnanza)
        discards = game.ais[0].choose_cards_to_discard(
            game, 0, player.played_today[:-1]
        )
        self.assertNotIn(1, discards)

    def test_serial_and_parallel_simulation_parity(self):
        competitors = tuple(Competitor("ArgonBrain", ArgonBrain) for _ in range(4))
        serial = simulate_games(
            4,
            seed=20260820,
            competitors=competitors,
            rotate_seats=True,
            workers=1,
        )
        parallel = simulate_games(
            4,
            seed=20260820,
            competitors=competitors,
            rotate_seats=True,
            workers=2,
        )
        self.assertEqual(serial.rows(), parallel.rows())
        self.assertEqual(serial.ais["ArgonBrain"], parallel.ais["ArgonBrain"])


if __name__ == "__main__":
    unittest.main()
