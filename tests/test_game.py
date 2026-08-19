import copy
import unittest
from collections import Counter

from thenos.ais import RandomAI
from thenos.cards import CARD_REGISTRY, create_default_deck
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.game import DAILY_PICKS, DAYS_PER_GAME, PLAYER_COUNT, Game, fractional_wins


class GameTests(unittest.TestCase):
    def test_deepcopy_shares_definition_but_copies_instance_state(self) -> None:
        definition = CARD_REGISTRY["wedding-anniversary"]
        original = CardInstance(
            99_200_000,
            definition,
            markers={"nested": {"value": 1}},
        )

        copied = copy.deepcopy(original)

        self.assertIsNot(copied, original)
        self.assertIs(copied.definition, definition)
        self.assertIsNot(copied.markers, original.markers)
        self.assertIsNot(copied.markers["nested"], original.markers["nested"])

    def test_player_may_go_to_bed_with_a_playable_card(self) -> None:
        class GoToBedAI(RandomAI):
            def choose_to_go_to_bed(
                self, game, player_index, playable_hand_indices
            ):
                return True

        game = Game.default(seed=2)
        game.ais[0] = GoToBedAI()
        player = game.players[0]
        playable = CardInstance(
            99_100_000,
            CardDefinition(
                slug="playable-test",
                title="Playable Test",
                tags=frozenset(),
                cost=0,
                base_fun=1,
            ),
        )
        player.hand.append(playable)

        game.playing_phase()

        self.assertTrue(player.asleep)
        self.assertEqual(player.hand, [playable])
        self.assertEqual(player.played_today, [])
        self.assertEqual(game.starting_player, 0)

    def test_active_tomorrow_card_runs_only_tomorrow_effects(self) -> None:
        events = Counter()

        class LifecycleBehavior(CardBehavior):
            has_tomorrow_action = True

            def on_start_day(self, game, player, card):
                events["tomorrow_starts"] += 1

            def on_score(self, game, player, card):
                events["ordinary_scores"] += 1

            def on_end_day(self, game, player, card):
                events["ordinary_end_days"] += 1

            def on_card_play(self, game, player, source, played_card):
                events["ordinary_card_plays"] += 1

            def on_tomorrow_card_play(
                self, game, player, source, played_card
            ):
                events["tomorrow_card_plays"] += 1

        game = Game.default(seed=1)
        player = game.players[0]
        card = CardInstance(
            99_000_000,
            CardDefinition(
                slug="lifecycle-test",
                title="Lifecycle Test",
                tags=frozenset(),
                cost=0,
                base_fun=5,
                behavior=LifecycleBehavior(),
            ),
        )
        player.energy = 7
        player.hand.append(card)

        game.play_card(0, len(player.hand) - 1)
        game.end_day()

        self.assertEqual(player.fun, 5)
        self.assertEqual(events["ordinary_scores"], 1)
        self.assertEqual(events["ordinary_end_days"], 1)
        self.assertEqual(events["ordinary_card_plays"], 1)

        game.start_day()
        self.assertEqual(events["tomorrow_starts"], 1)

        player.hand.append(
            CardInstance(
                99_000_001,
                CardDefinition(
                    slug="ordinary-test",
                    title="Ordinary Test",
                    tags=frozenset(),
                    cost=0,
                    base_fun=1,
                ),
            )
        )
        game.play_card(0, len(player.hand) - 1)

        game.end_day()

        self.assertEqual(player.fun, 6)
        self.assertEqual(events["ordinary_scores"], 1)
        self.assertEqual(events["ordinary_end_days"], 1)
        self.assertEqual(events["ordinary_card_plays"], 1)
        self.assertEqual(events["tomorrow_card_plays"], 1)

    def test_default_deck_has_one_copy_of_each_implemented_card(self) -> None:
        counts = Counter(card.title for card in create_default_deck())
        self.assertEqual(len(counts), len(CARD_REGISTRY))
        self.assertTrue(all(count == 1 for count in counts.values()))

    def test_complete_seeded_game_runs_six_days(self) -> None:
        game = Game.default(seed=12345)

        result = game.run()

        self.assertEqual(result.days_played, DAYS_PER_GAME)
        self.assertGreaterEqual(
            sum(game.stats.suitcase_picks.values()),
            PLAYER_COUNT * DAILY_PICKS * DAYS_PER_GAME,
        )
        self.assertAlmostEqual(sum(result.win_shares), 1.0)
        self.assertTrue(all(score >= 0 for score in result.scores))

    def test_tied_players_split_one_win(self) -> None:
        self.assertEqual(fractional_wins((10, 10, 4, 2)), (0.5, 0.5, 0.0, 0.0))
        self.assertEqual(fractional_wins((7, 7, 7, 7)), (0.25, 0.25, 0.25, 0.25))

    def test_seeded_games_are_reproducible(self) -> None:
        first = Game.default(seed=77).run()
        second = Game.default(seed=77).run()
        self.assertEqual(first, second)

    def test_free_pick_statistics_only_record_presented_choices(self) -> None:
        class ChooseSecondAI:
            def choose_suitcase_card(self, game, player_index, suitcase):
                return 1

        game = Game.default(seed=17)
        game.setup()
        game.ais[0] = ChooseSecondAI()
        offered = tuple(game.suitcase)

        game.pick_from_suitcase(0)

        self.assertEqual(
            game.stats.free_pick_offers,
            Counter(card.title for card in offered),
        )
        self.assertEqual(game.stats.free_picks, Counter({offered[1].title: 1}))

        free_pick_offers = game.stats.free_pick_offers.copy()
        free_picks = game.stats.free_picks.copy()
        game.pick_suitcase_cards(0, (game.suitcase[0],))
        self.assertEqual(game.stats.free_pick_offers, free_pick_offers)
        self.assertEqual(game.stats.free_picks, free_picks)

    def test_unpack_costs_fun_discards_and_refills_four_cards(self) -> None:
        game = Game.default(seed=8)
        game.setup()
        original_ids = {card.instance_id for card in game.suitcase}

        game.unpack(0)

        self.assertEqual(game.players[0].fun, -1)
        self.assertEqual(len(game.suitcase), 4)
        self.assertTrue(original_ids.isdisjoint(card.instance_id for card in game.suitcase))
        self.assertEqual(
            original_ids,
            {card.instance_id for card in game.discard},
        )


if __name__ == "__main__":
    unittest.main()
