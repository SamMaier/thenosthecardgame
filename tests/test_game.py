import copy
import unittest
from collections import Counter

from thenos.ais import RandomAI
from thenos.cards import CARD_REGISTRY, create_default_deck, make_card
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.game import (
    DAILY_PICKS,
    DAYS_PER_GAME,
    PLAYER_COUNT,
    STARTING_HAND_SIZE,
    Game,
    fractional_wins,
)


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
        counts = Counter(card.title for card in create_default_deck(daily_conditions=True))
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

    def test_setup_deals_three_starting_cards_to_each_player(self) -> None:
        game = Game.default(seed=12345)

        game.setup()

        self.assertEqual(STARTING_HAND_SIZE, 3)
        self.assertEqual(
            [len(player.hand) for player in game.players],
            [3] * PLAYER_COUNT,
        )

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

    def test_first_seen_tracks_first_player_offered_each_suitcase_card(self) -> None:
        class ChooseIndexAI(RandomAI):
            def __init__(self, index: int) -> None:
                super().__init__()
                self.index = index

            def choose_suitcase_card(self, game, player_index, suitcase):
                return self.index

        game = Game.default(seed=23)
        game.setup()
        c0 = make_card("azul")
        c1 = make_card("agricola")
        c2 = make_card("splendor")
        c3 = make_card("wingspan")
        sleep_in = make_card("sleep-in")
        c4 = make_card("carcassonne")
        c5 = make_card("terra-mystica")
        game.suitcase = [c0, c1, c2, c3]
        game.trunk.extend([c5, c4, sleep_in])

        # Player 0 picks c0; Sleep In flips into slot 0 for Player 1.
        game.ais[0] = ChooseIndexAI(0)
        game.pick_from_suitcase(0)
        self.assertEqual(
            game.stats.first_seen_by_player[0],
            Counter({c0.title: 1, c1.title: 1, c2.title: 1, c3.title: 1}),
        )

        # Player 1 sees Sleep In first, passes on it (picks c1), c4 flips into slot 1.
        game.ais[1] = ChooseIndexAI(1)
        game.pick_from_suitcase(1)
        self.assertEqual(
            game.stats.first_seen_by_player[1],
            Counter({"Sleep In": 1}),
        )

        # Player 2 picks Sleep In, but only newly flipped c4 is first-seen by Player 2.
        game.ais[2] = ChooseIndexAI(0)
        picked = game.pick_from_suitcase(2)
        self.assertIs(picked, sleep_in)
        self.assertEqual(
            game.stats.first_seen_by_player[2],
            Counter({c4.title: 1}),
        )

        # Direct card-effect Suitcase picks do not count as first-seen offers,
        # while their Trunk replacement is tracked for the next free picker.
        replacement = make_card("fajitas")
        game.trunk.append(replacement)
        game.pick_suitcase_cards(3, (game.suitcase[1],))
        self.assertEqual(game.stats.first_seen_by_player[3], Counter())

        clone = game.copy_for_simulation()
        clone.ais[3] = ChooseIndexAI(0)
        clone.pick_from_suitcase(3)
        self.assertEqual(game.stats.first_seen_by_player[3], Counter())

        game.ais[3] = ChooseIndexAI(0)
        game.pick_from_suitcase(3)
        self.assertEqual(
            game.stats.first_seen_by_player[3],
            Counter({c5.title: 1, replacement.title: 1}),
        )

    def test_first_seen_ignores_recycled_and_hand_cycled_cards_but_tracks_new_flips(
        self,
    ) -> None:
        class SwapFirstAI(RandomAI):
            def choose_cards_to_discard(self, game, player_index, cards):
                return (0,)

            def choose_card_to_discard(self, game, player_index, cards):
                return 0

            def choose_suitcase_card(self, game, player_index, suitcase):
                return 0

        game = Game.default(seed=31)
        game.setup()
        c0 = make_card("azul")
        c1 = make_card("agricola")
        c2 = make_card("splendor")
        c3 = make_card("wingspan")
        unseen_flip = make_card("sleep-in")
        hand_dump = make_card("booby-prize")
        shirt = make_card("last-years-shirt")
        game.suitcase = [c0, c1, c2, c3]
        game.trunk.append(unseen_flip)

        # Player 0 sees [c0..c3], picks c0 -> unseen_flip enters slot 0.
        game.ais[0] = SwapFirstAI()
        game.pick_from_suitcase(0)

        # Player 0 plays Last Year's Shirt, choosing over original_suitcase (marking
        # unseen_flip as already evaluated) and placing hand_dump into slot 1.
        class PickSecondAI(SwapFirstAI):
            def choose_suitcase_card(self, game, player_index, suitcase):
                return 1

        game.ais[0] = PickSecondAI()
        game.players[0].energy = 10
        game.players[0].hand = [hand_dump, shirt]
        game.play_card(0, 1)
        self.assertIs(game.suitcase[0], unseen_flip)
        self.assertIs(game.suitcase[1], hand_dump)

        # Neither the passed Last Year's Shirt card nor hand_dump is newly first-seen.
        after_pick = make_card("carcassonne")
        game.trunk.append(after_pick)
        game.ais[1] = SwapFirstAI()
        game.pick_from_suitcase(1)
        self.assertEqual(game.stats.first_seen_by_player[1], Counter())

        # Tell a Story puts a hand card on top of the Trunk; flipping it into the
        # Suitcase does not count as a fresh first-seen Trunk card.
        story_dump = make_card("wurstchen")
        tell_story = make_card("tell-a-story")
        game.players[0].hand = [story_dump, tell_story]
        game.play_card(0, 1)
        game.ais[2] = SwapFirstAI()
        game.pick_from_suitcase(2)
        self.assertEqual(
            game.stats.first_seen_by_player[2],
            Counter({after_pick.title: 1}),
        )

        # Unpack draws 3 fresh cards plus 1 already-seen recycled card; only the
        # 3 genuinely unseen cards are recorded, and a card discarded unseen by
        # unpack still counts if reshuffled into the Suitcase later.
        never_seen_before_unpack = make_card("terra-mystica")
        game.suitcase[3] = never_seen_before_unpack
        u0 = make_card("steak")
        u1 = make_card("canoe")
        u2 = make_card("kayak")
        game.trunk = [c0, u2, u1, u0]
        game.discard = []
        game.unpack(0)
        game.trunk = [never_seen_before_unpack]
        game.ais[3] = SwapFirstAI()
        game.pick_from_suitcase(3)
        self.assertEqual(
            game.stats.first_seen_by_player[3],
            Counter({u0.title: 1, u1.title: 1, u2.title: 1}),
        )
        game.ais[1] = SwapFirstAI()
        game.pick_from_suitcase(1)
        self.assertEqual(
            game.stats.first_seen_by_player[1],
            Counter({never_seen_before_unpack.title: 1}),
        )

    def test_first_seen_handles_multi_picks_and_next_day_starting_player(self) -> None:
        class BedFirstAI(RandomAI):
            def choose_suitcase_card(self, game, player_index, suitcase):
                return 0

            def choose_to_go_to_bed(
                self, game, player_index, playable_hand_indices
            ):
                return True

        game = Game.default(seed=41)
        game.setup()
        for idx in range(PLAYER_COUNT):
            game.ais[idx] = BedFirstAI()

        # Make Plans picks 2 Suitcase cards sequentially, so Player 1 first-sees
        # both the initial unseen Suitcase card and the first pick's Trunk refill.
        s0 = make_card("azul")
        s1 = make_card("agricola")
        s2 = make_card("splendor")
        s3 = make_card("wingspan")
        refill_1 = make_card("sleep-in")
        refill_2 = make_card("fajitas")
        make_plans = make_card("make-plans")
        game.suitcase = [s0, s1, s2, s3]
        game.stats.seen_suitcase_ids.update(c.instance_id for c in (s1, s2, s3))
        game.trunk.extend([refill_2, refill_1])
        game.players[1].energy = 10
        game.players[1].hand = [make_plans]
        game.play_card(1, 0)
        self.assertEqual(
            game.stats.first_seen_by_player[1],
            Counter({s0.title: 1, "Sleep In": 1}),
        )

        # refill_2 remains unseen at end of Day 1; if Player 2 goes to bed first,
        # Player 2 becomes Day 2's starting player and first-sees refill_2, while the
        # final replacement on Day 6 stays unattributed when the game ends.
        game.starting_player = 2
        game.playing_phase()
        self.assertEqual(game.starting_player, 2)
        game.end_day()
        game.day = DAYS_PER_GAME - 1
        final_unseen = make_card("booby-prize")
        filler_cards = [make_card("steak") for _ in range(11)]
        game.trunk.extend([final_unseen, *filler_cards])
        game.start_day()
        game.draw_phase()
        self.assertEqual(game.stats.first_seen_by_player[2][refill_2.title], 1)
        for idx in range(PLAYER_COUNT):
            self.assertEqual(
                game.stats.first_seen_by_player[idx][final_unseen.title], 0
            )

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
