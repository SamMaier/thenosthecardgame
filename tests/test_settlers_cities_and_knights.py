import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from thenos.game import DAYS_PER_GAME
from tests.helpers import empty_game


class FirstPickAI(RandomAI):
    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class SettlersCitiesAndKnightsTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("cities-and-knights")

        self.assertEqual(card.title, "Cities and Knights")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 3)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))

    def test_end_day_picks_two_when_3_of_one_tag_and_2_of_another(
        self,
    ) -> None:
        game = empty_game()
        game.day = 2
        game.ais[0] = FirstPickAI(game.rng)
        player = game.players[0]
        player.energy = 15
        # Colouring grants Relax to cards played after it:
        # - cities-and-knights (Board Game)
        # - fit-to-print (Board Game, Indoors)
        # - colouring (Relax)
        # - tres-fute (Board Game + granted Relax -> counts as both!)
        player.hand.extend(
            [
                make_card("cities-and-knights"),
                make_card("fit-to-print"),
                make_card("colouring"),
                make_card("tres-fute"),
            ]
        )
        for _ in range(4):
            game.play_card(0, 0)

        pick1 = make_card("biography")
        pick2 = make_card("waterski")
        rep1 = make_card("chalk-art")
        rep2 = make_card("nap")
        game.suitcase = [
            pick1,
            make_card("fajitas"),
            make_card("solo"),
            make_card("azul"),
        ]
        game.trunk = [rep2, rep1, pick2]

        game.end_day()

        self.assertEqual(player.hand, [pick1, pick2])
        self.assertEqual(game.suitcase[0], rep1)
        self.assertEqual(player.picked_cards[pick1.title], 1)
        self.assertEqual(player.picked_cards["Waterski"], 1)
        self.assertEqual(game.stats.card_acquisitions[pick1.title], 1)

    def test_does_not_trigger_with_3_and_1_or_2_and_2(self) -> None:
        for slugs in (
            ("cities-and-knights", "tres-fute", "splendor", "biography"),
            ("cities-and-knights", "tres-fute", "biography", "nap"),
        ):
            with self.subTest(slugs=slugs):
                game = empty_game()
                game.day = 2
                player = game.players[0]
                player.energy = 15
                player.hand.extend(make_card(s) for s in slugs)
                for _ in slugs:
                    game.play_card(0, 0)
                suitcase = [make_card("fajitas") for _ in range(4)]
                game.suitcase = suitcase.copy()

                game.end_day()

                self.assertEqual(player.hand, [])
                self.assertEqual(game.suitcase, suitcase)

    def test_day_six_end_day_picks_do_not_track_acquisitions(self) -> None:
        game = empty_game()
        game.day = DAYS_PER_GAME
        game.ais[0] = FirstPickAI(game.rng)
        player = game.players[0]
        player.energy = 15
        player.hand.extend(
            [
                make_card("cities-and-knights"),
                make_card("tres-fute"),
                make_card("splendor"),
                make_card("biography"),
                make_card("nap"),
            ]
        )
        for _ in range(5):
            game.play_card(0, 0)

        p1 = make_card("doritos")
        p2 = make_card("fajitas")
        game.suitcase = [p1, make_card("solo"), make_card("azul"), make_card("risk")]
        game.trunk = [make_card("chalk-art"), p2]

        game.end_day()

        self.assertEqual(player.hand, [p1, p2])
        self.assertEqual(player.acquired_cards["Doritos"], 0)
        self.assertEqual(game.stats.card_acquisitions["Doritos"], 0)


if __name__ == "__main__":
    unittest.main()
