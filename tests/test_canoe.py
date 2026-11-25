import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CanoeTests(unittest.TestCase):
    def test_next_item_gets_its_written_energy_cost_as_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 8
        player.hand.extend(
            [make_card("canoe"), make_card("cheap-white"), make_card("nos-shirt")]
        )

        card = game.play_card(0, 0)
        non_item = game.play_card(0, 0)
        item = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Exercise", "Outdoors"}))
        self.assertEqual(game.card_fun(0, card), 3)
        self.assertEqual(game.card_fun(0, non_item), 3)
        self.assertEqual(game.card_fun(0, item), 1)

    def test_written_cost_ignores_a_later_energy_modifier(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 9
        player.hand.extend(
            [
                make_card("rouladen"),
                make_card("canoe"),
                make_card("nos-shirt"),
            ]
        )

        game.play_card(0, 0)
        canoe = game.play_card(0, 0)
        item = player.hand[0]

        self.assertEqual(game.energy_cost(0, item), 0)
        game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, canoe), 3)
        self.assertEqual(game.card_fun(0, item), 1)

    def test_canonical_fast_sim_scoring_parity(self) -> None:
        from thenos.ais import OsmiumAI

        osmium = OsmiumAI()
        for first_slug, second_slug, expected in (
            ("canoe", "ponyback", 6),
            ("fancy-craft", "ponyback", 7),
        ):
            with self.subTest(pair=(first_slug, second_slug)):
                g = empty_game()
                g.trunk = [make_card("nap") for _ in range(12)]
                g.suitcase = [make_card("nap") for _ in range(4)]
                g.players[0].energy = 10
                g.players[0].hand.extend([make_card(first_slug), make_card(second_slug)])

                sim_live = osmium._prepare_observable_game(g, 0)
                self.assertTrue(sim_live._all_canonical_defs)
                g.play_card(0, 0)
                g.play_card(0, 0)
                sim_live.play_card(0, 0)
                sim_live.play_card(0, 0)

                exact = sum(g.card_fun(0, c) for c in g.players[0].played_today)
                self.assertEqual(exact, expected)
                self.assertEqual(sim_live.score_played_today_fast(0)[0], exact)

                sim_post = osmium._prepare_observable_game(g, 0)
                self.assertTrue(sim_post._all_canonical_defs)
                self.assertEqual(sim_post.score_played_today_fast(0)[0], exact)


if __name__ == "__main__":
    unittest.main()
