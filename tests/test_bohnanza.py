import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class BohnanzaAI(RandomAI):
    def __init__(self, discard_indices, rng) -> None:
        super().__init__(rng)
        self.discard_indices = tuple(discard_indices)

    def choose_cards_to_discard(self, game, player_index, cards):
        return self.discard_indices


class BohnanzaTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("bohnanza")

        self.assertEqual(card.title, "Bohnanza")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))

    def test_trashes_selected_earlier_played_cards_for_three_fun_each(self) -> None:
        game = empty_game()
        game.ais[0] = BohnanzaAI((0, 2), game.rng)
        player = game.players[0]
        player.energy = 10
        first_card = make_card("fajitas")
        kept_card = make_card("biography")
        third_card = make_card("evening-chat")
        bohnanza = make_card("bohnanza")
        player.hand.extend([first_card, kept_card, third_card, bohnanza])

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)
        player.asleep = False
        played_bohnanza = game.play_card(0, 0)

        self.assertEqual(player.played_today, [kept_card, played_bohnanza])
        self.assertIn(first_card, game.discard)
        self.assertIn(third_card, game.discard)
        self.assertEqual(game.card_fun(0, played_bohnanza), 6)

        game.end_day()

        self.assertEqual(player.fun, 8)
        self.assertEqual(player.tomorrow_cards, [])

    def test_trashing_zero_cards_scores_zero_fun(self) -> None:
        game = empty_game()
        game.ais[0] = BohnanzaAI((), game.rng)
        player = game.players[0]
        player.energy = 7
        earlier = make_card("biography")
        bohnanza = make_card("bohnanza")
        player.hand.extend([earlier, bohnanza])

        game.play_card(0, 0)
        played_bohnanza = game.play_card(0, 0)

        self.assertEqual(player.played_today, [earlier, played_bohnanza])
        self.assertEqual(game.discard, [])
        self.assertEqual(game.card_fun(0, played_bohnanza), 0)

    def test_galaxybrain_trashes_low_fun_played_cards_and_keeps_high_fun_cards_on_day_six(self) -> None:
        from thenos.ais import GalaxybrainAI

        game = empty_game()
        game.day = 6
        game.ais[0] = GalaxybrainAI(game.rng)
        player = game.players[0]
        player.energy = 10
        zero_fun = make_card("fajitas")
        high_fun = make_card("waterski")
        player.played_today.extend([zero_fun, high_fun])
        player.hand.append(make_card("bohnanza"))

        played_bohnanza = game.play_card(0, 0)

        self.assertEqual(player.played_today, [high_fun, played_bohnanza])
        self.assertIn(zero_fun, game.discard)
        self.assertEqual(game.card_fun(0, played_bohnanza), 3)


if __name__ == "__main__":
    unittest.main()
