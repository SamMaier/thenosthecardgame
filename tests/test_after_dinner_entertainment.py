import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class ExtraPickAI(RandomAI):
    def choose_extra_suitcase_pick(self, game, player_index, suitcase):
        return True


class AfterDinnerEntertainmentTests(unittest.TestCase):
    def test_next_social_is_cheaper(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("after-dinner-entertainment"),
                make_card("johnny-appleseed"),
                make_card("johnny-appleseed"),
            ]
        )

        card = game.play_card(0, 0)
        next_social = game.play_card(0, 0)
        later_social = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.energy_cost(0, later_social), 1)
        self.assertEqual(player.energy, 4)
        self.assertEqual(next_social.definition.cost, 1)

    def test_tomorrow_can_pay_for_two_picks_in_each_selection(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        player.hand.append(make_card("after-dinner-entertainment"))

        game.play_card(0, 0)
        game.end_day()
        self.assertEqual(len(player.tomorrow_cards), 1)

        player.energy = 7
        game.ais[0] = ExtraPickAI(game.rng)
        game.suitcase = [make_card("biography") for _ in range(4)]
        game.trunk = [make_card("biography") for _ in range(15)]
        game.draw_phase()

        self.assertEqual(player.energy, 4)
        self.assertEqual(sum(player.picked_cards.values()), 6)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 15)


if __name__ == "__main__":
    unittest.main()
