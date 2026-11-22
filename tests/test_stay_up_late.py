import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class StayUpLateTests(unittest.TestCase):
    def test_immediate_energy_gain_and_tomorrow_penalty(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 0
        player.hand.append(make_card("stay-up-late"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 0)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Social", "Indoors"}))
        self.assertEqual(player.energy, 2)
        self.assertEqual(game.card_fun(0, card), 0)

        game.end_day()

        self.assertEqual(player.tomorrow_cards, [card])
        self.assertTrue(card.is_tomorrow)

        game.start_day()

        self.assertEqual(player.energy, 5)

    def test_stacked_tomorrow_energy_penalties_clamp_starting_energy_to_zero(self) -> None:
        game = empty_game()
        player = game.players[0]
        penalties = [
            make_card("stay-up-late"),
            make_card("stay-up-late"),
            make_card("stay-up-late"),
            make_card("stay-up-late"),
            make_card("evening-chat"),
        ]
        for card in penalties:
            card.is_tomorrow = True
        player.tomorrow_cards.extend(penalties)

        game.start_day()

        self.assertEqual(player.energy, 0)

    def test_mixed_tomorrow_penalties_and_gains_clamp_after_all_tomorrow_effects(self) -> None:
        game = empty_game()
        player = game.players[0]
        tomorrow = [
            make_card("stay-up-late"),
            make_card("stay-up-late"),
            make_card("stay-up-late"),
            make_card("stay-up-late"),
            make_card("morning-walk"),
        ]
        for card in tomorrow:
            card.is_tomorrow = True
        player.tomorrow_cards.extend(tomorrow)

        game.start_day()

        # 7 - 8 + 3 = 2 (does not clamp early to 0 before +3)
        self.assertEqual(player.energy, 2)


if __name__ == "__main__":
    unittest.main()
