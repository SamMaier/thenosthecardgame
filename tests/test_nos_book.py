import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class NosBookAI(RandomAI):
    def choose_card_target(self, game, player_index, eligible_cards):
        return next(
            (
                index for index, card in enumerate(eligible_cards)
                if card.title == "Keeper"
            ),
            0,
        )

    def choose_cards_for_trunk(self, game, player_index, cards):
        wanted = {"Biography", "Fajitas"}
        return [index for index, card in enumerate(cards) if card.title in wanted]

    def order_cards_for_trunk(self, game, player_index, cards):
        return tuple(reversed(range(len(cards))))


class NosBookTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("nos-book")

        self.assertEqual(card.title, "Nos Book")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

    def test_collects_cleanup_cards_keeps_one_and_stacks_a_chosen_subset(self) -> None:
        game = empty_game()
        game.ais[0] = NosBookAI(random.Random(0))
        player = game.players[0]
        player.energy = 7

        old_tomorrow = make_card("fajitas")
        old_tomorrow.is_tomorrow = True
        game.players[2].tomorrow_cards.append(old_tomorrow)
        biography = make_card("biography")
        nos_book = make_card("nos-book")
        keeper = make_card("keeper")
        player.played_today.extend([biography, nos_book])
        game.players[1].played_today.append(keeper)

        game.end_day()

        self.assertIn(keeper, player.hand)
        self.assertEqual([card.title for card in game.trunk[-2:]], ["Biography", "Fajitas"])
        self.assertEqual(game.discard, [nos_book])
        self.assertFalse(old_tomorrow.is_tomorrow)

    def test_does_not_collect_cards_that_move_to_tomorrow(self) -> None:
        game = empty_game()
        game.ais[0] = NosBookAI(random.Random(0))
        player = game.players[0]
        nos_book = make_card("nos-book")
        stay_up_late = make_card("stay-up-late")
        player.played_today.extend([nos_book, stay_up_late])

        game.end_day()

        self.assertEqual(player.tomorrow_cards, [stay_up_late])
        self.assertTrue(stay_up_late.is_tomorrow)


if __name__ == "__main__":
    unittest.main()
