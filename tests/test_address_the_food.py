import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstSuitcaseTargetAI(RandomAI):
    def choose_card_target(self, game, player_index, eligible_cards):
        return 0


class AddressTheFoodTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("address-the-food")

        self.assertEqual(card.title, "Address the Food")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))

    def test_picks_food_refills_and_plays_it_for_free(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseTargetAI(game.rng)
        player = game.players[0]
        player.energy = 7
        address = make_card("address-the-food")
        player.hand.append(address)

        food = make_card("fajitas")
        refill = make_card("biography")
        game.suitcase = [make_card("biography"), food, make_card("nap"), make_card("solo")]
        game.trunk = [refill]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 9)
        self.assertEqual(player.played_today, [address, food])
        self.assertEqual(player.hand, [])
        self.assertIs(game.suitcase[1], refill)
        self.assertEqual(player.picked_cards[food.title], 1)
        self.assertEqual(player.acquired_cards[food.title], 1)
        self.assertEqual(game.stats.card_plays[food.title], 1)

    def test_plays_food_for_free_even_when_its_cost_is_unaffordable(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseTargetAI(game.rng)
        player = game.players[0]
        player.energy = 2
        address = make_card("address-the-food")
        player.hand.append(address)

        food = make_card("fajitas")
        refill = make_card("thriller-book")
        game.suitcase = [make_card("biography"), food, make_card("nap"), make_card("solo")]
        game.trunk = [refill]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(player.played_today, [address, food])
        self.assertEqual(player.hand, [])
        self.assertEqual(game.stats.card_plays[food.title], 1)
        self.assertEqual(player.picked_cards[food.title], 1)
        self.assertIs(game.suitcase[1], refill)

    def test_keeps_food_in_hand_when_the_target_is_illegal(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseTargetAI(game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("address-the-food"))

        restricted_food = make_card("morning-coffee")
        refill = make_card("thriller-book")
        game.suitcase = [restricted_food, make_card("biography"), make_card("nap"), make_card("solo")]
        game.trunk = [refill]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 5)
        self.assertEqual(player.played_today[0].title, "Address the Food")
        self.assertEqual(player.hand, [restricted_food])
        self.assertEqual(game.stats.card_plays[restricted_food.title], 0)
        self.assertIs(game.suitcase[0], refill)

    def test_does_nothing_when_no_food_is_visible(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("address-the-food"))
        game.suitcase = [make_card("biography") for _ in range(4)]

        game.play_card(0, 0)

        self.assertEqual(player.hand, [])
        self.assertEqual(player.energy, 5)
        self.assertEqual(sum(player.picked_cards.values()), 0)


if __name__ == "__main__":
    unittest.main()
