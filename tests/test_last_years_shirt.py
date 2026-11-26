import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TrunkChoiceAI(RandomAI):
    def __init__(self, discard_index: int, rng) -> None:
        super().__init__(rng)
        self.discard_index = discard_index
        self.seen_options = None

    def choose_card_to_discard(self, game, player_index, cards):
        self.seen_options = tuple(cards)
        return self.discard_index


class LastYearsShirtTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("last-years-shirt")

        self.assertEqual(card.title, "Last Year's Shirt")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

    def test_draws_two_from_trunk_then_places_chosen_hand_card_on_top_of_trunk(
        self,
    ) -> None:
        game = empty_game()
        ai = TrunkChoiceAI(1, game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        shirt = make_card("last-years-shirt")
        keep_card = make_card("fajitas")
        to_trunk = make_card("waterski")
        to_trunk.is_tomorrow = True
        to_trunk.markers["energy_cube"] = True
        player.hand.extend([shirt, keep_card, to_trunk])

        bottom_trunk = make_card("risk")
        drawn_2 = make_card("azul")
        drawn_1 = make_card("wingspan")
        game.trunk = [bottom_trunk, drawn_2, drawn_1]
        suitcase_cards = [
            make_card("solo"),
            make_card("splendor"),
            make_card("nap"),
            make_card("movie"),
        ]
        game.suitcase = list(suitcase_cards)

        game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(
            ai.seen_options, (keep_card, to_trunk, drawn_1, drawn_2)
        )
        self.assertEqual(player.hand, [keep_card, drawn_1, drawn_2])
        self.assertEqual(game.suitcase, suitcase_cards)
        self.assertEqual(game.trunk, [bottom_trunk, to_trunk])
        self.assertIs(game.trunk[-1], to_trunk)
        self.assertFalse(to_trunk.is_tomorrow)
        self.assertEqual(to_trunk.markers, {})
        self.assertEqual(player.acquired_cards[drawn_1.title], 1)
        self.assertEqual(player.acquired_cards[drawn_2.title], 1)
        self.assertEqual(game.stats.card_acquisitions[drawn_1.title], 1)
        self.assertEqual(game.stats.card_acquisitions[drawn_2.title], 1)

    def test_can_place_newly_drawn_card_back_on_top_of_trunk_including_from_empty_hand(
        self,
    ) -> None:
        game = empty_game()
        game.ais[0] = TrunkChoiceAI(0, game.rng)
        player = game.players[0]
        player.energy = 7
        shirt = make_card("last-years-shirt")
        player.hand.append(shirt)
        bottom_trunk = make_card("solo")
        drawn_2 = make_card("canoe")
        drawn_1 = make_card("biography")
        game.trunk = [bottom_trunk, drawn_2, drawn_1]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(player.hand, [drawn_2])
        self.assertEqual(game.trunk, [bottom_trunk, drawn_1])
        self.assertEqual(player.acquired_cards[drawn_1.title], 1)
        self.assertEqual(player.acquired_cards[drawn_2.title], 1)
        self.assertEqual(game.stats.card_acquisitions[drawn_1.title], 1)
        self.assertEqual(game.stats.card_acquisitions[drawn_2.title], 1)

    def test_reshuffles_discard_when_trunk_has_fewer_than_two_cards(self) -> None:
        game = empty_game()
        game.ais[0] = TrunkChoiceAI(0, game.rng)
        player = game.players[0]
        player.energy = 2
        keep_card = make_card("fajitas")
        player.hand.extend([make_card("last-years-shirt"), keep_card])
        drawn_1 = make_card("canoe")
        recycled = make_card("biography")
        game.trunk = [drawn_1]
        game.discard = [recycled]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 1)
        self.assertEqual(player.hand, [drawn_1, recycled])
        self.assertEqual(game.trunk, [keep_card])
        self.assertEqual(game.discard, [])

    def test_triggers_paddleboat_twice_for_two_drawn_cards(self) -> None:
        game = empty_game()
        game.ais[0] = TrunkChoiceAI(0, game.rng)
        player = game.players[0]
        player.energy = 7
        paddleboat = make_card("paddleboat")
        shirt = make_card("last-years-shirt")
        keep_card = make_card("fajitas")
        drawn_2 = make_card("canoe")
        drawn_1 = make_card("biography")
        player.hand.extend([paddleboat, shirt, keep_card])
        game.trunk = [drawn_2, drawn_1]

        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.energy, 3)
        self.assertEqual(game.card_fun(0, paddleboat), 4)
        self.assertEqual(player.hand, [drawn_1, drawn_2])
        self.assertEqual(game.trunk, [keep_card])


if __name__ == "__main__":
    unittest.main()
