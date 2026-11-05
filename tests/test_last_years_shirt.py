import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class ShirtChoiceAI(RandomAI):
    def __init__(self, discard_indices, suitcase_indices, rng) -> None:
        super().__init__(rng)
        self.discard_indices = tuple(discard_indices)
        self.suitcase_indices = list(suitcase_indices)

    def choose_cards_to_discard(self, game, player_index, cards):
        return self.discard_indices

    def choose_suitcase_card(self, game, player_index, suitcase):
        return self.suitcase_indices.pop(0)


class LastYearsShirtTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("last-years-shirt")

        self.assertEqual(card.title, "Last Year's Shirt")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

    def test_picks_multiple_suitcase_cards_and_replenishes_slots_from_hand(self) -> None:
        game = empty_game()
        game.ais[0] = ShirtChoiceAI((0, 2), [1, 1], game.rng)
        player = game.players[0]
        player.energy = 7
        shirt = make_card("last-years-shirt")
        give_one = make_card("fajitas")
        keep_card = make_card("nap")
        give_two = make_card("waterski")
        player.hand.extend([shirt, give_one, keep_card, give_two])

        slot0 = make_card("solo")
        slot1 = make_card("wingspan")
        slot2 = make_card("azul")
        slot3 = make_card("splendor")
        game.suitcase = [slot0, slot1, slot2, slot3]
        trunk_card = make_card("risk")
        game.trunk = [trunk_card]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(player.hand, [keep_card, slot1, slot2])
        self.assertEqual(game.suitcase, [slot0, give_one, give_two, slot3])
        self.assertEqual(game.trunk, [trunk_card])
        self.assertEqual(player.picked_cards["Wingspan"], 1)
        self.assertEqual(player.picked_cards["Azul"], 1)
        self.assertEqual(game.stats.suitcase_picks["Wingspan"], 1)
        self.assertEqual(game.stats.suitcase_picks["Azul"], 1)

    def test_choosing_zero_cards_leaves_suitcase_and_hand_unchanged(self) -> None:
        game = empty_game()
        game.ais[0] = ShirtChoiceAI((), [], game.rng)
        player = game.players[0]
        player.energy = 7
        shirt = make_card("last-years-shirt")
        keep_card = make_card("fajitas")
        player.hand.extend([shirt, keep_card])
        suitcase_cards = [
            make_card("solo"),
            make_card("wingspan"),
            make_card("azul"),
            make_card("splendor"),
        ]
        game.suitcase = list(suitcase_cards)

        game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(player.hand, [keep_card])
        self.assertEqual(game.suitcase, suitcase_cards)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 0)

    def test_empty_hand_does_nothing(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("last-years-shirt"))
        suitcase_cards = [
            make_card("solo"),
            make_card("wingspan"),
            make_card("azul"),
            make_card("splendor"),
        ]
        game.suitcase = list(suitcase_cards)

        game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(player.hand, [])
        self.assertEqual(game.suitcase, suitcase_cards)

    def test_galaxybrain_evaluates_remaining_suitcase_cards_across_multiple_picks(self) -> None:
        from thenos.ais import GalaxybrainAI

        game = empty_game()
        game.ais[0] = GalaxybrainAI(game.rng)
        player = game.players[0]
        player.energy = 15
        shirt = make_card("last-years-shirt")
        low_one = make_card("m-ms")
        low_two = make_card("bubly")
        player.hand.extend([shirt, low_one, low_two])

        slot0 = make_card("waterski")
        slot1 = make_card("m-ms")
        slot2 = make_card("bubly")
        slot3 = make_card("adventure-race")
        game.suitcase = [slot0, slot1, slot2, slot3]
        game.trunk = [make_card("non-fiction-book")]

        game.play_card(0, 0)

        self.assertIn(slot0, player.hand)
        self.assertIn(slot3, player.hand)
        self.assertEqual(game.suitcase[1], slot1)
        self.assertEqual(game.suitcase[2], slot2)


if __name__ == "__main__":
    unittest.main()
