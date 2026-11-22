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
        self.assertEqual(card.definition.cost, 0)
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

        self.assertEqual(player.energy, 7)
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

        self.assertEqual(player.energy, 7)
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

        self.assertEqual(player.energy, 7)
        self.assertEqual(player.hand, [])
        self.assertEqual(game.suitcase, suitcase_cards)

    def test_picking_marked_suitcase_card_clears_suitcase_markers(self) -> None:
        game = empty_game()
        game.ais[0] = ShirtChoiceAI((0,), [0], game.rng)
        player = game.players[0]
        player.energy = 7
        give_one = make_card("fajitas")
        player.hand.extend([make_card("last-years-shirt"), give_one])
        marked = make_card("solo")
        marked.markers["energy_cube"] = True
        marked.markers["_scout_energy_cube_99"] = True
        game.suitcase = [
            marked,
            make_card("wingspan"),
            make_card("azul"),
            make_card("splendor"),
        ]

        game.play_card(0, 0)

        self.assertEqual(player.hand, [marked])
        self.assertEqual(marked.markers, {})


if __name__ == "__main__":
    unittest.main()
