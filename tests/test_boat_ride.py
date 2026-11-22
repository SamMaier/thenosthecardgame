import unittest

from thenos.ais.hydrogen import _hydrogen_copy_game
from thenos.cards import make_card
from tests.helpers import empty_game


class BoatRideTests(unittest.TestCase):
    def test_next_item_costs_zero_and_later_items_cost_normally(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("boat-ride"), make_card("biography"), make_card("nos-shirt"), make_card("nos-shirt")]
        )

        card = game.play_card(0, 0)
        non_item = game.play_card(0, 0)
        first_item = game.play_card(0, 0)
        later_item = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Outdoors"}))
        self.assertEqual(non_item.definition.tags, frozenset({"Relax"}))
        self.assertEqual(first_item.definition.cost, 3)
        self.assertEqual(later_item.definition.cost, 3)
        self.assertEqual(player.energy, 1)

    def test_next_item_stays_free_even_after_later_cost_increases_and_matches_sims(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 15
        player.hand.extend(
            [
                make_card("boat-ride"),
                make_card("beaver-burgers"),
                make_card("bring-a-friend"),
                make_card("hold-the-baby"),
            ]
        )
        for _ in range(4):
            game.play_card(0, 0)

        item = make_card("nos-shirt")
        non_item = make_card("waterski")
        player.hand = [item, non_item]

        self.assertEqual(game.energy_cost(0, item), 0)
        self.assertEqual(game.energy_cost(0, non_item), 13)

        for sim_copy in (_hydrogen_copy_game(game),):
            self.assertEqual(sim_copy.energy_cost(0, sim_copy.players[0].hand[0]), 0)
            self.assertEqual(sim_copy.energy_cost(0, sim_copy.players[0].hand[1]), 13)
            if hasattr(sim_copy, "cards_energy_costs"):
                self.assertEqual(
                    sim_copy.cards_energy_costs(0, sim_copy.players[0].hand),
                    [0, 13],
                )
            if hasattr(sim_copy, "hand_playable_with_costs"):
                self.assertEqual(
                    sim_copy.hand_playable_with_costs(0), [(0, 0)]
                )

    def test_copied_boat_ride_keeps_next_item_free_and_matches_sims(self) -> None:
        game = empty_game()
        p0 = game.players[0]
        p0.energy = 15
        game.players[1].played_today.append(make_card("boat-ride"))
        p0.hand.extend(
            [
                make_card("wedding-anniversary"),
                make_card("beaver-burgers"),
                make_card("bring-a-friend"),
            ]
        )
        for _ in range(3):
            game.play_card(0, 0)

        item = make_card("nos-shirt")
        p0.hand = [item]
        self.assertEqual(game.energy_cost(0, item), 0)

        for sim_copy in (_hydrogen_copy_game(game),):
            self.assertEqual(sim_copy.energy_cost(0, sim_copy.players[0].hand[0]), 0)
            if hasattr(sim_copy, "cards_energy_costs"):
                self.assertEqual(
                    sim_copy.cards_energy_costs(0, sim_copy.players[0].hand), [0]
                )
            if hasattr(sim_copy, "hand_playable_with_costs"):
                self.assertEqual(
                    sim_copy.hand_playable_with_costs(0), [(0, 0)]
                )


if __name__ == "__main__":
    unittest.main()
