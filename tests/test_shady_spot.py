import unittest

from thenos.ais.hydrogen import _hydrogen_copy_game
from thenos.cards import make_card
from tests.helpers import empty_game


class ShadySpotTests(unittest.TestCase):
    def test_next_outdoors_card_costs_half_rounded_down(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("shady-spot"), make_card("waterski"), make_card("dock-fishing")]
        )

        card = game.play_card(0, 0)
        outdoors = game.play_card(0, 0)
        later_outdoors = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Outdoors"}))
        self.assertEqual(game.energy_cost(0, later_outdoors), 1)
        self.assertEqual(player.energy, 2)
        self.assertEqual(outdoors.definition.cost, 5)

    def test_per_step_zero_clamp_for_shady_spot_and_double_cost_matches_sims(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 20
        game.suitcase = [make_card("nap") for _ in range(4)]
        game.trunk = [make_card("nap") for _ in range(4)]
        player.hand.extend(
            [
                make_card("sunscreen"),
                make_card("rouladen"),
                make_card("shady-spot"),
                make_card("beaver-burgers"),
                make_card("medical-advice"),
            ]
        )
        for _ in range(5):
            game.play_card(0, 0)

        chalk = make_card("chalk-art")
        player.hand = [chalk]
        self.assertEqual(game.energy_cost(0, chalk), 2)

        for sim_copy in (_hydrogen_copy_game(game),):
            self.assertEqual(sim_copy.energy_cost(0, sim_copy.players[0].hand[0]), 2)
            if hasattr(sim_copy, "cards_energy_costs"):
                self.assertEqual(
                    sim_copy.cards_energy_costs(0, sim_copy.players[0].hand), [2]
                )
            if hasattr(sim_copy, "hand_playable_with_costs"):
                self.assertEqual(sim_copy.hand_playable_with_costs(0), [(0, 2)])

    def test_treat_cereal_clamps_before_bring_a_friend_and_hold_the_baby(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("treat-cereal"),
                make_card("bring-a-friend"),
                make_card("hold-the-baby"),
            ]
        )
        for _ in range(3):
            game.play_card(0, 0)

        stretch = make_card("stretch")
        player.hand = [stretch]
        self.assertEqual(game.energy_cost(0, stretch), 1)

        for sim_copy in (_hydrogen_copy_game(game),):
            self.assertEqual(sim_copy.energy_cost(0, sim_copy.players[0].hand[0]), 1)


if __name__ == "__main__":
    unittest.main()
