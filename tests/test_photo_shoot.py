import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstSuitcaseAI(RandomAI):
    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class PhotoShootTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("photo-shoot")

        self.assertEqual(card.title, "Photo Shoot")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, -1)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Event", "Outdoors"}),
        )

    def test_tomorrow_picks_after_each_play(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseAI(game.rng)
        player = game.players[0]
        player.energy = 7
        photo_shoot = make_card("photo-shoot")
        player.hand.append(photo_shoot)

        game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, photo_shoot), -1)
        self.assertEqual(sum(player.picked_cards.values()), 0)

        game.end_day()
        self.assertEqual(player.tomorrow_cards, [photo_shoot])
        self.assertTrue(photo_shoot.is_tomorrow)

        picked = make_card("biography")
        replacement = make_card("waterski")
        second_replacement = make_card("fajitas")
        game.suitcase = [
            picked,
            make_card("fajitas"),
            make_card("nap"),
            make_card("solo"),
        ]
        game.trunk = [second_replacement, replacement]
        game.start_day()
        player.hand.append(make_card("biography"))

        played = game.play_card(0, 0)

        self.assertIs(played, player.played_today[0])
        self.assertIn(picked, player.hand)
        self.assertEqual(sum(player.picked_cards.values()), 1)
        self.assertEqual(player.energy, 6)
        self.assertIs(game.suitcase[0], replacement)

        game.play_card(0, 0)

        self.assertEqual(sum(player.picked_cards.values()), 2)
        self.assertEqual(player.energy, 5)
        self.assertIs(game.suitcase[0], second_replacement)

        game.end_day()
        self.assertEqual(player.tomorrow_cards, [])
        self.assertFalse(photo_shoot.is_tomorrow)


if __name__ == "__main__":
    unittest.main()
