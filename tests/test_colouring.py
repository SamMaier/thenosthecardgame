import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ColouringTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("colouring")

        self.assertEqual(card.title, "Colouring")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_cards_played_after_count_as_having_all_tags(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        played_before = make_card("biography")
        colouring = make_card("colouring")
        played_after = make_card("biography")
        player.hand.extend([played_before, colouring, played_after])

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(played_before.tags, frozenset({"Relax"}))
        self.assertEqual(colouring.tags, frozenset({"Relax"}))
        self.assertIn("Food", played_after.tags)
        self.assertIn("Item", played_after.tags)
        self.assertIn("Indoors", played_after.tags)

    def test_all_tags_are_cleared_when_the_card_is_discarded(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        colouring = make_card("colouring")
        played_after = make_card("biography")
        player.hand.extend([colouring, played_after])

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.end_day()

        self.assertNotIn(colouring, player.tomorrow_cards)
        self.assertIn(colouring, game.discard)
        self.assertEqual(played_after.tags, frozenset({"Relax"}))

    def test_hand_cost_modifiers_and_bohnanza_or_copy_interactions(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        sunscreen = make_card("sunscreen")
        colouring = make_card("colouring")
        fajitas = make_card("fajitas")
        player.hand.extend([sunscreen, colouring, fajitas])

        game.play_card(0, 0)
        self.assertEqual(game.energy_cost(0, fajitas), 3)

        game.play_card(0, 0)
        # Fajitas (Food) in hand now counts as Outdoors after Colouring, so Sunscreen reduces its cost by 1.
        self.assertEqual(game.energy_cost(0, fajitas), 2)
        self.assertNotIn("_all_tags", fajitas.markers)
        game.play_card(0, 0)
        self.assertIn("Outdoors", fajitas.tags)

        # If Colouring is removed from played_today (e.g. Bohnanza), Fajitas loses all-tags.
        player.played_today.remove(colouring)
        game.discard_card(colouring)
        game._sync_played_today_tags(player)
        self.assertEqual(fajitas.tags, frozenset({"Food"}))


if __name__ == "__main__":
    unittest.main()
