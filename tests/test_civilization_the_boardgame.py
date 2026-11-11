import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CivilizationTheBoardgameTests(unittest.TestCase):
    def test_cost_tags_and_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.append(make_card("civilization-the-boardgame"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.slug, "civilization-the-boardgame")
        self.assertEqual(card.title, "Civilization the Boardgame")
        self.assertEqual(card.definition.cost, 10)
        self.assertEqual(card.definition.base_fun, 14)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Board Game", "Indoors"}),
        )
        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, card), 14)


if __name__ == "__main__":
    unittest.main()
