import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CouchTubeTests(unittest.TestCase):
    def test_printed_values_and_cost_bonus_before_and_after(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("biography"),
                make_card("couch-tube"),
                make_card("waterski"),
            ]
        )

        before = game.play_card(0, 0)
        couch_tube = game.play_card(0, 0)
        after = game.play_card(0, 0)

        self.assertEqual(couch_tube.title, "Couch Tube")
        self.assertEqual(couch_tube.definition.cost, 4)
        self.assertEqual(couch_tube.definition.base_fun, 0)
        self.assertEqual(
            couch_tube.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )
        self.assertEqual(game.card_fun(0, before), 2)
        self.assertEqual(game.card_fun(0, couch_tube), 6)
        self.assertEqual(game.card_fun(0, after), 6)

    def test_cost_bonus_belongs_to_couch_tube_before_other_modifiers(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("couch-tube"),
                make_card("bring-a-friend"),
                make_card("cheap-white"),
            ]
        )

        couch_tube = game.play_card(0, 0)
        bring_a_friend = game.play_card(0, 0)
        other = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, couch_tube), 3)
        self.assertEqual(game.card_fun(0, bring_a_friend), 0)
        self.assertEqual(game.card_fun(0, other), 6)

    def test_copied_card_uses_its_effective_written_cost(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 20
        player.hand.extend(
            [make_card("couch-tube"), make_card("wedding-anniversary")]
        )
        target = make_card("adventure-race")
        game.players[1].played_today.append(target)

        couch_tube = game.play_card(0, 0)
        copied_card = game.play_card(0, 0)

        self.assertEqual(copied_card.definition.cost, 0)
        self.assertEqual(copied_card.effective_cost, 12)
        self.assertEqual(game.card_fun(0, couch_tube), 12)
        self.assertEqual(game.card_fun(0, copied_card), 20)


if __name__ == "__main__":
    unittest.main()
