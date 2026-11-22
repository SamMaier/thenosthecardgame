import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CouchTubeTests(unittest.TestCase):
    def test_printed_values_and_no_bonus_when_played_alone(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("couch-tube"))

        couch_tube = game.play_card(0, 0)

        self.assertEqual(couch_tube.title, "Couch Tube")
        self.assertEqual(couch_tube.definition.cost, 3)
        self.assertEqual(couch_tube.definition.base_fun, 2)
        self.assertEqual(
            couch_tube.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )
        self.assertEqual(game.card_fun(0, couch_tube), 2)

    def test_bonus_applies_with_exactly_one_other_card_before_or_after(self) -> None:
        for order in (["biography", "couch-tube"], ["couch-tube", "biography"]):
            with self.subTest(order=order):
                game = empty_game()
                player = game.players[0]
                player.energy = 7
                player.hand.extend([make_card(slug) for slug in order])

                first = game.play_card(0, 0)
                second = game.play_card(0, 0)
                couch_tube = second if order[1] == "couch-tube" else first

                self.assertEqual(game.card_fun(0, couch_tube), 6)

    def test_no_bonus_with_two_other_cards_down(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        player.hand.extend(
            [
                make_card("biography"),
                make_card("couch-tube"),
                make_card("waterski"),
            ]
        )

        game.play_card(0, 0)
        couch_tube = game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, couch_tube), 2)

    def test_bonus_applies_before_other_modifiers(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend([make_card("bring-a-friend"), make_card("couch-tube")])

        game.play_card(0, 0)
        couch_tube = game.play_card(0, 0)

        # (2 base + 4 bonus) * 2 from Bring a Friend = 12.
        self.assertEqual(game.card_fun(0, couch_tube), 12)

    def test_copied_couch_tube_scores_bonus_with_one_other_card(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend([make_card("biography"), make_card("wedding-anniversary")])
        game.players[1].played_today.append(make_card("couch-tube"))

        game.play_card(0, 0)
        copied = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, copied), 6)

    def test_unplayed_couch_tube_and_tomorrow_card_do_not_trigger_bonus(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.tomorrow_cards.append(make_card("zumba"))
        couch_tube = make_card("couch-tube")
        player.hand.extend([make_card("biography"), couch_tube])

        game.play_card(0, 0)
        self.assertEqual(game.card_fun(0, couch_tube), 2)

        game.play_card(0, 0)
        self.assertEqual(game.card_fun(0, couch_tube), 6)


if __name__ == "__main__":
    unittest.main()
