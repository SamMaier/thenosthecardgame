"""Exercise card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class MorningWalkBehavior(CardBehavior):
    """Start the next day with two additional Energy; play first today."""

    has_tomorrow_action = True

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return not player.played_today

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy += 2


class MorningBikeBehavior(CardBehavior):
    """Start the next day with five additional Energy; play first today."""

    has_tomorrow_action = True

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return not player.played_today

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy += 5


class MorningRunBehavior(CardBehavior):
    """Start the next day with seven additional Energy; play first today."""

    has_tomorrow_action = True

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return not player.played_today

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy += 7


class ZumbaBehavior(CardBehavior):
    """Start the next day with three additional Energy."""

    has_tomorrow_action = True

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy += 3


class PlayWithTheKidsBehavior(CardBehavior):
    """Score a bonus when this player has the largest hand at day's end."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        has_more_cards = all(
            len(player.hand) > len(opponent.hand)
            for opponent in game.players
            if opponent is not player
        )
        bonus = 3 if has_more_cards else 0
        return card.effective_base_fun + bonus


class WrestleTheKidsBehavior(CardBehavior):
    """Score one additional Fun for each card remaining in this player's hand."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + len(player.hand)


class WakesurfBehavior(CardBehavior):
    """Score a bonus when this player's hand is empty at day's end."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        bonus = 6 if not player.hand else 0
        return card.effective_base_fun + bonus


class SlalomStartBehavior(CardBehavior):
    """Score a bonus when this player has exactly one card at day's end."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        bonus = 6 if len(player.hand) == 1 else 0
        return card.effective_base_fun + bonus


class SkiOnCousinsShouldersBehavior(CardBehavior):
    """Score a bonus when this player's total is below at least half of opponents."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        opponents_with_higher_scores = sum(
            player.fun < opponent.fun
            for opponent in game.players
            if opponent is not player
        )
        bonus = (
            5
            if opponents_with_higher_scores * 2 >= len(game.players) - 1
            else 0
        )
        return card.effective_base_fun + bonus


class ThrowABaseballBehavior(CardBehavior):
    """Pick one card from the Suitcase when this card is played."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.pick_from_suitcase(game.players.index(player))


class ChuckAFrisbeeBehavior(CardBehavior):
    """Return this card to its owner's hand after it scores."""

    def on_score(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.played_today.remove(card)
        game.give_card(game.players.index(player), card)


class PaddleboatBehavior(CardBehavior):
    """Gain two Fun for each card acquired after this card today."""

    def on_card_acquire(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        acquired_card: CardInstance,
    ) -> None:
        energy_cubes = int(source.markers.get("energy_cubes", 0)) + 1
        source.markers["energy_cubes"] = energy_cubes

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + 2 * int(
            card.markers.get("energy_cubes", 0)
        )


class PaddleboardBehavior(CardBehavior):
    """Discard this player's hand and pick the same number of Suitcase cards."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        hand_size = len(player.hand)
        game.discard_cards_from_hand(player_index, range(hand_size))
        for _ in range(hand_size):
            game.pick_from_suitcase(player_index)


class NavySEALingBehavior(CardBehavior):
    """Discard any number of hand cards for three Fun per discarded card."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        hand = tuple(player.hand)
        chooser = game.ais[player_index].choose_cards_to_discard
        choices = tuple(chooser(game, player_index, hand)) if hand else ()
        game.discard_cards_from_hand(player_index, choices)
        card.markers["energy_cubes"] = len(choices)

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + 3 * int(
            card.markers.get("energy_cubes", 0)
        )


class KayakBehavior(CardBehavior):
    """Allow optional remaining Energy to increase this card's Fun."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        additional_energy = game.choose_energy_to_spend(
            player_index, card, player.energy
        )
        player.energy -= additional_energy
        card.markers["energy_cubes"] = additional_energy

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + int(card.markers.get("energy_cubes", 0))


ZUMBA = CardDefinition(
    slug="zumba",
    title="Zumba",
    tags=frozenset({"Exercise"}),
    cost=4,
    base_fun=2,
    behavior=ZumbaBehavior(),
)

PLAY_WITH_THE_KIDS = CardDefinition(
    slug="play-with-the-kids",
    title="Play With the Kids",
    tags=frozenset({"Exercise"}),
    cost=4,
    base_fun=3,
    behavior=PlayWithTheKidsBehavior(),
)

WRESTLE_THE_KIDS = CardDefinition(
    slug="wrestle-the-kids",
    title="Wrestle the Kids",
    tags=frozenset({"Exercise"}),
    cost=5,
    base_fun=1,
    behavior=WrestleTheKidsBehavior(),
)

WAKESURF = CardDefinition(
    slug="wakesurf",
    title="Wakesurf",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=4,
    base_fun=1,
    behavior=WakesurfBehavior(),
)

SLALOM_START = CardDefinition(
    slug="slalom-start",
    title="Slalom Start",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=5,
    base_fun=2,
    behavior=SlalomStartBehavior(),
)

SKI_ON_COUSINS_SHOULDERS = CardDefinition(
    slug="ski-on-cousins-shoulders",
    title="Ski on Cousin's Shoulders",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=5,
    base_fun=3,
    behavior=SkiOnCousinsShouldersBehavior(),
)

MORNING_WALK = CardDefinition(
    slug="morning-walk",
    title="Morning Walk",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=1,
    behavior=MorningWalkBehavior(),
)

MORNING_BIKE = CardDefinition(
    slug="morning-bike",
    title="Morning Bike",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=3,
    behavior=MorningBikeBehavior(),
)

MORNING_RUN = CardDefinition(
    slug="morning-run",
    title="Morning Run",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=5,
    behavior=MorningRunBehavior(),
)

THROW_A_BASEBALL = CardDefinition(
    slug="throw-a-baseball",
    title="Throw a Baseball",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=2,
    base_fun=1,
    behavior=ThrowABaseballBehavior(),
)

CHUCK_A_FRISBEE = CardDefinition(
    slug="chuck-a-frisbee",
    title="Chuck a Frisbee",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=3,
    base_fun=3,
    behavior=ChuckAFrisbeeBehavior(),
)

PADDLEBOAT = CardDefinition(
    slug="paddleboat",
    title="Paddleboat",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=4,
    behavior=PaddleboatBehavior(),
)

PADDLEBOARD = CardDefinition(
    slug="paddleboard",
    title="Paddleboard",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=4,
    base_fun=6,
    behavior=PaddleboardBehavior(),
)

NAVY_SEALING = CardDefinition(
    slug="navy-sealing",
    title="Navy SEALing",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=6,
    base_fun=4,
    behavior=NavySEALingBehavior(),
)

KAYAK = CardDefinition(
    slug="kayak",
    title="Kayak",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=2,
    base_fun=2,
    behavior=KayakBehavior(),
)
