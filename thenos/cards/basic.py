"""The first three implemented cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class FajitasBehavior(CardBehavior):
    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 4, card)


class WingspanBehavior(CardBehavior):
    """Score this card immediately, in addition to its normal scoring."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.fun += game.card_fun(game.players.index(player), card)


class TheCrewBehavior(CardBehavior):
    """Remember whether enough opponents played Board Games before this."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        opponents = [opponent for opponent in game.players if opponent is not player]
        board_game_opponents = sum(
            any(
                "Board Game" in played_card.tags
                for played_card in opponent.played_today
            )
            for opponent in opponents
        )
        if board_game_opponents * 2 >= len(opponents):
            card.markers["energy_cube"] = True
            card.markers["_the_crew_energy_cube"] = True

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + (
            2 if card.markers.get("_the_crew_energy_cube") else 0
        )


class SoloBehavior(CardBehavior):
    """Score a bonus when no opponent played a Board Game previously today."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        opponents = [opponent for opponent in game.players if opponent is not player]
        if not any(
            "Board Game" in played_card.tags
            for opponent in opponents
            for played_card in opponent.played_today
        ):
            card.markers["energy_cube"] = True
            card.markers["_solo_energy_cube"] = True

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + (
            2 if card.markers.get("_solo_energy_cube") else 0
        )


class CarcassonneBehavior(CardBehavior):
    """Score a bonus when directly between two Board Games played today."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        if card_position is None or card_position == 0:
            return card.effective_base_fun

        if card_position + 1 >= len(player.played_today):
            return card.effective_base_fun

        neighbors = (
            player.played_today[card_position - 1],
            player.played_today[card_position + 1],
        )
        if all("Board Game" in neighbor.tags for neighbor in neighbors):
            return card.effective_base_fun + 4
        return card.effective_base_fun


class FiveTenFifteenBehavior(CardBehavior):
    """Pick one card from the Suitcase when this card is played."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.pick_from_suitcase(game.players.index(player))


class SanJuanBehavior(CardBehavior):
    """Track a qualifying player and reward them for avoiding Board Games."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        eligible_player_indices = [
            player_index
            for player_index, candidate in enumerate(game.players)
            if len(candidate.played_today) <= 2
        ]
        if not eligible_player_indices:
            return

        player_index = game.players.index(player)
        target_player_index = game.choose_player(
            player_index, eligible_player_indices
        )
        card.markers["energy_cube"] = True
        card.markers["target_player_index"] = target_player_index

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        target_player_index = card.markers.get("target_player_index")
        if not isinstance(target_player_index, int):
            return card.effective_base_fun

        target_player = game.players[target_player_index]
        if not any(
            "Board Game" in played_card.tags
            for played_card in target_player.played_today
        ):
            return card.effective_base_fun + 3
        return card.effective_base_fun


class PuertoRicoBehavior(CardBehavior):
    """Mark a Suitcase card and reward it surviving through end-of-day."""

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return len(game.cards_played_before(player, card)) < 2

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        target = game.choose_suitcase_target(game.players.index(player))
        target_marker = f"_puerto_rico_target_{card.instance_id}"
        target.markers["energy_cube"] = True
        target.markers[target_marker] = True
        card.markers["suitcase_target"] = target
        card.markers["suitcase_target_marker"] = target_marker

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        target = card.markers.get("suitcase_target")
        target_marker = card.markers.get("suitcase_target_marker")
        target_survived = (
            isinstance(target, CardInstance)
            and isinstance(target_marker, str)
            and bool(target.markers.get(target_marker))
            and any(suitcase_card is target for suitcase_card in game.suitcase)
        )
        return card.effective_base_fun + (4 if target_survived else 0)


class SettlersCitiesAndKnightsBehavior(CardBehavior):
    """Discard one card from hand to gain this card's Fun bonus."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        if not player.hand:
            return

        player_index = game.players.index(player)
        hand = tuple(player.hand)
        choice = game.ais[player_index].choose_card_to_discard(
            game, player_index, hand
        )
        if choice < 0 or choice >= len(hand):
            raise ValueError(f"AI returned invalid hand discard index: {choice}")

        game.discard_from_hand(player_index, choice)
        card.markers["discarded_card"] = True

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        bonus = 4 if card.markers.get("discarded_card") else 0
        return card.effective_base_fun + bonus


class EpicPrankBehavior(CardBehavior):
    """Discard an Item from hand to increase this card's Fun."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        item_cards = tuple(
            hand_card for hand_card in player.hand if "Item" in hand_card.tags
        )
        if not item_cards:
            return

        player_index = game.players.index(player)
        choice = game.ais[player_index].choose_card_to_discard(
            game, player_index, item_cards
        )
        if choice < 0 or choice >= len(item_cards):
            raise ValueError(f"AI returned invalid Item discard index: {choice}")

        discarded_item = item_cards[choice]
        hand_index = next(
            index
            for index, hand_card in enumerate(player.hand)
            if hand_card is discarded_item
        )
        game.discard_from_hand(player_index, hand_index)
        card.markers["discarded_item"] = discarded_item

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        bonus = 6 if card.markers.get("discarded_item") is not None else 0
        return card.effective_base_fun + bonus


class EpicDuelsBehavior(CardBehavior):
    """Allow the player to play additional cards during this turn."""

    def allows_extra_card_plays(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return True


class FitToPrintBehavior(CardBehavior):
    """Score a bonus when this player played more cards than every opponent."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        player_card_count = len(player.played_today)
        opponent_card_counts = [
            len(opponent.played_today)
            for opponent in game.players
            if opponent is not player
        ]
        bonus = 4 if all(
            player_card_count > opponent_count
            for opponent_count in opponent_card_counts
        ) else 0
        return card.effective_base_fun + bonus


class EuchreTournamentBehavior(CardBehavior):
    """Pick the currently visible Item and Food cards from the Suitcase."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        targets = tuple(
            suitcase_card
            for suitcase_card in game.suitcase
            if {"Item", "Food"} & suitcase_card.tags
        )
        if targets:
            game.pick_suitcase_cards(game.players.index(player), targets)


class EuchreTournamentAwardsCeremonyBehavior(CardBehavior):
    """Pick three cards from the Suitcase, with each slot refilled immediately."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        for _ in range(3):
            game.pick_from_suitcase(player_index)


class MakePlansBehavior(CardBehavior):
    """Pick two cards from the Suitcase, with each slot refilled immediately."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        for _ in range(2):
            game.pick_from_suitcase(player_index)


class ScrabbleBehavior(CardBehavior):
    """Trade any number of hand cards for the same number of Suitcase picks."""

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
        for _ in choices:
            game.pick_from_suitcase(player_index)


BIOGRAPHY = CardDefinition(
    slug="biography",
    title="Biography",
    tags=frozenset({"Relax"}),
    cost=1,
    base_fun=2,
)

FAJITAS = CardDefinition(
    slug="fajitas",
    title="Fajitas",
    tags=frozenset({"Food"}),
    cost=3,
    behavior=FajitasBehavior(),
)

WINGSPAN = CardDefinition(
    slug="wingspan",
    title="Wingspan",
    tags=frozenset({"Board Game"}),
    cost=3,
    base_fun=1,
    behavior=WingspanBehavior(),
)

THE_CREW = CardDefinition(
    slug="the-crew",
    title="The Crew",
    tags=frozenset({"Board Game"}),
    cost=1,
    base_fun=1,
    behavior=TheCrewBehavior(),
)

SOLO = CardDefinition(
    slug="solo",
    title="Solo",
    tags=frozenset({"Board Game"}),
    cost=3,
    base_fun=3,
    behavior=SoloBehavior(),
)

CARCASSONNE = CardDefinition(
    slug="carcassonne",
    title="Carcassonne",
    tags=frozenset({"Board Game"}),
    cost=2,
    base_fun=1,
    behavior=CarcassonneBehavior(),
)

FIVE_TEN_FIFTEEN = CardDefinition(
    slug="5-10-15",
    title="5 10 15",
    tags=frozenset({"Board Game"}),
    cost=3,
    base_fun=2,
    behavior=FiveTenFifteenBehavior(),
)

SAN_JUAN = CardDefinition(
    slug="san-juan",
    title="San Juan",
    tags=frozenset({"Board Game"}),
    cost=3,
    base_fun=2,
    behavior=SanJuanBehavior(),
)

PUERTO_RICO = CardDefinition(
    slug="puerto-rico",
    title="Puerto Rico",
    tags=frozenset({"Board Game"}),
    cost=4,
    base_fun=3,
    behavior=PuertoRicoBehavior(),
)

SETTLERS_CITIES_AND_KNIGHTS = CardDefinition(
    slug="settlers-cities-and-knights",
    title="Settlers (Cities and Knights)",
    tags=frozenset({"Board Game"}),
    cost=5,
    base_fun=4,
    behavior=SettlersCitiesAndKnightsBehavior(),
)

EPIC_PRANK = CardDefinition(
    slug="epic-prank",
    title="Epic Prank",
    tags=frozenset({"Event"}),
    cost=4,
    base_fun=2,
    behavior=EpicPrankBehavior(),
)

EPIC_DUELS = CardDefinition(
    slug="epic-duels",
    title="Epic Duels",
    tags=frozenset({"Board Game", "Indoors"}),
    cost=2,
    base_fun=2,
    behavior=EpicDuelsBehavior(),
)

FIT_TO_PRINT = CardDefinition(
    slug="fit-to-print",
    title="Fit to Print",
    tags=frozenset({"Board Game", "Indoors"}),
    cost=3,
    base_fun=1,
    behavior=FitToPrintBehavior(),
)

EUCHRE_TOURNAMENT = CardDefinition(
    slug="euchre-tournament",
    title="Euchre Tournament",
    tags=frozenset({"Board Game", "Indoors"}),
    cost=7,
    base_fun=5,
    behavior=EuchreTournamentBehavior(),
)

EUCHRE_TOURNAMENT_AWARDS_CEREMONY = CardDefinition(
    slug="euchre-tournament-awards-ceremony",
    title="Euchre Tournament Awards Ceremony",
    tags=frozenset({"Event", "Outdoors"}),
    cost=4,
    behavior=EuchreTournamentAwardsCeremonyBehavior(),
)

MAKE_PLANS = CardDefinition(
    slug="make-plans",
    title="Make Plans",
    tags=frozenset({"Social"}),
    cost=3,
    behavior=MakePlansBehavior(),
)

SCRABBLE = CardDefinition(
    slug="scrabble",
    title="Scrabble",
    tags=frozenset({"Board Game", "Outdoors"}),
    cost=3,
    base_fun=1,
    behavior=ScrabbleBehavior(),
)

WATERSKI = CardDefinition(
    slug="waterski",
    title="Waterski",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=5,
    base_fun=6,
)
