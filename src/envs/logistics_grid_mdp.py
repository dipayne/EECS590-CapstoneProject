from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

State = int
Action = int
Transition = Dict[State, float]  # s_next -> prob


@dataclass(frozen=True)
class LogisticsGridSpec:
    rows: int = 6
    cols: int = 6
    depot: Tuple[int, int] = (0, 0)
    customer: Tuple[int, int] = (5, 5)

    blocked: Tuple[Tuple[int, int], ...] = (
        (1, 2), (2, 2), (3, 2), (4, 2), (4, 3), (1, 4)
    )

    step_cost: float = -1.0
    delivery_bonus: float = 20.0

    gamma: float = 0.95
    slip_prob: float = 0.10


class LogisticsGridMDP:
    ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]
    UP, RIGHT, DOWN, LEFT, WAIT = 0, 1, 2, 3, 4

    def __init__(self, spec: LogisticsGridSpec = LogisticsGridSpec()):
        self.spec = spec
        self.rows = spec.rows
        self.cols = spec.cols
        self.nA = 5
        self.gamma = float(spec.gamma)

        self._blocked_states = {self._to_state(rc) for rc in spec.blocked}
        self._depot_state = self._to_state(spec.depot)
        self._customer_state = self._to_state(spec.customer)

        self._states: List[State] = [
            s for s in range(self.rows * self.cols) if s not in self._blocked_states
        ]
        self.nS = self.rows * self.cols

        self._terminal_states = {self._customer_state}

        self.P: List[List[Transition]] = [
            [dict() for _ in range(self.nA)] for _ in range(self.rows * self.cols)
        ]
        self.R: List[List[Dict[State, float]]] = [
            [dict() for _ in range(self.nA)] for _ in range(self.rows * self.cols)
        ]

        self._build_models()

    def _to_state(self, rc: Tuple[int, int]) -> State:
        r, c = rc
        return r * self.cols + c

    def _to_rc(self, s: State) -> Tuple[int, int]:
        return divmod(s, self.cols)

    def states(self) -> List[State]:
        return list(self._states)

    def actions(self, s: State) -> List[Action]:
        if s in self._blocked_states:
            return []
        return list(range(self.nA))

    def is_terminal(self, s: State) -> bool:
        return s in self._terminal_states

    def start_state(self) -> State:
        return self._depot_state

    def goal_state(self) -> State:
        return self._customer_state

    def transitions(self, s: State, a: Action) -> Transition:
        return self.P[s][a]

    def reward(self, s: State, a: Action, s_next: State) -> float:
        if self.is_terminal(s):
            return 0.0
        if s_next == self._customer_state:
            return float(self.spec.delivery_bonus)
        return float(self.spec.step_cost)

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_blocked_rc(self, r: int, c: int) -> bool:
        return self._to_state((r, c)) in self._blocked_states

    def _move_deterministic(self, s: State, a: Action) -> State:
        if s in self._blocked_states:
            return s

        r, c = self._to_rc(s)
        if a == self.UP:
            r2, c2 = r - 1, c
        elif a == self.RIGHT:
            r2, c2 = r, c + 1
        elif a == self.DOWN:
            r2, c2 = r + 1, c
        elif a == self.LEFT:
            r2, c2 = r, c - 1
        else:
            r2, c2 = r, c

        if not self._in_bounds(r2, c2):
            return s
        if self._is_blocked_rc(r2, c2):
            return s
        return self._to_state((r2, c2))

    def _left_of(self, a: Action) -> Action:
        return (a - 1) % 4

    def _right_of(self, a: Action) -> Action:
        return (a + 1) % 4

    def _build_models(self) -> None:
        slip = float(self.spec.slip_prob)
        main_p = 1.0 - 2.0 * slip
        if main_p < 0:
            raise ValueError("slip_prob too large; need 1 - 2*slip_prob >= 0")

        for s in range(self.rows * self.cols):
            for a in range(self.nA):
                if s in self._blocked_states:
                    self.P[s][a] = {s: 1.0}
                    self.R[s][a] = {s: 0.0}
                    continue

                if self.is_terminal(s):
                    self.P[s][a] = {s: 1.0}
                    self.R[s][a] = {s: 0.0}
                    continue

                if a == self.WAIT:
                    s_next = self._move_deterministic(s, a)
                    self.P[s][a] = {s_next: 1.0}
                    self.R[s][a] = {s_next: self.reward(s, a, s_next)}
                    continue

                candidates = [
                    (a, main_p),
                    (self._left_of(a), slip),
                    (self._right_of(a), slip),
                ]

                trans: Transition = {}
                rewards: Dict[State, float] = {}

                for a_eff, p in candidates:
                    s_next = self._move_deterministic(s, a_eff)
                    trans[s_next] = trans.get(s_next, 0.0) + p
                    rewards[s_next] = self.reward(s, a, s_next)

                self.P[s][a] = trans
                self.R[s][a] = rewards
