import numpy as np


def add_exploration_arguments(parser):
    parser.add_argument(
        "--exploration", choices=("epsilon-greedy", "ucb"), default="epsilon-greedy",
        help="enable a UCB-style bonus in the replay reward with 'ucb'",
    )
    parser.add_argument("--ucb-coef", type=float, default=1.0, help="UCB reward bonus weight")
    parser.add_argument(
        "--count-key", choices=("simhash", "exact"), default="simhash",
        help="how UCB groups boards when counting visits",
    )
    parser.add_argument(
        "--hash-bits", type=int, default=12, help="length of the SimHash bucket ID",
    )


def make_explorer(args, state_shape, num_actions):
    if args.exploration == "ucb":
        return UCBExploration(
            state_shape, num_actions, args.ucb_coef,
            args.count_key, args.hash_bits, args.seed,
        )
    return None


class UCBExploration:
    def __init__(self, state_shape, num_actions, coefficient, count_key, hash_bits, seed):
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("ucb-coef must be finite and nonnegative")
        if count_key == "simhash" and hash_bits < 1:
            raise ValueError("hash-bits must be positive")

        self.num_actions = num_actions
        self.coefficient = coefficient
        self.counts = {}
        self.first_visits = 0
        self.total_visits = 0
        self.total_bonus = 0.0

        self.projection = None
        if count_key == "simhash":
            hash_seed = np.random.SeedSequence(seed).spawn(1)[0]
            hash_rng = np.random.default_rng(hash_seed)
            self.projection = hash_rng.standard_normal(
                (hash_bits, int(np.prod(state_shape))), dtype=np.float32
            )

    def key(self, state):
        if self.projection is None:
            return state.tobytes()
        features = np.asarray(state, dtype=np.float32).ravel()
        return np.packbits(self.projection @ features >= 0).tobytes()

    def record_visit(self, key, action):
        """Return the collection-time bonus and count the executed action."""
        counts = self.counts.get(key)
        if counts is None:
            counts = np.zeros(self.num_actions, dtype=np.int64)
            self.counts[key] = counts
        bonus = self.coefficient * np.sqrt(
            np.log1p(counts.sum()) / (1 + counts[action])
        )
        self.first_visits += int(counts[action] == 0)
        counts[action] += 1
        self.total_visits += 1
        self.total_bonus += bonus
        return float(bonus)

    def metrics(self):
        return {
            "exploration/buckets": len(self.counts),
            "exploration/first_visit_rate": self.first_visits / self.total_visits,
            "exploration/mean_reward_bonus": self.total_bonus / self.total_visits,
        }
