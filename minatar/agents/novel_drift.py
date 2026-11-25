import random

import numpy as np
import torch


class NovelDrift:
    """Compare one-update Q drift on a novel pair and older replay pairs."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.last_window = -1

    def capture_before(self, agent, state, action):
        window = agent.update_cnt // 5000
        if window == self.last_window:
            return None

        replay = self.rng.sample(agent.replay_buffer.buffer, agent.batch_size)
        boards = [state, *(transition.state for transition in replay)]
        actions = [action, *(transition.action for transition in replay)]
        states = torch.as_tensor(
            np.stack(boards), dtype=torch.float32, device=agent.device
        )
        states = states.permute(0, 3, 1, 2).contiguous()
        actions = torch.as_tensor(actions, dtype=torch.long, device=agent.device)
        actions = actions.unsqueeze(1)
        before = self._action_scores(agent.Q_net, states, actions)
        self.last_window = window
        return states, actions, before

    def metrics_after(self, agent, capture):
        states, actions, before = capture
        after = self._action_scores(agent.Q_net, states, actions)
        drift = (after - before).abs()
        return {
            "novel_drift/novel_pair_q_abs": drift[0].item(),
            "novel_drift/replay_pair_q_abs": drift[1:].mean().item(),
        }

    @staticmethod
    @torch.no_grad()
    def _action_scores(network, states, actions):
        return network(states).gather(1, actions).squeeze(1)
