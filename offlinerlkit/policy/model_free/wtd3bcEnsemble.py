import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import TD3Policy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.policy import BasePolicy


class WBCEnsemblePolicy(BasePolicy):
    """
    TD3+BC <Ref: https://arxiv.org/abs/2106.06860>
    """

    def __init__(
        self,
        actor: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critics: nn.ModuleList,
        critics_optim: torch.optim.Optimizer,
        critic_v: nn.Module,
        critic_v_optim: torch.optim.Optimizer,
        critic_1:nn.Module,
        critic_1_optim:torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        max_action: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        alpha: float = 2.5,
        expectile:float = 0.8,
        temperature: float = 3.0,
        scaler: StandardScaler = None,
        
    ):
        super().__init__()
        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.actor_old.eval()
        self.actor_optim = actor_optim
        self._alpha = alpha
        self.scaler = scaler
        self.critic_v = critic_v
        self.critic_v_optim = critic_v_optim
        self.expectile = expectile
        self.temperature = temperature

        self.q_critics = critics
        self.q_critics_old = deepcopy(critics)
        self.q_critics_old.eval()
        self.critics_optim = critics_optim

        self.critic_1 = critic_1
        self.critic_1_old = deepcopy(critic_1)
        self.critic_1_old.eval()
        self.critic_1_optim = critic_1_optim


        self._tau = tau
        self._gamma = gamma

        self._max_action = max_action
        self.exploration_noise = exploration_noise
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._freq = update_actor_freq

        self._cnt = 0
        self._last_actor_loss = 0
    
    def train(self) -> None:
        self.actor.train()        
        self.q_critics.train()
        self.critic_1.train()
        self.critic_v.train()

    def eval(self) -> None:
        self.actor.eval()
        self.q_critics.eval()
        self.critic_v.eval()
        self.critic_1.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_1_old.parameters(), self.critic_1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.q_critics_old.parameters(), self.q_critics.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action
    

    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
        return weight * (diff**2)


    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]


        with torch.no_grad():
            qas = self.q_critics_old(obss, actions)
            ensemble_min = torch.min(qas, 0)[0] # (num_critics, batch, 1)
            q_min = torch.min(ensemble_min, self.critic_1_old(obss, actions))
            # q_min = torch.min(self.critic1_old(obss, actions), self.critic2_old(obss, actions))
        
        current_v = self.critic_v(obss)
        critic_v_loss = self._expectile_regression(q_min - current_v).mean()
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()

        # update critic
        with torch.no_grad():
            # noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            # next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            # next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            # target_q = rewards + self._gamma * (1 - terminals) * next_q

            next_v = self.critic_v(next_obss)
            target_q = rewards + self._gamma *(1 - terminals) * next_v # (batch, 1)
            # clip_return = 1 / (1 - self._gamma)
            # target_q = torch.clamp(target_q, -clip_return, 0)
        
        qs = self.q_critics(obss, actions) # (num_critic, batch, 1)
        critics_loss = ((qs - target_q.unsqueeze(0)).pow(2)).mean(dim=(1, 2)).sum()

        self.critics_optim.zero_grad()
        critics_loss.backward()
        self.critics_optim.step()

        q1 = self.critic_1(obss, actions)
        critic1_loss = (q1 - target_q).pow(2).mean()
        self.critic_1_optim.zero_grad()
        critic1_loss.backward()
        self.critic_1_optim.step()

        with torch.no_grad():
            v = self.critic_v(obss)
            # actions_next = self.actor(next_obss)
            # q = r_tensor + self.args.gamma * self.critic_target_network(inputs_next_norm_tensor, actions_next)
            # q = rewards + self._gamma *(1-terminals)* torch.min(self.critic1_old(next_obss, actions_next),
            #                                                     self.critic2_old(next_obss, actions_next))
            q = self.q_critics_old(obss, actions)
            q_s = torch.min(q, 0)[0] # (num_critics, batch, 1)
            q_min = torch.min(q_s, self.critic_1_old(obss, actions))
            adv = q_min - v
            weights = torch.clip(torch.exp(adv) * self.temperature, None, 100.0)
        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            q = self.critic_1(obss, a)
            lmbda = self._alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + torch.mean(weights * ((a - actions).pow(2)))
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._sync_weight()
        
        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/critic1": critic1_loss.item(),
            "loss/ensemble_critic": critics_loss.item(),
            "weights": (weights.mean()).item(),
        }