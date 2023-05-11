import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import SACPolicy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler


class AWACPolicy(SACPolicy):
    """
    AWAC
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        awac_temperature:float=1.0,
    ) -> None:

        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
        )

        self.temperature = awac_temperature
    
    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        # for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
        #     o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            # noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions, _  = self.actforward(next_obss)
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q
        
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor

        with torch.no_grad():
            pi_actions, _ = self.actforward(obss)
            v = torch.min(self.critic1(obss, pi_actions), self.critic2(obss, pi_actions))
            q1 = torch.min(self.critic1(obss, actions), self.critic2(obss, actions))
            adv = q1-v
            weights = torch.clamp(adv.exp()/ self.temperature, None, 100.0)

        dist = self.actor(obss)
        log_probs = dist.log_prob(actions)
        actor_loss = - (weights * log_probs).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self._last_actor_loss = actor_loss.item()
        self._sync_weight()
    

        return {
            "loss/actor": self._last_actor_loss,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }