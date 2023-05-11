import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from offlinerlkit.policy import TD3Policy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler


class ISARPolicy(TD3Policy):
    """
    TD3+BC <Ref: https://arxiv.org/abs/2106.06860>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        critic_v_optim: torch.optim.Optimizer,
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
            max_action=max_action,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq
        )

        self._alpha = alpha
        self.scaler = scaler
        self.critic_v = critic_v
        self.critic_v_optim = critic_v_optim
        self.expectile = expectile
        self.temperature = temperature
    
    def train(self) -> None:
        self.actor.train()        
        self.critic1.train()
        self.critic2.train()
        self.critic_v.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.critic_v.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
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
            q_min = torch.min(self.critic1_old(obss, actions), self.critic2_old(obss, actions))
        
        current_v = self.critic_v(obss)
        critic_v_loss = self._expectile_regression(q_min - current_v).mean()
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()


        
        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            # noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            # next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            # next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            # target_q = rewards + self._gamma * (1 - terminals) * next_q

            next_v = self.critic_v(next_obss)
            target_q = rewards + self._gamma *(1 - terminals) * next_v
            # clip_return = 1 / (1 - self._gamma)
            # target_q = torch.clamp(target_q, -clip_return, 0)
        
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()


        with torch.no_grad():
            v = self.critic_v(obss)
            # actions_next = self.actor(next_obss)
            # q = r_tensor + self.args.gamma * self.critic_target_network(inputs_next_norm_tensor, actions_next)
            # q = rewards + self._gamma *(1-terminals)* torch.min(self.critic1_old(next_obss, actions_next),
            #                                                     self.critic2_old(next_obss, actions_next))
            q = torch.min(self.critic1_old(obss, actions), self.critic2_old(obss, actions))
            adv = q - v
            weights = torch.clip(torch.exp(adv) * self.temperature, None, 100.0)
        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            q = self.critic1(obss, a)
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
            "loss/critic2": critic2_loss.item(),
            "weights": (weights.mean()).item(),
        }