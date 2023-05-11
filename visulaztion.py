import argparse
import random

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import Actor, Critic
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
# from offlinerlkit.policy import TD3BCPolicy
from offlinerlkit.policy import ISARPolicy
from offlinerlkit.modules import ActorProb, TanhDiagGaussian
from offlinerlkit.policy import AWACPolicy

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="isar")
    parser.add_argument("--task", type=str, default="hopper-expert-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--device", type=str, default='cuda:0' if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def normalize_rewards(dataset):
    terminals_float = np.zeros_like(dataset["rewards"])
    for i in range(len(terminals_float) - 1):
        if np.linalg.norm(dataset["observations"][i + 1] -
                            dataset["next_observations"][i]
                            ) > 1e-6 or dataset["terminals"][i] == 1.0:
            terminals_float[i] = 1
        else:
            terminals_float[i] = 0

    terminals_float[-1] = 1

    # split_into_trajectories
    trajs = [[]]
    for i in range(len(dataset["observations"])):
        trajs[-1].append((dataset["observations"][i], dataset["actions"][i], dataset["rewards"][i], 1.0-dataset["terminals"][i],
                        terminals_float[i], dataset["next_observations"][i]))
        if terminals_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])
    
    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    # normalize rewards
    dataset["rewards"] /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset["rewards"] *= 1000.0

    return dataset



def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
    if ("halfcheetah" in args.task or "walker2d" in args.task or "hopper" in args.task):
        dataset = normalize_rewards(dataset)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    critic_v = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    actor = Actor(actor_backbone, args.action_dim, device=args.device)

    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    critic_v = Critic(critic_v, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.critic_lr)

    # scaler for normalizing observations
    scaler = StandardScaler(mu=obs_mean, std=obs_std)

    # create policy
    policy = ISARPolicy(
        actor,
        critic1,
        critic2,
        critic_v,
        actor_optim,
        critic1_optim,
        critic2_optim,
        critic_v_optim,
        tau=args.tau,
        gamma=args.gamma,
        max_action=args.max_action,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        update_actor_freq=args.update_actor_freq,
        alpha=args.alpha,
        expectile = args.expectile,
        temperature  = args.temperature,
        scaler=scaler
    )


    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    path = './log/hopper-expert-v2/ISAR/seed_5&timestamp_23-0317-132313/model/policy.pth'
    state_dicts = torch.load(path)
    policy.load_state_dict(state_dicts)
    policy.eval()

    fig, ax = plt.subplots()
    tsne = TSNE(n_components=2, n_iter=1000, random_state=10)
    pca = PCA(n_components=2)
    all_actions = []
    all_policy_actions = []
    all_rewards = []
    sample_num = 10
    for i in range(sample_num):
        print(i)
        batch = buffer.sample(args.batch_size)
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        rewards = rewards.cpu().numpy()

        # print(rewards.shape)
        all_rewards.append(rewards)
        all_actions.append(actions.cpu().numpy())

        with torch.no_grad():
            policy_actions = policy.actor(obss).cpu().numpy()
        all_policy_actions.append(policy_actions)
      

    all_actions = np.array(all_actions).reshape(sample_num * args.batch_size, -1)
    all_policy_actions = np.array(all_policy_actions).reshape(sample_num * args.batch_size, -1)
    rewards = np.array(all_rewards).reshape(sample_num * args.batch_size, -1)
    rewards = np.exp(rewards)
    # tsne.fit_transform(all_actions)
    actions = np.concatenate([all_actions, all_policy_actions],axis=0)
    points = tsne.fit_transform(actions)
    points1 = points[:all_actions.shape[0],:]
    # points2 = points[all_actions.shape[0]:all_actions.shape[0]*2,:]
    points2 = points[all_actions.shape[0]:,:]
    norm_reward = (rewards - rewards.min()) / (rewards.max() - rewards.min())

    fig, ax = plt.subplots()
    # c = ax.scatter(points1[:,0], points1[:,1], c=norm_reward,s=5,cmap='RdBu')
    ax.scatter(points2[:,0], points2[:,1], color='r',s=10,label='policy action')
    # ax.scatter(points3[:,0], points3[:,1], color='#ADFF2F',s=10,label='policy action awac')
    c = ax.scatter(points1[:,0], points1[:,1], c=norm_reward,s=10,cmap='plasma',label='dataset action')
  
    fig.colorbar(c, ax=ax)
    plt.legend()
    plt.savefig('action_visulize_{}.png'.format(args.task))
    print('succesful save')
        

        

if __name__ == "__main__":
    train()