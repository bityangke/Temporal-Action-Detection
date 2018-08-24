import argparse
import os
import time

import numpy as np
import torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from dataset import FeatureDataset
from env import make_env
from ops.a2c_acktr import A2C_ACKTR
from ops.fuser import Fuser
from ops.ppo import PPO
from ops.storage import RolloutStorage
from ops.utils import AverageMeter
from models.module import Policy, load_clf_from_i3d, Classifier


def main(args):
    print('[MAIN] Experiment {} start!'.format(args.exp_name))

    # define necessary variable
    torch.set_num_threads(1)
    feature_length = 800
    filepath = 'None'
    obs_shape = [800, 800, 800]
    num_class = 101
    log_file = "result/rl/" + args.exp_name + "_log.csv"
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    with open(log_file, 'w') as f:
        f.write(
            'updates,num_timesteps,FPS,mean_reward,median_reward,min_reward,max_reward,entropy,value_loss,policy_loss,clf_loss,score,all_top1,all_top5\n')

    # define classifier
    i3d_model_checkpoint = "result/0804_1708_e2e_ucf_model.pth.tar"
    clf = Classifier(feature_length, num_class, isbn=False)
    clf = load_clf_from_i3d(clf, i3d_model_checkpoint)
    clf = torch.nn.DataParallel(clf, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()

    # clf_criterion = torch.nn.CrossEntropyLoss().cuda()
    # clf_optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr)

    # define dataset
    train_dataset = FeatureDataset('features/thumos14/val/data.csv')
    eval_dataset = FeatureDataset('features/thumos14/test/data.csv', is_thumos14_test_folder=True)  # eval detection

    # define environment
    fuser = Fuser(fuse_type='average')
    envs = []
    for i in range(args.num_processes):
        print("[MAIN]\tBegin prepare the {}th env!".format(i))
        envs.append(
            make_env(dataset=train_dataset, classifier=clf, fuser=fuser, observation_space=obs_shape, index=int(i),
                     threshold=0.4))
    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    envs = VecNormalize(envs, ob=False, ret=False, gamma=args.gamma)

    # define actor
    actor_critic = Policy(obs_shape, envs.action_space, output_size=256)
    if args.cuda:
        actor_critic.cuda()

    # define actor's update algorithm
    if args.algo == 'a2c':
        agent = A2C_ACKTR(actor_critic, args.value_loss_coef,
                          args.entropy_coef, lr=args.lr,
                          eps=args.eps, alpha=args.alpha,
                          max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                    args.value_loss_coef, args.entropy_coef, lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = A2C_ACKTR(actor_critic, args.value_loss_coef,
                          args.entropy_coef, acktr=True)

    # prepare rollouts/observation
    rollouts = RolloutStorage(args.num_steps, args.num_processes, (sum(obs_shape),), envs.action_space, 1)
    current_obs = torch.zeros(args.num_processes, (sum(obs_shape, )))
    def update_current_obs(obs, current_obs):
        print(envs.observation_space.shape)
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        current_obs[:, -shape_dim0:] = obs
        return current_obs
    obs = envs.reset()
    current_obs = update_current_obs(obs, current_obs)
    rollouts.observations[0].copy_(current_obs)
    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    # These variables are used to log training.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    score = AverageMeter()
    avg_prop_length = AverageMeter()
    start = time.time()
    top1 = top5 = -1

    # start training
    for j in range(num_updates):
        score.reset()
        if j==10:
            break
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Here is the step!
            obs, reward, done, info = envs.step(cpu_actions)
            print(
                "[MAIN]\tIn updates {}, step {}, startframe {}, endframe {}, totleframe {}, action{}, reward {}, prop_s {}, start_s {}, end_s {}".format(
                    j, \
                    step, [i['start_frame'] for i in info], [i['end_frame'] for i in info],
                    [i['max_frame'] * 16 + 15 for i in info], cpu_actions,
                    reward, [i['proposal_score'] for i in info], [i['start_score'] for i in info],
                    [i['end_score'] for i in info]))
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward
            label = torch.from_numpy(np.expand_dims(np.stack([i['label'] for i in info]), 1)).float()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            score.update(((1 - masks.numpy()) * np.array([i['proposal_score'] for i in info])).mean(),
                         n=np.sum(1 - masks.numpy(), dtype=np.int32))
            avg_prop_length.update(
                np.mean((1 - masks.numpy()) * np.array([i['start_frame'] - i['end_frame'] for i in info])),
                n=np.sum(1 - masks.numpy(), dtype=np.int32))
            if args.cuda:
                masks = masks.cuda()
            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            elif current_obs.dim() == 2:
                current_obs *= masks
            else:
                current_obs *= masks.unsqueeze(2)
            update_current_obs(obs, current_obs)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks, label)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()
        clf_loss = 0
        # if j > 200:
        #     clf_loss = train_classifier(data=rollouts, model=clf, criterion=clf_criterion, optimizer=clf_optimizer)

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            state = {'updates': j + 1,
                     'state_dict': actor_critic.state_dict()}
            filepath = os.path.join(save_path, args.exp_name + "_up{:06d}_model.pth.tar".format(j + 1))
            torch.save(state, filepath)

        # if j % args.clf_test_interval == 0:
        #     top1, top5 = validate(val_loader=eval_loader, model=clf, criterion=clf_criterion)

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                "[MAIN]\tUpdates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}, score {:.5f}".
                    format(j, total_num_steps,
                           int(total_num_steps / (end - start)),
                           final_rewards.mean(),
                           final_rewards.median(),
                           final_rewards.min(),
                           final_rewards.max(), dist_entropy,
                           value_loss, action_loss, score.avg))
            if top1:
                print('[MAIN]\tCLF TEST RUNNED! Top1 {}, TOP5 {}'.format(top1, top5))
            with open(log_file, 'a') as f:
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(j, total_num_steps,
                                                                             int(total_num_steps / (end - start)),
                                                                             final_rewards.mean(),
                                                                             final_rewards.median(),
                                                                             final_rewards.min(),
                                                                             final_rewards.max(), dist_entropy,
                                                                             value_loss, action_loss, clf_loss,
                                                                             score.avg, top1, top5))
            top1 = top5 = None
    return filepath


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    now = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for rewards (default: 0.95)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--clf-test-interval', type=int, default=10,
                        help='classifier test interval, one save per n updates (default: 10)')
    parser.add_argument('--num-frames', type=int, default=10e6, help='number of frames to train (default: 10e6)')
    parser.add_argument('--log-dir', default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='result/rl/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--exp-name', default=now, type=str)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert args.algo in ['a2c', 'ppo', 'acktr']


    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    latest_path = main(args)


    # time.sleep(3)
    # from test_RL import validate_RL
    # rst = validate_RL(20, ckpt=latest_path, force_num=10)
    # import pickle
    # with open("result/rl/" + args.exp_name+"_prediction.pk", 'wb') as f:
    #     pickle.dump(rst, f)
    # import pandas as pd
    # data = pd.DataFrame(rst)
    # data = data.rename(columns={'video_index': "video-id", "start_frame":"t-start", "end_frame":"t-end"})
    # data['t-start'] = data['t-start']/25
    # data['t-end'] = data['t-end']/25
    # with open("result/rl/" + args.exp_name+"_prediction.pk", 'wb') as f:
    #     pd.to_pickle(data, "result/rl/" + args.exp_name+"_prediction_df.pk")