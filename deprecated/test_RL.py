import time

import torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from dataset import FeatureDataset
from deprecated.env import TADEnv
from deprecated.env import make_env
from models.module import Policy, Classifier, load_clf_from_i3d
from ops.fuser import Fuser

GPU_NUMBER = 4


# device = torch.device("cuda:1")

def run_single_RL(env, actor_critic, specify_video_index=None):
    assert isinstance(env, TADEnv)
    st = time.time()
    ob = env.reset(specify_index=specify_video_index, force_reset=True)
    current_video_index = env.current_video_index
    masks = torch.zeros([1, 1]).cuda()
    states = torch.zeros([1, 1]).cuda()
    done = False
    action_buffer = []
    props = []
    while env.current_video_index == current_video_index:
        while not done:
            with torch.no_grad():
                ob = torch.unsqueeze(torch.from_numpy(ob).float(), 0).cuda()
                value, action, action_log_prob, states = actor_critic.act(
                    inputs=ob,
                    states=masks,
                    masks=states,
                    deterministic=True
                )
            cpu_actions = torch.squeeze(action, 0).cpu().numpy()
            action_buffer.append(cpu_actions)
            ob, reward, done, info = env.step(cpu_actions)
        props.append({
            'start_frame': info['start_frame'],
            'end_frame': info['end_frame'],
            'duration': info['end_frame'] - info['start_frame'] + 1,
            'label': info['label'],
            'score': info['proposal_score'],
            'video_index': current_video_index})
        ob = env.reset()
        done = False
    print('[RL Test]\tVideo {} finished! Take time: {}.'.format(current_video_index, time.time() - st))
    return props


def runner_func(index_queue, result_queue, ckpt):
    env, actor_critic = buildnet(ckpt)
    while True:
        index = index_queue.get()
        props = run_single_RL(env, actor_critic, index)
        result_queue.put(props)


def validate_RL(num_process, ckpt, force_num=None):
    from torch import multiprocessing
    ########### multiprocess
    ctx = multiprocessing.get_context('spawn')
    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    props = []
    workers = [ctx.Process(target=runner_func, args=(index_queue, result_queue, ckpt))
               for i in range(num_process)]
    for w in workers:
        w.daemon = True
        w.start()

    video_num = len(FeatureDataset('features/thumos14/test/data.csv', is_thumos14_test_folder=True))
    if force_num:
        video_num = force_num

    for i in range(video_num):
        index_queue.put(i)

    for i in range(video_num):
        props.extend(result_queue.get())

    return props


def buildnet(actor_ckpt=None, clf_ckpt="result/0804_1708_e2e_ucf_model.pth.tar"):

    fuser = Fuser(fuse_type='average')
    feature_length = 800
    num_class = 101
    i3d_model_checkpoint = clf_ckpt
    clf = Classifier(feature_length, num_class, isbn=False)
    clf = load_clf_from_i3d(clf, i3d_model_checkpoint)
    clf = torch.nn.DataParallel(clf, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()

    obs_shape = [800, 800, 800]
    eval_dataset = FeatureDataset('features/thumos14/test/data.csv', is_thumos14_test_folder=True)
    env = make_env(dataset=eval_dataset, classifier=clf, fuser=fuser, observation_space=obs_shape, index=int(0),
                   threshold=0.4, verbose=False)
    envs = DummyVecEnv([env])
    env = envs.envs[0]
    actor_critic = Policy(obs_shape, envs.action_space, output_size=256)
    if actor_ckpt:
        act_ckpt = torch.load(actor_ckpt)
        actor_critic.load_state_dict(act_ckpt['state_dict'])
    actor_critic = actor_critic.cuda()
    return env, actor_critic


if __name__ == '__main__':
    stime = time.time()

    rst = validate_RL(num_process=10)
    import pickle

    with open('0806_test_rl.pk', 'wb') as f:
        pickle.dump(rst, f)
    print('Finish! Duration:', time.time() - stime)
