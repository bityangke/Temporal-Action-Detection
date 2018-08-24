import gym
import torch
from gym import spaces
import numpy as np

from ops.fuser import Fuser


def make_env(dataset, classifier, fuser, observation_space, threshold=0.4, index=-1, verbose=True):
    def _thunk():
        return TADEnv(dataset=dataset, classifier=classifier, fuser=fuser, observation_shape=observation_space,
                      threshold=threshold, index=int(index), shuffle=True, verbose=verbose)

    return _thunk


class TADEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, dataset, classifier, fuser, observation_shape,
                 segment_length=16, initial_sample_interval=4, max_step=20, threshold=0.4, index=-1, shuffle=True,
                 verbose=True):  # 每隔4个seg取一个初始观察点 也就是2.56秒
        print("[ENV]\tBegin prepare Env index: ", index)
        super(TADEnv, self).__init__()

        assert isinstance(observation_shape, list)
        assert len(observation_shape) == 3

        # Declare some variable that in fact will not change.
        self.env_index = index
        self.verbose = verbose
        self.dataset = dataset
        self.initial_sample_interval = initial_sample_interval
        self.fuser = fuser
        self.max_step = max_step
        self.threshold = threshold
        self.segment_length = segment_length
        self.classifier = classifier
        self.ifshuffle = shuffle
        self.classifier.eval()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(sum(observation_shape),),
                                            dtype=np.float32)
        self.softmax = torch.nn.Softmax(1)

        # Declare some true variable. Refresh in self.reset! Arrange in the order the varible shown in self.reset.
        self.waiting_list = []
        # would (possibly) be updated only at self.reset
        self.current_video_index = None
        self.current_video_label = None
        self.current_video_length = None
        self.current_features = None
        self.current_maximum_segment = None
        # would be updated at self.step
        self.current_start_frame = None
        self.current_end_frame = None
        self.current_proposal_score = 0
        self.current_start_score = 0
        self.current_end_score = 0
        self.prop_buffer = []
        self.step_cnt = 0
        print("[ENV]\tFinish prepare Env index:", index)

    def step(self, action):
        """
        :param action:
        :return: ob, reward, done, info
        """
        self.current_start_frame, self.current_end_frame, action = self._get_proposal(action)
        proposal_feature = self._get_proposal_feature()
        edge_feature = self._get_context_feature()
        reward = self._get_reward(proposal_feature=proposal_feature, action=action)
        # similarity = self.get_cosine_similarity(self.get_features())
        # cosreward = similarity - self.current_cosine_similarity
        # self.current_cosine_similarity = similarity
        # reward = reward - float((action[1]<action[0]) or (action[1]>1) or (action[0]<0))
        done = self._get_done()
        self.step_cnt += 1
        self.prop_buffer.append((self.current_start_frame, self.current_end_frame))
        observation = np.concatenate([edge_feature["start"], proposal_feature, edge_feature["end"]])
        assert observation.shape[0] == (800 + 800 + 800)
        assert len(observation.shape) == 1
        return observation, reward, done, \
               {'proposal_score': self.current_proposal_score,
                'start_score': self.current_start_score,
                'end_score': self.current_end_score,
                'start_frame': self.current_start_frame,
                'end_frame': self.current_end_frame,
                'max_frame': (self.current_maximum_segment + 1) * self.segment_length - 1,
                'video_index': self.current_video_index,
                'label': self.current_video_label,
                'maximum_segment': self.current_maximum_segment,
                'video_length': self.current_video_length,
                'step_count': self.step_cnt,
                'proposal_buffer': self.prop_buffer}

    def reset(self, specify_index=None, force_reset=False):
        """
        Reset the environment but not clear the video,
        until all activated point is search.
        :return:
        """
        if (not self.waiting_list) or force_reset:  # If self.waiting_list is empty list, load a new video.

            # get the new video index.
            if not specify_index:
                if self.ifshuffle:
                    self.current_video_index = np.random.randint(0, len(self.dataset))
                else:
                    if self.current_video_index is None:
                        self.current_video_index = 0
                    else:
                        self.current_video_index = (self.current_video_index + 1) % len(self.dataset)
            else:
                assert isinstance(specify_index, int), type(specify_index)
                self.current_video_index = specify_index
            d = self.dataset[self.current_video_index]
            self.current_features, self.current_video_label, self.current_video_length = d['data'], d['label'], d['video_length']
            self.current_maximum_segment = self.current_features.shape[0] - 1  # maximum_segment should be indexable!
            self.current_start_frame, self.current_end_frame, self.current_proposal_score, index = self._get_initial_proposal()
            self.current_start_score = self.current_end_score = self.current_proposal_score
            assert self.current_proposal_score <= 1, self.current_proposal_score
            feature = self._get_proposal_feature()
        else:
            candidate = self.waiting_list.pop()
            self.current_start_frame = candidate["index"] * self.segment_length
            self.current_end_frame = (candidate["index"] + 1) * self.segment_length - 1
            feature = self._get_proposal_feature()
            self.current_start_score = self.current_end_score = self.current_proposal_score = self._get_score(feature)
            assert self.current_proposal_score <= 1, self.current_proposal_score

        self.prop_buffer = []
        self.step_cnt = 0
        if self.verbose:
            print("[ENV]\tI am in RESET! The New Proposal is start {}, end {}, total {}, score {}, video ind {}".format(
                self.current_start_frame, self.current_end_frame, (self.current_maximum_segment + 1) * 16 - 1,
                self.current_proposal_score, self.current_video_index))
        rst = np.concatenate([feature, feature, feature])
        assert rst.shape == self.observation_space.shape
        return rst

    def _get_proposal_feature(self, features=None):
        if not isinstance(features, np.ndarray):
            start_seg = self._frame_to_segment(self.current_start_frame)
            end_seg = self._frame_to_segment(self.current_end_frame)
            assert end_seg >= start_seg
            features = self.current_features[start_seg:end_seg + 1]
        return self.fuser(features)

    def _get_context_feature(self, stride=1):
        before_seg_ind = max(self._frame_to_segment(self.current_start_frame) - stride, 0)
        after_seg_ind = min(self._frame_to_segment(self.current_end_frame) + stride, self.current_maximum_segment)
        return {"start": self.current_features[before_seg_ind], "end": self.current_features[after_seg_ind]}

    def _get_scores(self, feature, clf=None):
        if not clf:
            clf = self.classifier
        with torch.no_grad():
            feature = torch.unsqueeze(torch.from_numpy(feature).cuda(), 0)
        scores = clf(feature)
        scores = self.softmax(scores).detach().cpu().numpy()
        return scores

    def _get_score(self, feature, clf=None):
        return self._get_scores(feature, clf)[0, self.current_video_label]

    # def get_edge_reward(self, st1, st0, limit=10, activate_fn=lambda x: x):
    #     if st1 - st0 > 0:
    #         rst = (st1 - st0) / (st0 + 1e-7)
    #     else:
    #         rst = (st1 - st0) / (1 - st0 + 1e-7)
    #     return activate_fn(max(min(rst, limit), -limit))

    # def get_representativeness_reward(self, features):
    #     assert isinstance(features, np.ndarray)
    #     return 0

    # def get_cosine_similarity(self, features):
    #     assert isinstance(features, np.ndarray)
    #     # Let's say features.shape = [931, 800]
    #     # I will apply matrix computation!
    #     if features.shape[0]==1:
    #         return 0
    #     norm_feat = np.linalg.norm(features, axis=1)    # [931,]
    #     norm_feat = np.expand_dims(norm_feat, 1).repeat((features.shape[1]), axis=1)    # [931, 800]
    #     norm_feat = np.divide(features, norm_feat)    # [931, 800]
    #     matsum = np.dot(norm_feat, np.transpose(norm_feat))     # [931, 931]
    #     matsum -= np.multiply(matsum, np.eye(norm_feat.shape[0]))  # [931, 800] delete elements in the diagonal
    #     assert matsum.shape[0]==matsum.shape[1]
    #     sim = np.sum(matsum)/(matsum.shape[0]*(matsum.shape[0]-1))   # [1]
    #     assert sim<=1, "current reward:".format(sim)
    #     assert sim>=0
    #     return sim

    def _get_reward(self, proposal_feature, action):
        """
        Defined action:
            0: Do nothing
            1: start-1
            2: end+1
        Calculate reward and update all new scores.
        :param proposal_feature:
        :param edge_feature:
        :param action:
        :return:
        """
        proposal_score = self._get_score(proposal_feature)
        if action in [1, 2]:
            current_score = self._get_score(self._frame_to_feature(self.current_start_frame))
            old_score = self.current_start_score
            self.current_start_score = current_score
        elif action in [3, 4]:
            current_score = self._get_score(self._frame_to_feature(self.current_end_frame))
            old_score = self.current_end_score
            self.current_end_score = current_score
        else:
            return 0
        self.current_proposal_score = proposal_score

        # reward = proposal_score * self.get_edge_reward(st1=current_score, st0=old_score)
        # reward = proposal_score * (self.current_end_frame - self.current_start_frame) * self.get_edge_reward(st1=current_score, st0=old_score)

        d = 0
        if action in [1, 4]:
            d = self.segment_length
        elif action in [2, 3]:
            d = -self.segment_length
        reward = proposal_score * ((current_score+old_score)/2 - 0.4*(2-current_score-old_score)/2) * d
        return reward

    def _get_done(self):
        if self.step_cnt == self.max_step:
            if self.verbose:
                print(
                    "[ENV]\tI am in DONE! The Final Proposal is start {}, end {}, total {}, score {}, video ind {}".format(
                        self.current_start_frame, self.current_end_frame, (self.current_maximum_segment + 1) * 16 - 1,
                        self.current_proposal_score, self.current_video_index))
            return True
        return False

    def _get_initial_proposal(self):
        """
        Update self.waiting_list and self.watched_candidates for a new video,
        and return initial proposal and the highest score.
        :return: {"index": the index, "score": the score}
        """
        indices = list(range(0, self.current_maximum_segment, self.initial_sample_interval))
        val = []
        for i in indices:
            feature = self.current_features[i]
            score = self._get_score(feature)
            val.append({"index": i, "score": score})
        waiting_list = sorted(val, key=lambda x: x["score"])
        processed_waiting_list = self._preprocess_waiting_list(waiting_list, threshold=self.threshold)
        if processed_waiting_list == []:
            self.waiting_list = waiting_list[-int(0.3 * len(waiting_list)):]  # avoid all proposal failed.
        else:
            self.waiting_list = processed_waiting_list

        candidate = self.waiting_list.pop()
        index = candidate["index"]
        return index * self.segment_length, (index + 1) * self.segment_length - 1, candidate["score"], index

    def _preprocess_waiting_list(self, waiting_list, threshold):
        return list(filter(lambda x: x["score"] > threshold, waiting_list))

    def _get_proposal(self, action, stride=1):
        """
        Defined action:
            0: Do nothing
            1: start-1
            2: start+1
            3: end-1
            4: end+1
        :param action:
        :return:
        """
        if isinstance(action, np.int64) or isinstance(action, int):
            index = action
        elif isinstance(action, np.ndarray):
            index = np.argmax(action)
        else:
            raise TypeError
        stride = stride * self.segment_length
        if index == 1:
            return max(self.current_start_frame - stride, 0), self.current_end_frame, index
        elif index == 2:
            return min(self.current_start_frame + stride,
                       self.current_end_frame - self.segment_length + 1), self.current_end_frame, index
        elif index == 3:
            return self.current_start_frame, max(self.current_start_frame + self.segment_length - 1,
                                                 self.current_end_frame - stride), index
        elif index == 4:
            return self.current_start_frame, min(self.current_end_frame + stride, self.current_video_length - 1), index
        return self.current_start_frame, self.current_end_frame, index

    def render(self, mode='human'):
        return None

    def _frame_to_feature(self, frame):
        assert isinstance(frame, int) or isinstance(frame, np.int64), type(frame)
        return self.current_features[self._frame_to_segment(frame)]

    def _frame_to_segment(self, index_of_frame):
        assert index_of_frame < self.current_video_length
        assert index_of_frame >= 0
        rst = int(index_of_frame // self.segment_length)
        if rst > self.current_maximum_segment:
            return self.current_maximum_segment
        return rst


if __name__ == "__main__":
    from dataset import FeatureDataset
    from models import buildClassifier

    clf = buildClassifier('result/clf_avg_ucfandt4_model.pth.tar')
    fuser = Fuser(fuse_type='average')
    eval_dataset = FeatureDataset('features/thumos14/test/data.csv', is_thumos14_test_folder=True)
    env = TADEnv(eval_dataset, clf, fuser)
    ob = env.reset()
    # import numpy as np

    a = np.zeros([5])
    nob, rew, done, info = env.step(a)
