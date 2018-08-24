import numpy as np


class Fuser(object):
    # Return the feature required by classifier network. In form of ndarray.
    # 约定：传入features应该是这样的：[frames, feature_length]
    def __init__(self, fuse_type=None):
        assert fuse_type in ['average', 'cnn', 'max', 'concat', 'none', 'lstm']
        self.fuse_type = fuse_type

    def __call__(self, features):
        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 2
        if self.fuse_type == 'cnn':
            rst = np.expand_dims(np.stack(features), 0)
            assert len(rst.shape) == 3
            # print('In cnn fuser, the output shape is:', rst.shape)
        elif self.fuse_type == 'lstm':
            rst = features
        elif self.fuse_type == 'average':
            rst = np.mean(features, 0)
            # print('In aveage fuser, the output shape is:' , rst.shape)
        elif self.fuse_type == 'max':
            rst = np.max(features, 0)
        elif self.fuse_type == 'concat':
            # rst = features.reshape([features.shape[0], features.shape[1] * features.shape[2]])
            rst = np.concatenate(features)
            # print('In concat fuser, the output shape is:', rst.shape)
        elif self.fuse_type == 'none':
            rst = features
        else:
            raise ValueError(
                "Input fuse_type must be in ['average', 'cnn', 'max'], but you input {}.".format(self.fuse_type))
        return rst