import torch

from models.i3d import I3D

class I3DFeatureExtractor(torch.nn.Module):
    def __init__(self, feature_type='concat_logit'):
        assert feature_type in ['concat_1024', 'concat_logit', 'add_1024', 'add_logit']
        super(I3DFeatureExtractor, self).__init__()
        self.shape = None
        if feature_type=='concat_1024':
            self.shape = [1, 2048]
        elif feature_type=='concat_logit':
            self.shape = [1, 800]
        elif feature_type=='add_1024':
            self.shape = [1, 1024]
        elif feature_type=='add_logit':
            self.shape = [1, 400]
        self.feature_type = feature_type
        rgb_pt_checkpoint = 'models/model_rgb.pth'
        flow_pt_checkpoint = 'models/model_flow.pth'

        i3d_rgb = I3D(num_classes=400, modality='rgb')
        i3d_rgb.eval()
        i3d_rgb.load_state_dict(torch.load(rgb_pt_checkpoint))

        i3d_flow = I3D(num_classes=400, modality='flow')
        i3d_flow.eval()
        i3d_flow.load_state_dict(torch.load(flow_pt_checkpoint))

        self.i3d_rgb = i3d_rgb
        self.i3d_flow = i3d_flow

    def forward(self, data):
        # print(data[0].shape)
        rgb, flow = data
        rgb_out, rgb_logit, rgb_feature1024 = self.i3d_rgb(rgb)
        flow_out, flow_logit, flow_feature1024 = self.i3d_flow(flow)
        if self.feature_type == 'concat_1024':
            rst = torch.cat([rgb_feature1024, flow_feature1024], dim=1)
            assert rst.shape[1]==2048
        elif self.feature_type == 'concat_logit':
            rst = torch.cat([rgb_logit, flow_logit], dim=1)
            assert rst.shape[1]==800
        elif self.feature_type == 'add_1024':
            rst = rgb_feature1024 + flow_feature1024
        elif self.feature_type == 'add_logit':
            rst = rgb_logit + flow_logit
        return rst
