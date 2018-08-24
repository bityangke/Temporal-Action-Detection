import os


def prepare_one_video_annotation(annotation_dict, video_path):
    video_name = video_path.split('/')[-1]
    if video_path[-1]=='/':
        video_name = video_path.split('/')[-2]  # path look like: /home/xx/data/thumos/val/video_validation_1111111/
    if video_name in annotation_dict:
        return annotation_dict[video_name]
    return []


def build_annotation_dict(annotation_root, index_file, FPS=25):
    label_dict = {}
    with open(index_file, 'r') as f:
        for line in f:
            ind, lab = line.strip().split(' ')
            label_dict[lab] = int(ind) - 1  # We assert the label should start at 0!
    assert isinstance(annotation_root, str)
    d = {}
    paths = os.listdir(annotation_root)
    for label in paths:
        if label.split('_')[0] == 'Ambiguous':
            index = -1
        else:
            index = label_dict[label.split('_')[0]]
        with open(os.path.join(annotation_root, label), 'r') as f:
            for line in f:
                name, _, s, e = line.strip().split(' ')
                s = int(eval(s) * FPS)
                e = int(eval(e) * FPS)
                if not (name in d):
                    d[name] = [(s, e, index)]
                else:
                    d[name].append((s, e, index))
    return d