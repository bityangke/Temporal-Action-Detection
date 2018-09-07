import os

fucklistval = [178, 932, 173, 665, 183, 786, 171]



def fuckoff(path):
    # assert isinstance(path, str)
    if int(path.split('/')[-2][-4:]) in fucklistval:
        return True
    return False

def read(txt):
    list_file = []
    list_label = []
    list_l = []
    with open(txt, 'r') as f:
        for i in f:
            f, l, la = i.strip().split(' ')
            list_file.append(f)
            list_l.append(l)
            list_label.append(la)
    result = []
    lastfile = ''
    lastlabel = []
    aline = None
    for path, length, label in zip(list_file, list_l, list_label):
        if fuckoff(path):
            continue

        if path == lastfile:
            lastlabel.append(label)
            aline = path + ' ' + length + ' ' + str(lastlabel) + '\n'
        else:
            if aline:
                result.append(aline)
            lastlabel = [label]
            aline = path + ' ' + length + ' ' + label + '\n'
            # result.append(aline)
            lastfile = path
            # aline = None
    with open('thumos_val_rgb_new.txt', 'w') as f:
        f.writelines(result)
        for i in result:
            print(i)

def ucfread(root, indextxt, outputfilename):
    # indextxt should be downloaded at: http://crcv.ucf.edu/THUMOS14/Class%20Index.txt
    index_dict = {}
    labels = []
    result = []

    # build dict
    with open(indextxt, 'r') as f:
        for i in f:
            ind, label = i.strip().split(' ')
            index_dict[label] = int(ind)
            labels.append(label)
    cnt = 0
    for l in labels:
        subroot = os.path.join(root, l)
        print('Enter:', subroot)
        dirs = os.listdir(subroot)
        for dir in dirs:
            path = os.path.join(subroot, dir)
            print('Enter:', path)
            assert os.path.isdir(path)
            label = index_dict[l] - 1  # From [1,101] to [0, 100] in order fit the training process.
            aline = path + ' ' + str(cnt) + ' ' + str(label) + '\n'
            result.append(aline)
            cnt += 1

    with open(outputfilename, 'w') as f:
        f.writelines(result)


def readucfrgb(root, indextxt, outputfilename):
    # indextxt should be downloaded at: http://crcv.ucf.edu/THUMOS14/Class%20Index.txt
    index_dict = {}
    labels = []
    result = []

    # build dict
    with open(indextxt, 'r') as f:
        for i in f:
            ind, label = i.strip().split(' ')
            index_dict[label] = int(ind)
            labels.append(label)
    cnt = 0

    dirs = os.listdir(root)
    for dir in dirs:
        label = dir.split("_")[1]
        label = index_dict[label]
        aline = os.path.join(root, dir) + ' ' + str(cnt) + ' ' + str(label) + '\n'
        result.append(aline)
        cnt += 1

    with open(outputfilename, 'w') as f:
        f.writelines(result)


def ucfalign(rgbtxt, flowtxt, output):
    rgb = []
    flow = []
    new_rgb = []
    with open(flowtxt) as f:
        for line in f:
            flow.append(line)
    with open(rgbtxt) as f:
        for line in f:
            rgb.append(line)
    rgb_dict = {}

    for ind, line in enumerate(rgb):
        assert isinstance(line, str)
        path = line.strip().split(' ')[0]
        name = path.split('/')[-1].split('.')[0]
        rgb_dict[name] = ind

    for line in flow:
        assert isinstance(line, str)
        path = line.strip().split(' ')[0]
        name = path.split('/')[-1]
        ind = rgb_dict[name]
        new_rgb.append(rgb[ind])
    with open(output, 'w') as f:
        f.writelines(new_rgb)


def readucf(txt, out):
    list_file = []
    list_label = []
    with open(txt, 'r') as f:
        for i in f:
            f, _, l = i.strip().split(' ')
            list_file.append(f)
            list_label.append(l)
    result = []
    for path, label in zip(list_file, list_label):
        file_list = os.listdir(path)
        file_list = [x for x in file_list if x.split('.')[1] == 'jpg']
        cnt = len(file_list)
        aline = path + ' ' + str(cnt) + ' ' + label + '\n'
        result.append(aline)

    with open(out, 'w') as f:
        f.writelines(result)

def readbackground(root, out):
    list_dir = os.listdir(root)
    result = []
    for path in list_dir:
        # big_file = sorted(os.listdir(os.path.join(root ,path)), key=lambda x: int(x[-9:-4]))[-1]
        # big_file =
        # video_length = int(big_file[-9:-4])
        video_length = int(len(os.listdir(os.path.join(root ,path)))/3)
        result.append("{} {} -1\n".format(os.path.join(root, path), video_length))
        print("Enter {}, length {}".format(path, video_length))
    with open(out, 'w') as f:
        f.writelines(result)


if __name__ == '__main__':
    # root = "/nfs_data/lmwang/Data/t15background/data"
    # rgbtxt = "ucf_rgb.txt"
    # read(root, txt)
    # ucfread(root, 'ucf_index.txt', output)
    # readucfrgb(root, 'ucf_index.txt', output)
    # rgbtxt = 'ucf_newrgb.txt'
    # flowtxt = 'ucf_flow.txt'
    # output = 'ucf_listrgb.txt'
    # ucfalign("ucf_listrgb.txt", "ucf_listflow.txt", "ucf_listrgb.txt")
    # readucf(rgbtxt, output)
    readbackground("/nfs_data/lmwang/Data/t15background/data", "background_flow.txt")
