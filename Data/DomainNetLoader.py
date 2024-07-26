import pickle
import DomainAdaptationTrainer.CST_Utils.common.vision.datasets as datasets
import os
from typing import Optional, List, AnyStr, Tuple
from Data.OfficeHome_Utils.imagelist import ImageList
from Data.OfficeHome_Utils._util import download as download_data
import torchvision.transforms as T
from Data.OfficeHome_Utils.randaugment import rand_augment_transform

rgb_mean = (0.485, 0.456, 0.406)
ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)

class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
          (h, w), output size will be matched to this. If size is an int,
          output size will be (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class DomainNet(ImageList):
    """`DomainNet <http://ai.bu.edu/M3SDA/#dataset>`_ (cleaned version, recommended)

    See `Moment Matching for Multi-Source Domain Adaptation <https://arxiv.org/abs/1812.01754>`_ for details.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'c'``:clipart, \
            ``'i'``: infograph, ``'p'``: painting, ``'q'``: quickdraw, ``'r'``: real, ``'s'``: sketch
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            clipart/
            infograph/
            painting/
            quickdraw/
            real/
            sketch/
            image_list/
                clipart.txt
                ...
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/90ecb35bbd374e5e8c41/?dl=1"),
        ("clipart", "clipart.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip"),
        ("infograph", "infograph.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip"),
        ("painting", "painting.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip"),
        ("quickdraw", "quickdraw.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"),
        ("real", "real.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"),
        ("sketch", "sketch.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"),
    ]
    image_list = {
        "c": "clip",
        "i": "info",
        "p": "paint",
        "q": "quick",
        "r": "real",
        "s": "sketch",
    }
    CLASSES = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil',
               'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat',
               'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench',
               'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang',
               'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket',
               'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera',
               'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan',
               'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud',
               'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile',
               'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut',
               'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant',
               'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant',
               'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower',
               'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee',
               'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones',
               'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital',
               'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream',
               'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf',
               'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster',
               'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave',
               'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom',
               'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush',
               'paint_can', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut',
               'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow',
               'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato',
               'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control',
               'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw',
               'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark',
               'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
               'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat',
               'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
               'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine',
               'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 'table', 'teapot', 'teddy-bear',
               'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 'The_Great_Wall_of_China',
               'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado',
               'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 't-shirt',
               'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide',
               'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

    def __init__(self, root: str, task: str, split: Optional[str] = 'train', download: Optional[float] = False, **kwargs):
        assert task in self.image_list
        assert split in ['train', 'test']
        data_list_file = os.path.join(root, "image_list", "{}_{}.txt".format(self.image_list[task], split))
        print("loading {}".format(data_list_file))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))

        super(DomainNet, self).__init__(root, DomainNet.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())

class TransformFixMatch(object):
    def __init__(self):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.weak = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        self.strong = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong

def load_DomainNet(data_dir:AnyStr, source: AnyStr, target:AnyStr, transforms:Tuple=None):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transforms is None:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

        unlabeled_transform = TransformFixMatch()

        val_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
    else:
        train_transform, unlabeled_transform, val_transform = transforms
    source = ImageList(root=data_dir, task=source, download=False, transform=train_transform)
    unlabeled_target = ImageList(root=data_dir, task=target, download=False, transform=unlabeled_transform)
    valid_target = ImageList(root=data_dir, task=target, download=False, transform=val_transform)
    return source, unlabeled_target, valid_target


"""
if __name__ == '__main__':
    with open('./cst_officehome_args.pkl', 'rb') as rb:
        args = pickle.load(rb)
    args.root = '/mnt/VMSTORE/OfficeHome/'
    dataset = datasets.__dict__[args.data]
    # candidate domains: [
    #    "c" # "clipart",
    #    "i" # "infograph",
    #    "p" # "painting",
    #    "q" # "quickdraw",
    #    "r" # "real",
    #    "s" # "sketch",
    # ] 
    train_source_dataset = dataset(root=args.root, task=args.source, download=False, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, task=args.target, download=False, transform=unlabeled_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
"""


""" 
import numpy as np
def scan_dir(dir_name):
    image_paths = [pname for pname in os.listdir(dir_name) if pname.__contains__('image')]
    label = np.load(os.path.join(dir_name, 'label.npy')).reshape(-1).tolist()
    assert len(label) == len(image_paths)
    return [(f'image_{i}.npy', f'{c}') for i, c in enumerate(label)]

if __name__ == '__main__':
    root_dir = './'
    for split in ['train', 'test']:
        for dname in [
            "clip",
            "info",
            "paint",
            "quick",
            "real",
            "sketch",
        ]:
            dir_name = os.path.join(root_dir, f'domainnet-{dname}-{split}')
            items = scan_dir(dir_name)
            lines = [f"domainnet-{dname}-{split}/{p} {c}\n" for p, c in items]
            with open(
                os.path.join(root_dir, "image_list", "{}_{}.txt".format(dname, split)),
                'w'
            )as fw:
                fw.writelines(lines)
"""