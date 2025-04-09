import torch
from pytracking.utils.loading import load_network


class NetWrapper:
    """Used for wrapping networks in pytracking.
    Network modules and functions can be accessed directly as if they were members of this class."""
    _rec_iter=0
    def __init__(self, net_path, use_gpu=True, initialize=False, **kwargs):
        self.net_path = net_path
        self.use_gpu = use_gpu
        self.net = None
        self.net_kwargs = kwargs
        if initialize:
            self.initialize()

    def __getattr__(self, name):
        if self._rec_iter > 0:
            self._rec_iter = 0
            return None
        self._rec_iter += 1
        try:
            ret_val = getattr(self.net, name)
        except Exception as e:
            self._rec_iter = 0
            raise e
        self._rec_iter = 0
        return ret_val

    def load_network(self):
        self.net = load_network(self.net_path, **self.net_kwargs)
        if self.use_gpu:
            self.cuda()
        self.eval()

    def initialize(self):
        self.load_network()


class NetWithBackbone(NetWrapper):
    """Wraps a network with a common backbone.
    Assumes the network have a 'extract_backbone_features(image)' function."""

    def __init__(self, net_path, use_gpu=True, initialize=False, image_format='rgb',
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        super().__init__(net_path, use_gpu, initialize, **kwargs)

        self.image_format = image_format
        self._mean = torch.Tensor(mean).view(1, -1, 1, 1)
        self._std = torch.Tensor(std).view(1, -1, 1, 1)

    def initialize(self, image_format='rgb', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().initialize()

    def preprocess_image(self, im: torch.Tensor):
        """Normalize the image with the mean and standard deviation used by the network."""

        if self.image_format in ['rgb', 'bgr']:
            im = im/255

        if self.image_format in ['bgr', 'bgr255']:
            im = im[:, [2, 1, 0], :, :]
        im -= self._mean
        im /= self._std

        if self.use_gpu:
            im = im.cuda()

        return im

    def extract_backbone(self, im: torch.Tensor):
        """Extract backbone features from the network.
        Expects a float tensor image with pixel range [0, 255]."""
        im = self.preprocess_image(im)
        return self.net.extract_backbone_features(im)


class TrackerParams:
    """Class for tracker parameters."""
    def set_default_values(self, default_vals: dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)

    def get(self, name: str, *default):
        """Get a parameter value with the given name. If it does not exists, it return the default value given as a
        second argument or returns an error if no default value is given."""
        if len(default) > 1:
            raise ValueError('Can only give one default value.')

        if not default:
            return getattr(self, name)

        return getattr(self, name, default[0])

    def has(self, name: str):
        """Check if there exist a parameter with the given name."""
        return hasattr(self, name)
    

def parameters(seg_model_path):
    params = TrackerParams()

    ##########################################
    # General parameters
    ##########################################

    params.debug = 0
    params.visualization = False
    params.multiobj_mode = 'parallel'
    params.use_gpu = True

    ##########################################
    # Bounding box init network
    ##########################################
    params.sta_image_sample_size = (30 * 16, 52 * 16)
    params.sta_search_area_scale = 4.0

    params.sta_net = NetWithBackbone(net_path=seg_model_path,
                                     use_gpu=params.use_gpu,
                                     image_format='bgr255',
                                     mean=[102.9801, 115.9465, 122.7717],
                                     std=[1.0, 1.0, 1.0]
                                     )

    params.sta_net.load_network()

    ##########################################
    # Segmentation Branch parameters
    ##########################################
    params.seg_to_bb_mode = 'var'
    params.min_mask_area = 100

    params.image_sample_size = (30 * 16, 52 * 16)
    params.search_area_scale = 6.0
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = None
    params.max_scale_change = (0.8, 1.2)

    # Learning parameters
    params.sample_memory_size = 32
    params.learning_rate = 0.1
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_target_model = True
    params.net_opt_iter = 20
    params.net_opt_update_iter = 3

    return params