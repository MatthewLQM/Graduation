import torch.nn as nn
import network.layer as layer


class YoloNet(nn.Module):
    def __init__(self, cfg_file):
        super(YoloNet, self).__init__()
        self.blocks = self.load_cfg(cfg_file)
        self.net_info, self.module_list = self.create_module_by_blocks()

    # TODO 实现 forward 函数
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):

    #  TODO 实现 load_weights 函数
    #  TODO 实现 calcu_loss 函数

    def show_blocks(self):
        for block in self.blocks:
            print(block)

    def load_cfg(self, cfg_file):
        """
        :param cfg_file: a configuration file
        :return: a list of blocks. Each blocks describes a block in the neural network to be built.
        Block is represented as a dictionary in the list
        """
        file = open(cfg_file, 'r')
        lines = file.read().split('\n')  # store the lines in a list
        lines = [x for x in lines if len(x) > 0]  # get ride of the empty lines
        lines = [x for x in lines if x[0] != '#']  # get ride of comments
        lines = [x.rstrip().lstrip() for x in lines]  # get ride of fringe whitespaces

        block = {}  # block is a dictionary
        blocks = []  # blocks store in a list

        for line in lines:
            if line[0] == "[":  # This marks the start of a new block
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()
            else:
                key,value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)

        return blocks

    def create_module_by_blocks(self):
        net_info = self.blocks[0] # Captures the information about the input and pre-processing
        module_list = nn.ModuleList()
        prev_filters = 3
        output_filters = []

        for index, x in enumerate(self.blocks[1:]):
            module = nn.Sequential()

            # check the type of block
            # create a new module for the block
            # append to module_list

            # If it's a convolutional layer
            if x["type"] == "convolutional":
                # Get the info about the layer
                activation = x["activation"]
                try:
                    batch_normalize = int(x["batch_normalize"])
                    bias = False
                except:
                    batch_normalize = 0
                    bias = True
                filters = int(x["filters"])
                padding = int(x["pad"])
                kernel_size = int(x["size"])
                stride = int(x["stride"])

                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0

                # Add the convolutional layer
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module("conv_{0}".format(index), conv)

                # Add the Batch Norm Layer If Exist
                if batch_normalize:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(index), bn)

                # Check the activation
                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module("leaky_{0}".format(index), activn)

            # If it's an upsamping layer
            # We use Bilinear2dUpsampling
            elif x["type"] == "upsample":
                stride = int(x["stride"])
                upsample = nn.Upsample(scale_factor=2, mode="nearest")
                module.add_module("upsample_{}".format(index), upsample)

            # TODO 了解这一层是干嘛的
            # If it is a route layer
            elif x["type"] == "route":
                x["layers"] = x["layers"].split(',')
                # Start of a route
                start = int(x["layers"][0])
                # End, if there exists one
                try:
                    end = int(x["layers"][1])
                except:
                    end = 0
                # Positive anotation
                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index
                route = layer.EmptyLayer()
                module.add_module("route_{0}".format(index), route)
                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                else:
                    filters = output_filters[index + start]

            # Shortcut corresponds to skip connections
            elif x["type"] == "shortcut":
                shortcut = layer.EmptyLayer()
                module.add_module("shortcut_{}".format(index), shortcut)

            # Yolo is the detection layer
            elif x["type"] == "yolo":
                mask = x["mask"].split(",")
                mask = [int(x) for x in mask]

                anchors = x["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]

                detection = layer.DetectionLayer(anchors)
                module.add_module("Detection_{}".format(index), detection)

            module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)

        return (net_info, module_list)

net = YoloNet("cfg/yolov3.cfg")
net.show_blocks()
