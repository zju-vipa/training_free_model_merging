from .base_graph import BIGGraph, NodeType


class ResNetGraph(BIGGraph):

    def __init__(self,
                 model,
                 shortcut_name='shortcut',
                 layer_name='layer',
                 head_name='linear',
                 num_layers=3):
        super().__init__(model)

        self.shortcut_name = shortcut_name
        self.layer_name = layer_name
        self.num_layers = num_layers
        self.head_name = head_name

    def add_basic_block_nodes(self, name_prefix, input_node):
        shortcut_prefix = name_prefix + f'.{self.shortcut_name}'
        shortcut_output_node = input_node
        if shortcut_prefix in self.named_modules and len(
                self.get_module(shortcut_prefix)) >= 2:
            """
            小于2的时候为maxpool,可以忽略
            大于等于2的时候为卷积
            """
            # There's a break in the skip connection here, so add a new prefix
            input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX],
                                                      input_node)

            shortcut_output_node = self.add_nodes_from_sequence(
                name_prefix=shortcut_prefix,
                list_of_names=['0', '1'],
                input_node=input_node)

        skip_node = self.add_nodes_from_sequence(name_prefix=name_prefix,
                                                 list_of_names=[
                                                     'conv1', 'bn1',
                                                     NodeType.PREFIX, 'conv2',
                                                     'bn2', NodeType.SUM
                                                 ],
                                                 input_node=input_node)

        self.add_directed_edge(shortcut_output_node, skip_node)

        return skip_node

    def add_bottleneck_block_nodes(self, name_prefix, input_node):
        shortcut_prefix = name_prefix + f'.{self.shortcut_name}'
        shortcut_output_node = input_node
        if shortcut_prefix in self.named_modules and len(
                self.get_module(shortcut_prefix)) >= 2:
            """
            小于2的时候为maxpool,可以忽略
            大于等于2的时候为卷积
            """
            # There's a break in the skip connection here, so add a new prefix
            input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX],
                                                      input_node)
            # print(f"add down sample {shortcut_prefix}")
            shortcut_output_node = self.add_nodes_from_sequence(
                name_prefix=shortcut_prefix,
                list_of_names=['0', '1'],
                input_node=input_node)

        skip_node = self.add_nodes_from_sequence(name_prefix=name_prefix,
                                                 list_of_names=[
                                                     'conv1', 'bn1',
                                                     NodeType.PREFIX, 'conv2',
                                                     'bn2', NodeType.PREFIX,
                                                     'conv3', 'bn3',
                                                     NodeType.SUM
                                                 ],
                                                 input_node=input_node)

        self.add_directed_edge(shortcut_output_node, skip_node)

        return skip_node

    def add_layer_nodes(self, name_prefix, input_node, maxpool_final=False):
        source_node = input_node

        for layer_index, block in enumerate(self.get_module(name_prefix)):
            block_class = block.__class__.__name__

            if block_class == 'BasicBlock':
                source_node = self.add_basic_block_nodes(
                    name_prefix + f'.{layer_index}', source_node)
            elif block_class == 'Bottleneck':
                source_node = self.add_bottleneck_block_nodes(
                    name_prefix + f'.{layer_index}', source_node)
            else:
                raise NotImplementedError(block_class)

        return source_node

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)
        input_node = self.add_nodes_from_sequence('',
                                                  ['conv1', 'bn1', "maxpool"],
                                                  input_node,
                                                  sep='')
        stride_layer = [0, 1]
        for i in range(1, self.num_layers + 1):
            input_node = self.add_layer_nodes(f'{self.layer_name}{i}',
                                              input_node)

        input_node = self.add_nodes_from_sequence(
            '', [NodeType.PREFIX, self.head_name, NodeType.OUTPUT],
            input_node,
            sep='')

        return self


def taskonomy_encoder(model):
    return ResNetGraph(model,
                       shortcut_name='downsample',
                       head_name='compress1',
                       num_layers=4)


if __name__ == '__main__':
    # unit test, nice
    # call from root directory with `python -m "graphs.resnet_graph"`
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import torchvision.models.resnet as resnet
    from visualpriors import visualpriors

    from model_merger import ModelMerge
    from matching_functions import match_tensors_identity, match_tensors_zipit
    from copy import deepcopy

    data_x = torch.rand(4, 3, 224, 224)
    data_y = torch.zeros(4)

    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=4)

    model1, model2 = visualpriors.load_models(
        feature_tasks=["depth_zbuffer", "depth_euclidean"])

    graph1 = taskonomy_encoder(deepcopy(model1)).graphify()
    graph2 = taskonomy_encoder(deepcopy(model2)).graphify()

    merge = ModelMerge(graph1, graph2)
    merge.transform(model3, dataloader, transform_fn=match_tensors_zipit)

    graph1.draw(nodes=range(20))
    graph1.draw(nodes=range(len(graph1.G) - 20, len(graph1.G)))

    print(model.eval().cuda()(data_x.cuda()))

    print(merge(data_x.cuda()))
