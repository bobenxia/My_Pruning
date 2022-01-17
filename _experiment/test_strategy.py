import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch_pruning as tp
from model.cifar import resnet as resnet


def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 32, 32))

    def prune_conv(conv, amount=0.2):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        # pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.strategy.L1Strategy()
        strategy = tp.strategy.GSGDStrategy()
        pruning_index = strategy(conv.weight, amount=amount)
        print(amount, len(pruning_index))
        # plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        # plan.exec()

    blk_id = 0
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock):
            a = m.conv1
            b = m.conv2
            prune_conv(m.conv1, 0.2)
            prune_conv(m.conv2, 0.2)
            blk_id += 1
    return model


previous_ckpt = 'save/train_and_prune/ResNet18-round%d.pth' % (1 - 1)
print("Pruning round %d, load model from %s" % (1, previous_ckpt))
model = torch.load(previous_ckpt)
prune_model(model)
# torch.save(model, 'save/train_and_prune/ResNet18-round%d.pth' % (1))
