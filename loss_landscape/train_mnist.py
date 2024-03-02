import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as T
import numpy as np
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import clip


def load_clip_features(class_names):
    """Create CLIP target labels for class names. Return a normalized tensor of shape (num_classes, 512)."""
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in class_names]).cuda()
    model, preprocess = clip.load('ViT-B/32', text_inputs.device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


epoch = 50
batch_size = 500
repeat_number = 5
num_worker = 8
task = "prime"
datadir = "mnist_dir"

task_class_name = {
    "prime": ["non-prime", "prime"],
    "odd": ["even", "odd"],
    "normal": ["zero","one", "two", "three", "four", "five", "six","seven", "eight", "nine"]
}

label_remaps_dict = {
    "prime": {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 0,
        5: 1,
        6: 0,
        7: 1,
        8: 0,
        9: 0
    },
    "odd": {i: i % 2
            for i in range(10)},
    "normal": {i: i
               for i in range(10)}
}
num_classes = 512


class Mlp(nn.Module):

    def __init__(
        self,
        in_features=784,
        hidden_features=1024,
        out_features=10,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, hidden_features, bias=False)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_features, hidden_features, bias=False)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_features, out_features, bias=False)
        self.act4 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.act4(x)

        return x

def reset_random(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MnistMultiTask(Dataset):

    def __init__(self, ds, tasks=[]) -> None:
        super().__init__()
        self.ds = ds

        self.label_remaps = [label_remaps_dict[t] for t in tasks]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]

        return img, [lm[label] for lm in self.label_remaps]


if __name__ == "__main__":
    print("Num Classes", num_classes)
    print("Classes name", task_class_name[task])
    reset_random(seed=0)

    for rep in range(repeat_number):
        root_dir = f"checkpoints/mnist_clip/{task}/{rep}"
        os.makedirs(root_dir, exist_ok=True)
        train_trans = T.Compose([
            T.RandomCrop(28, padding=4),
            T.RandomRotation(30),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        test_trans = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
        ds = MnistMultiTask(MNIST(datadir,
                                  train=True,
                                  transform=train_trans),
                            tasks=[task])
        ds_test = MnistMultiTask(MNIST(datadir,
                                       train=False,
                                       transform=test_trans),
                                 tasks=[task])

        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_worker)

        dl_test = DataLoader(ds_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_worker)
        net = Mlp(out_features=num_classes)
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=0.01,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        scaler = GradScaler()
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch, verbose=True)
        loss_func = nn.CrossEntropyLoss()
        best_acc = 0
        class_vectors = load_clip_features(task_class_name[task])
        for e in range(epoch):
            print(f'Round {rep} Train/Epoch {e}/{epoch}')
            net.train().cuda()
            loss = 0
            acc = 0
            for img, label in tqdm(dl, total=len(dl), desc="Training"):
                img = img.cuda().reshape(img.shape[0], -1)
                label = label[0].cuda()
                with autocast():
                    encodings = net(img)
                    normed_encodings = encodings / encodings.norm(dim=-1,
                                                                  keepdim=True)
                    logits = (100.0 * normed_encodings @ class_vectors.T)
                    remapped_labels = label
                    # pdb.set_trace()
                    loss_ = loss_func(logits, remapped_labels)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_).backward()
                scaler.step(optimizer)
                scaler.update()

                loss += loss_.cpu().detach().item() * img.shape[0]
                acc += (logits.argmax(
                    dim=1).data == label.data).sum().cpu().detach().item()
            loss /= len(ds)
            acc /= len(ds)
            print(f'Round {rep} Train/loss {loss} acc {acc}')
            print(f'Round {rep} Test/Epoch {e}/{epoch}')
            loss = 0
            acc = 0
            net.eval()
            with torch.no_grad(), autocast():
                for img, label in tqdm(dl_test,
                                       total=len(dl_test),
                                       desc="Test"):
                    img = img.cuda().reshape(img.shape[0], -1)
                    label = label[0].cuda()

                    encodings = net(img)
                    normed_encodings = encodings / encodings.norm(dim=-1,
                                                                  keepdim=True)
                    logits = (100.0 * normed_encodings @ class_vectors.T)
                    remapped_labels = label
                    # pdb.set_trace()
                    loss_ = loss_func(logits, remapped_labels)

                    loss += loss_.cpu().detach().item() * img.shape[0]
                    acc += (logits.argmax(
                        dim=1).data == label.data).sum().cpu().detach().item()
            loss /= len(ds_test)
            acc /= len(ds_test)
            print(f'Round {rep} Test/loss {loss} acc {acc}')
            scheduler.step()
            if acc > best_acc:
                print(f"saving checkpoint {e}")
                torch.save(net.cpu().state_dict(),
                           os.path.join(root_dir, "best.pth"))
