# coding: utf-8

import matplotlib.pyplot as plt


from pathlib import Path
import pprint


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite
from torchvision import models

import matplotlib
matplotlib.use("Agg")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default="result", help="output directory")
    parser.add_argument("-d", choices=["mnist", "cifar10"], default="mnist", help="dataset to train on")
    parser.add_argument("--cg", action="store_true", help="visualize computational graph (requires torchviz)")
    parser.add_argument("-b", type=int, default=64, help="batch size")
    parser.add_argument("-e", type=int, default=10, help="epoch")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="beta1 and beta2 of Adam")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)


def main(args):
    log_dir_path = Path(args.o)
    try:
        log_dir_path.mkdir(parents=True)
    except FileExistsError:
        pass

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("GPU mode")
    else:
        args.device = torch.device("cpu")
        print("CPU mode")

    net = PolyNet(in_channels=(1 if args.d == "mnist" else 3)).to(args.device)

    kwds = {"root": ".", "download": True, "transform": transforms.ToTensor()}
    dataset_class = {"mnist": MNIST, "cifar10": CIFAR10}[args.d]
    train_dataset = dataset_class(train=True, **kwds)
    test_dataset = dataset_class(train=False, **kwds)
    train_loader = data.DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.b)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)

    trainer = create_supervised_trainer(net, opt, F.cross_entropy, device=args.device)

    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(F.cross_entropy)
    }
    evaluator = create_supervised_evaluator(net, metrics=metrics, device=args.device)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluate(evaluator, train_loader, test_loader, log_dir_path))
    if args.cg:
        trainer.add_event_handler(Events.ITERATION_STARTED(once=1), computational_graph(
            net, train_dataset, log_dir_path, device=args.device))

    trainer.run(train_loader, max_epochs=args.e)


class PolyNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super().__init__()
        N = 16
        kwds1 = {"kernel_size": 4, "stride": 2, "padding": 1}
        kwds2 = {"kernel_size": 2, "stride": 1, "padding": 0}
        kwds3 = {"kernel_size": 3, "stride": 1, "padding": 1}
        self.conv11 = nn.Conv2d(in_channels, N, **kwds3)
        self.conv12 = nn.Conv2d(in_channels, N, **kwds3)
        self.conv21 = nn.Conv2d(N, N * 2, **kwds1)
        self.conv22 = nn.Conv2d(N, N * 2, **kwds1)
        self.conv31 = nn.Conv2d(N * 2, N * 4, **kwds1)
        self.conv32 = nn.Conv2d(N * 2, N * 4, **kwds1)
        self.conv41 = nn.Conv2d(N * 4, N * 8, **kwds2)
        self.conv42 = nn.Conv2d(N * 4, N * 8, **kwds2)
        self.conv51 = nn.Conv2d(N * 8, N * 16, **kwds1)
        self.conv52 = nn.Conv2d(N * 8, N * 16, **kwds1)

        self.fc = nn.Linear(N * 16 * 3 * 3, n_classes)

    def forward(self, x):
        h = self.conv11(x) * self.conv12(x)
        h = self.conv21(h) * self.conv22(h)
        h = self.conv31(h) * self.conv32(h)
        h = self.conv41(h) * self.conv42(h)
        h = self.conv51(h) * self.conv52(h)
        h = self.fc(h.flatten(start_dim=1))

        return h


def computational_graph(net, train_dataset, file_dir: Path, device=None):
    from torchviz import make_dot

    try:
        file_dir.mkdir(parents=True)
    except FileExistsError:
        pass

    def _computational_graph(engine):
        x, t = train_dataset[0]
        x = x.view(1, *x.size()).to(device)
        t = torch.tensor([t], dtype=torch.long, device=device)
        net.eval()
        y = net(x)
        loss = F.cross_entropy(y, t)
        net.train()

        dot = make_dot(loss, params=dict(net.named_parameters()))
        dot.render(str(file_dir / "cg.dot"))
        print("Computational graph generated")

    return _computational_graph


def evaluate(evaluator, train_loader, test_loader, file_dir: Path):
    try:
        file_dir.mkdir(parents=True)
    except FileExistsError:
        pass

    epochs = []
    train_loss, test_loss = [], []
    train_accuracy, test_accuracy = [], []

    def _evaluate(engine):
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        evaluator.run(test_loader)
        test_metrics = evaluator.state.metrics

        epochs.append(engine.state.epoch)
        train_loss.append(train_metrics["loss"])
        train_accuracy.append(train_metrics["accuracy"])
        test_loss.append(test_metrics["loss"])
        test_accuracy.append(test_metrics["accuracy"])

        print("Epoch {:d}".format(engine.state.epoch))
        print("  train: loss={}, accuracy={}".format(train_metrics["loss"], train_metrics["accuracy"]))
        print("  test: loss={}, accuracy={}".format(test_metrics["loss"], test_metrics["accuracy"]))

        plt.figure()
        plt.plot(epochs, train_loss, label="train")
        plt.plot(epochs, test_loss, label="test")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(str(file_dir / "loss.pdf"))
        plt.close()

        plt.figure()
        plt.plot(epochs, train_accuracy, label="train")
        plt.plot(epochs, test_accuracy, label="test")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.savefig(str(file_dir / "accuracy.pdf"))
        plt.close()

    return _evaluate


if __name__ == "__main__":
    parse_args()
