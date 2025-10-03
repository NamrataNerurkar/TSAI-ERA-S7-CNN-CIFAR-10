from __future__ import annotations

from model import (
    get_device,
    get_datasets,
    get_dataloader_args,
    get_dataloaders,
    Net,
    get_optimizer,
    get_scheduler,
    train,
    test,
)
import config


def main(epochs: int = config.EPOCHS, data_dir: str = config.DATA_DIR, download: bool = config.DOWNLOAD):
    device, cuda = get_device(seed=config.SEED)
    train_ds, test_ds = get_datasets(data_dir=data_dir, download=download)
    dl_args = get_dataloader_args(
        cuda,
        batch_size_cuda=config.BATCH_SIZE_CUDA,
        batch_size_cpu=config.BATCH_SIZE_CPU,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=config.SHUFFLE,
    )
    train_loader, test_loader = get_dataloaders(train_ds, test_ds, dl_args)

    model = Net().to(device)
    optimizer = get_optimizer(model, lr=config.LR, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, epochs)

    for epoch in range(epochs):
        print(f"EPOCH: {epoch} LR: {scheduler.get_last_lr()[0]:.6f}")
        train_loss, train_acc = train(model, device, train_loader, optimizer)
        scheduler.step()
        test_loss, test_acc = test(model, device, test_loader)
        print(
            f"Train: loss={train_loss:.4f} acc={train_acc:.2f} | "
            f"Test: loss={test_loss:.4f} acc={test_acc:.2f}"
        )


if __name__ == "__main__":
    main()


