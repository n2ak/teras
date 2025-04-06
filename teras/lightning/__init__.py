import lightning as L
from teras.data import to_ds
from teras.lightning.model import Model, Config, OptimFn


def train(
    model,
    optim_fns: OptimFn,
    loss_fn,
    X, y=None,
    epochs=2,
    validation_set=None, validation_split=None,
    train_batch_size=16,
    val_batch_size=None,
    num_workers=0,
    callbacks=[],
    train_collate_fn=None,
    eval_collate_fn=None,
    **trainer_kwargs,
):
    if not isinstance(optim_fns, list):
        optim_fns = [optim_fns]
    if val_batch_size is None:
        val_batch_size = train_batch_size
    train_ds, valid_ds = to_ds(X, y, validation_set, validation_split)
    config = Config(
        train_dataset=train_ds,
        val_dataset=valid_ds,
        train_collate_fn=train_collate_fn,
        eval_collate_fn=eval_collate_fn,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        optim_fns=optim_fns,
        loss_fn=loss_fn,
    )
    model = Model(model, config)
    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        # accelerator="gpu",
        # devices=[0],
        # accumulate_grad_batches=config.get("accumulate_grad_batches"),
        # check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        # gradient_clip_val=config.get("gradient_clip_val"),
        # precision="16-mixed",
        # limit_val_batches=5,
        # num_sanity_val_steps=0,
        # logger=wandb_logger,
        **trainer_kwargs,
    )
    trainer.fit(model)
    # print("="*60)
    # print(trainer.callback_metrics)
    # print(trainer.logged_metrics)
    # print("="*60)
    # 1/0
