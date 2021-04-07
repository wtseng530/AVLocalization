import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
from biCLR import biCLR
from datamodule_pl import LocDataModule
from argparse import ArgumentParser
from utils import OnlineEvaluator

def cli_main():

    parser = ArgumentParser()

    parser = biCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    dm = LocDataModule(rgb_dir=args.rgb_dir,
                     depth_dir=args.depth_dir,
                     batch_size= args.batch_size,
                     patch_dim= args.patch_dim,
                     num_workers=args.num_workers,
                     val_split=args.val_split)

    dm.transform = None
    args.num_samples = dm.num_samples()

    args.maxpool1 = False
    args.first_conv = False

    #model
    model = biCLR(**args.__dict__)

    args.online_ft = True
    online_evaluator = OnlineEvaluator()
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    early_stop = early_stopping.EarlyStopping(monitor='val_loss')
    callbacks = [model_checkpoint, online_evaluator, early_stop] if args.online_ft \
        else [model_checkpoint, early_stop]
    # fit
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend='ddp_spawn' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run= 0
    )

    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    cli_main()
