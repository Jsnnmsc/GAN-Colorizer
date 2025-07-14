from torch import nn


class cfg:
    # dataset
    data_amounts = 25000
    val_pct = 0.1
    image_size = 256
    bs = 32
    num_workers = 8

    # model layers
    # gen
    blur = True
    act = nn.ReLU
    nt = None  # NormType.Spectral, NormType.Batch
    sa = True
    sa_extra = False
    gen_update_interval = 1
    # disc
    n_blocks = 3
    dropout = 0.2
    disc_update_interval = 1

    # model LR
    gen_lr = 1e-4
    disc_lr = 2e-4

    # LR for pretrain
    pretrain_lr = 1e-4

    # optimizer
    beta1 = 0.5
    beta2 = 0.999

    # gan mode (MSE loss) and loss weights
    # gan_mode = "lsgan"
    lambda_dict = {"GAN": 1.0, "l1": 100.0, "perc": 0.08}
    lambda_dict_pre = {"l1": 1.0, "perc": 0.0}

    # Normal training
    epochs = 30

    # Progressive training
    prog_img_sizes = [64, 128, 256]
    prog_batchsizes = [64, 64, 32]
    prog_lr = [5e-4, 1e-4, 1e-4]

    # tensorboard
    version_name = ""
