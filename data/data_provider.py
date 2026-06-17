from torch.utils.data import DataLoader

from data.loaders import M5_loader

def get_provider(args, split):
    if args.data == "M5":
        return M5_provider(args, split)

def M5_provider(args, split):
    data_set = M5_loader.Loader(args, split)

    data_loader = DataLoader(
        data_set,
        batch_size = args.batch_size,
        shuffle = False if split.lower() == "test" else True,
        num_workers = args.num_workers,
        pin_memory = True,
    )

    return data_set, data_loader