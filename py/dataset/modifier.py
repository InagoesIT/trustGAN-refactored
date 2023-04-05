import torch


class Modifier:
    def __init__(self, nr_channels):
        self.nr_channels = nr_channels

    def convert_to_nr_channels(self, x):
        # select a channel randomly
        if (
            (self.nr_channels is not None)
            and (self.nr_channels == 1)
            and (x.shape[1] > 1)
        ):
            idx = torch.randint(0, x.shape[1], (1,))
            x = x[:, idx: idx + 1]
        # duplicate a channel
        elif (
            (self.nr_channels is not None)
            and (self.nr_channels > 1)
            and (x.shape[1] == 1)
        ):
            x = torch.cat([x for _ in range(self.nr_channels)], dim=1)

        return x
    
    @staticmethod
    def min_max_norm(x):
        dims = tuple(range(1, x.ndim))
    
        min_value = torch.amin(x, dim=dims, keepdim=True)
        max_value = torch.amax(x, dim=dims, keepdim=True)
    
        interval_size = max_value - min_value
        interval_size = interval_size + (interval_size == 0.0)
    
        new_x = (x - min_value) / interval_size
    
        return new_x
    
    @staticmethod
    def tanh_centered(x):
        x = 2.0 * x - 1.0
        return x

    def __call__(self, input):
        x = input[0]
        if len(x) == 1:
            y = None
        else:
            y = input[1]

        x = self.convert_to_nr_channels(x)
        x = self.min_max_norm(x)
        x = self.tanh_centered(x)

        return x, y
