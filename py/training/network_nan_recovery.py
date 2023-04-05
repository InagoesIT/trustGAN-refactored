import glob

import torch


class NetworkNaNRecovery:
    def __init__(self, networks_data, path_to_save, device, data_loaders, modifier):
        self.networks_data = networks_data
        self.path_to_save = path_to_save
        self.device = device
        self.data_loaders = data_loaders
        self.modifier = modifier

    @torch.inference_mode()
    def recover_from_nan_net(self):
        """
        Recover from Nan in net
        """

        self.networks_data.net.eval()

        data = next(iter(self.data_loaders.train))

        inputs = data[0][:1].to(self.device)
        inputs, _ = self.modifier((inputs, None))
        outputs = self.networks_data.net(inputs)

        if torch.any(torch.isnan(outputs)):
            print("\nWARNING: The Net gives NaN")
            nets = glob.glob("{}/nets/net-step-*.pth".format(self.path_to_save))
            nets = sorted(sorted(nets), key=len)
            idx = -1
            while torch.any(torch.isnan(outputs)):
                self.networks_data.net.load_state_dict(torch.load(nets[idx]))
                outputs = self.networks_data.net(inputs)
                idx -= 1
            print(f"WARNING: Recover a proper state there: {nets[idx + 1]}")

            self.networks_data.recovered_from_nan_net += 1
            print(f"WARNING: Recovered from NaN {self.networks_data.recovered_from_nan_net} times\n")
            self.networks_data.net_optim = torch.optim.AdamW(
                self.networks_data.net.parameters(),
                weight_decay=0.05,
                lr=1.0e-3 / 2 ** self.networks_data.recovered_from_nan_net,
            )

        self.networks_data.net.train()

    @torch.inference_mode()
    def recover_from_nan_gan(self):
        """
        Recover from Nan in GAN
        """

        self.networks_data.gan.eval()

        data = self.modifier(next(iter(self.data_loaders.train)))

        rand_inputs = torch.rand(data[0][:1].shape, device=self.device)
        gan_outputs = self.networks_data.gan(rand_inputs)

        if torch.any(torch.isnan(gan_outputs)):
            print("\nWARNING: The GAN gives NaN")
            nets = glob.glob(
                "{}/nets/gan-not-best-step-*.pth".format(self.path_to_save)
            )
            nets = sorted(sorted(nets), key=len)
            idx = -1
            while torch.any(torch.isnan(gan_outputs)):
                self.networks_data.gan.load_state_dict(torch.load(nets[idx]))
                gan_outputs = self.networks_data.gan(rand_inputs)
                idx -= 1
            print(f"WARNING: Recover a proper state there: {nets[idx]}")

            self.networks_data.recovered_from_nan_gan += 1
            print(f"WARNING: Recovered from NaN {self.networks_data.recovered_from_nan_gan} times\n")
            self.networks_data.gan_optim = torch.optim.AdamW(
                self.networks_data.gan.parameters(),
                weight_decay=0.05,
                lr=1.0e-3 / 2 ** self.networks_data.recovered_from_nan_gan,
            )
            self.networks_data.gan_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.networks_data.gan_optim,
                base_lr=1.0e-3 / 2 ** self.networks_data.recovered_from_nan_gan,
                max_lr=5.0e-3 / 2 ** self.networks_data.recovered_from_nan_gan,
                step_size_up=50,
                cycle_momentum=False,
            )

        self.networks_data.gan.train()
