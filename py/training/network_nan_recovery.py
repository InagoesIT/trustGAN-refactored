import glob

import torch


class NetworkNaNRecovery:
    def __init__(self, networks_data, path_to_save, device, data_loader_train, modifier):
        self.networks_data = networks_data
        self.path_to_save = path_to_save
        self.device = device
        self.data_loader_train = data_loader_train
        self.modifier = modifier

    @torch.inference_mode()
    def recover_from_nan_target_model(self):
        """
        Recover from Nan in target_model
        """

        self.networks_data.target_model.eval()

        data = next(iter(self.data_loader_train))

        inputs = data[0][:1].to(self.device)
        inputs, _ = self.modifier((inputs, None))
        outputs = self.networks_data.target_model(inputs)

        if torch.any(torch.isnan(outputs)):
            print("\nWARNING: The Net gives NaN")
            nets = glob.glob("{}/nets/target_model-step-*.pth".format(self.path_to_save))
            nets = sorted(sorted(nets), key=len)
            idx = -1
            while torch.any(torch.isnan(outputs)):
                self.networks_data.target_model.load_state_dict(torch.load(nets[idx]))
                outputs = self.networks_data.target_model(inputs)
                idx -= 1
            print(f"WARNING: Recover a proper state there: {nets[idx + 1]}")

            self.networks_data.recovered_from_nan_target_model += 1
            print(f"WARNING: Recovered from NaN {self.networks_data.recovered_from_nan_target_model} times\n")
            self.networks_data.target_model_optimizer = torch.optim.AdamW(
                self.networks_data.target_model.hyperparameters(),
                weight_decay=0.05,
                lr=1.0e-3 / 2 ** self.networks_data.recovered_from_nan_target_model,
            )

        self.networks_data.target_model.run()

    @torch.inference_mode()
    def recover_from_nan_gan(self):
        """
        Recover from Nan in GAN
        """

        self.networks_data.gan.eval()

        data = self.modifier(next(iter(self.data_loader_train)))

        rand_inputs = torch.rand(data[0][:1].shape, device=self.device)
        gan_outputs = self.networks_data.gan(rand_inputs)

        if torch.any(torch.isnan(gan_outputs)):
            print("\nWARNING: The GAN gives NaN")
            nets = glob.glob(
                "{}/nets/gan-not-is_best-step-*.pth".format(self.path_to_save)
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
            self.networks_data.gan_optimizer = torch.optim.AdamW(
                self.networks_data.gan.hyperparameters(),
                weight_decay=0.05,
                lr=1.0e-3 / 2 ** self.networks_data.recovered_from_nan_gan,
            )
            self.networks_data.gan_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.networks_data.gan_optimizer,
                base_lr=1.0e-3 / 2 ** self.networks_data.recovered_from_nan_gan,
                max_lr=5.0e-3 / 2 ** self.networks_data.recovered_from_nan_gan,
                step_size_up=50,
                cycle_momentum=False,
            )

        self.networks_data.gan.run()
