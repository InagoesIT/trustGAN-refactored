class TrainingParams:
    def __init__(
        self,
        nr_classes,
        batch_size=64,
        nr_epochs=2,
        nr_step_net_gan=1,
        nr_step_gan=1,
        nr_step_net_alone=1,
        proportion_net_alone=0.0,
        network_name="Net",
        nr_channels=None,
    ):
        self.nr_classes = nr_classes
        self.batch_size = batch_size
        self.nr_epochs = nr_epochs
        self.nr_step_net_gan = nr_step_net_gan
        self.nr_step_gan = nr_step_gan
        self.nr_step_net_alone = nr_step_net_alone
        self.proportion_net_alone = proportion_net_alone
        self.network_name = network_name
        self.nr_channels = nr_channels
