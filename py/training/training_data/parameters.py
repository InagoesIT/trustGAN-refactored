class Parameters:
    def __init__(
        self,
        nr_classes,
        batch_size=64,
        nr_epochs=2,
        nr_steps_target_model_on_gan=1,
        nr_steps_gan=1,
        nr_step_target_model_alone=1,
        proportion_target_model_alone=0.0,
        target_model_network_type="Net",
        nr_channels=None,
        k_fold=5,
        validation_interval=25
    ):
        self.nr_classes = nr_classes
        self.batch_size = batch_size
        self.total_epochs = nr_epochs
        self.nr_steps_target_model_gan = nr_steps_target_model_on_gan
        self.nr_steps_gan = nr_steps_gan
        self.nr_steps_target_model_alone = nr_step_target_model_alone
        self.proportion_target_model_alone = proportion_target_model_alone
        self.target_model_network_type = target_model_network_type
        self.nr_channels = nr_channels
        self.k_fold = k_fold
        self.validation_interval = validation_interval
