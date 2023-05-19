import os


class Paths:
    def __init__(self,
                 root_folder,
                 dataset,
                 load_target_model=None,
                 load_gan=None):
        self.root_folder = root_folder
        self.dataset = dataset
        self.load_target_model = load_target_model
        self.load_gan = load_gan
        self.process_root_folder()
        
    def process_root_folder(self):
        if self.root_folder is not None:
            if not os.path.isdir(self.root_folder):
                os.mkdir(self.root_folder)
            for folder in ["plots", "nets", "perfs-plots", "gifs"]:
                if not os.path.isdir(os.path.join(self.root_folder, folder)):
                    os.mkdir(os.path.join(self.root_folder, folder))
                else:
                    print("\nWARNING\n Folder exists")
