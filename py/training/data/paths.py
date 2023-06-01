import os


class Paths:
    def __init__(self,
                 root_folder,
                 dataset,
                 load_target_model=None,
                 load_gan=None,
                 path_to_performances=None):
        self.root_folder = root_folder
        self.dataset = dataset
        self.load_target_model = load_target_model
        self.load_gan = load_gan
        self.path_to_performances = path_to_performances
        Paths.process_root_folder(root_folder)
    
    @staticmethod
    def process_root_folder(root_folder):
        if root_folder is not None:
            if not os.path.isdir(root_folder):
                os.mkdir(root_folder)
            for folder in ["plots", "networks", "performance-plots", "gifs"]:
                if not os.path.isdir(os.path.join(root_folder, folder)):
                    os.mkdir(os.path.join(root_folder, folder))
                else:
                    print("\nWARNING\n Folder exists")
