import yaml, os

class ConfigParser:
    def __init__(self, args):
        # load model configuration
        cfg_file = os.path.join('conf', args.config+'.yaml')
        with open(cfg_file) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # load argument
        for arg in args.__dict__:
            self.config[arg] = args.__dict__[arg]

    def __getitem__(self, name):
        return self.config[name]

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-d', '--device', default=None, type=str)
    args.add_argument('-r', '--resume', action='store_true')
    
    args = args.parse_args()

    args.config = "./conf/resnet_cfg.yaml"

    cp = ConfigParser(args)
