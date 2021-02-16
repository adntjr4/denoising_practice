import yaml, os

cfg_file = os.path.join('dataset/prep/prep-SIDD_Medium_sRGB-cut512-ov128', 'info_GT', '0-0-ov_br.yml')
with open(cfg_file) as f:
    print(yaml.load(f, Loader=yaml.FullLoader))