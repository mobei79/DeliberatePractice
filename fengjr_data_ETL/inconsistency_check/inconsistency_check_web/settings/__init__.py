import os
import yaml

env = os.getenv('PYTHON_PROJECT_ENV')
data_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "config.yml"
config = dict()
with open(data_path, 'r') as f:
    y = yaml.load(f)
    if env == 'prod':
        config = y['prod']
    else:
        config = y['test']
