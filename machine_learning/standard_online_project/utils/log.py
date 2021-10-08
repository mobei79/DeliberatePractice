
import os
import logging.config

import yaml
cur_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep


def setup_logging(
    default_path=cur_dir+'logging.yaml',
    env_key='PYTHON_PROJECT_ENV'
):
    """Setup logging configuration

    """
    path = default_path
    with open(path, 'rt') as f:
        config = yaml.safe_load(f.read())
    # generate log path
    for k,v in config.get('handlers').items():
        filename = v.get("filename")
        if filename:
            file_dir = os.path.dirname(filename)
            if not os.path.exists(file_dir):
                os.mkdir(file_dir)
    logging.config.dictConfig(config)
    value = os.getenv(env_key, None)
    if value == "prod":
        return logging.getLogger('prod_log')
    return logging.getLogger('test_log')


logger = setup_logging()