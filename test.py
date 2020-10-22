from models.engine import load_engine
from utils.utils import read_config

if __name__ == "__main__":
    CONFIG_PATH = "d4pg-pytorch/configs/follow_rl.yml"
    print ("before read")
    config = read_config(CONFIG_PATH)

    print ("before load")
    engine = load_engine(config)
    engine.test()
