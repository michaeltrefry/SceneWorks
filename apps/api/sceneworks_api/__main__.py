from .settings import get_settings
from .server import run


if __name__ == "__main__":
    run(get_settings())
