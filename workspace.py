import os
import auralflow


def main():
    auralflow.utils.pull_config_template(os.getcwd() + "/my_model")

if __name__ == "__main__":
    main()

