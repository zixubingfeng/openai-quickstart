import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='langchain autogpt.')
        self.parser.add_argument('--config_file', type=str, default='config.yaml', help='Configuration file with model and API settings.')
        self.parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='choose llm to use.')

    def parse_arguments(self):
        args = self.parser.parse_args()
        return args
