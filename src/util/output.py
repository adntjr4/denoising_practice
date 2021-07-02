import os


class Output:
    def __init__(self, session_name:str, dir_list:list):
        self.output_folder = "./output"

        # init session
        self.session_name = session_name
        os.makedirs(os.path.join(self.output_folder, self.session_name), exist_ok=True)

        # mkdir
        for directory in dir_list:
            self.make_dir(directory)

    def make_dir(self, dir_name):
        os.makedirs(os.path.join(self.output_folder, self.session_name, dir_name), exist_ok=True)

    def get_dir(self, dir_name):
        # return ./output/<session_name>/dir_name
        return os.path.join(self.output_folder, self.session_name, dir_name)


if __name__ == "__main__":
    op = Output("test", ["t1", "t2"])
    