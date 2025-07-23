import os
import json

class DatasetBuilder:
    def __init__(self, source_directory, output_file):
        self.source_directory = source_directory
        self.output_file = output_file
        self.dataset = []

    def collect_data(self):
        for filename in os.listdir(self.source_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(self.source_directory, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    self.dataset.append(data)

    def save_dataset(self):
        with open(self.output_file, 'w') as file:
            json.dump(self.dataset, file, indent=4)

    def build(self):
        self.collect_data()
        self.save_dataset()

if __name__ == "__main__":
    source_dir = "path/to/source/directory"  # Update with the actual path
    output_file = "path/to/output/dataset.json"  # Update with the desired output path
    builder = DatasetBuilder(source_dir, output_file)
    builder.build()