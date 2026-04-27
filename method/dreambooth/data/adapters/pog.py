from method.dreambooth.data.base import DatasetBuilder


class POGDatasetBuilder(DatasetBuilder):
    def list_target_users(self):
        raise NotImplementedError("POG adapter removed in userpref-only refactor")

    def build_train_manifest(self, user_id, output_root):
        raise NotImplementedError("POG adapter removed in userpref-only refactor")

    def get_test_samples(self):
        raise NotImplementedError("POG adapter removed in userpref-only refactor")
