from method.dreambooth.data.base import DatasetBuilder


class FlickrAESDatasetBuilder(DatasetBuilder):
    def list_target_users(self):
        raise NotImplementedError("FlickrAES adapter removed in userpref-only refactor")

    def build_train_manifest(self, user_id, output_root):
        raise NotImplementedError("FlickrAES adapter removed in userpref-only refactor")

    def get_test_samples(self):
        raise NotImplementedError("FlickrAES adapter removed in userpref-only refactor")
