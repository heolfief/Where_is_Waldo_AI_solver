"""waldo_dataset dataset."""

import tensorflow_datasets as tfds

# TODO(waldo_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Where is Waldo dataset
"""

# TODO(waldo_dataset): BibTeX citation
_CITATION = """
"""


class WaldoDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for waldo_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(waldo_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(names=['notwaldo', 'waldo']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://github.com/vc1492a/Hey-Waldo',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(waldo_dataset): Downloads the data and defines the splits
    extracted_path = dl_manager.download_and_extract('https://github.com/vc1492a/Hey-Waldo/archive/master.zip')

    # TODO(waldo_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path=extracted_path / 'Hey-Waldo-master/64/notwaldo'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(waldo_dataset): Yields (key, example) tuples from the dataset
    for img_path in path.glob('*.jpg'):
      yield img_path.name, {
          'image': img_path,
          'label': 'waldo' if img_path.name.startswith('1_') else 'notwaldo',
      }