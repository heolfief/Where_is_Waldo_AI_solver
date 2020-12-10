"""waldo_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import pathlib

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
            'image': tfds.features.Image(shape=(64, 64, 3)),
            'label': tfds.features.ClassLabel(names=['notwaldo', 'waldo']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://github.com/heolfief/Where_is_Waldo_AI_solver',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(waldo_dataset): Downloads the data and defines the splits
    #extracted_path = "output-DS-images" #dl_manager.download_and_extract('https://github.com/vc1492a/Hey-Waldo/archive/master.zip')

    data_dir ="output-DS-images/"


    # TODO(waldo_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path=pathlib.Path(data_dir)),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(waldo_dataset): Yields (key, example) tuples from the dataset
    for img_path in path.glob('*/*.png'):
      yield img_path.name, {
          'image': img_path,
          'label': 'waldo' if img_path.name.startswith('waldo') else 'notwaldo',
      }
