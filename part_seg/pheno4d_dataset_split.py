from sklearn.model_selection import train_test_split
from pheno4d_dataset_all_normal import Pheno4dNormalDataset

class Pheno4dDataset():
  def __init__(self) -> None:
    self.train_dataset = None
    self.test_dataset = None
    self.train_i = None
    self.test_i = None

  def pheno4d_train_test_split(self, npoints=10000, category='maize', train_sample=1.0, random_state=42):
    models = [1,2,3,4,5,6,7]
    self.train_i, self.test_i = train_test_split(models, test_size=2, random_state=random_state)
    self.train_dataset = Pheno4dNormalDataset(category=category, indices=self.train_i, train_samples=train_sample)
    self.test_dataset = Pheno4dNormalDataset(category=category, indices=self.test_i)
    print(self.get_log_stats())
    return self.train_dataset, self.test_dataset

  def get_log_stats(self):
    log = (
    'Train Samples: %d' % len(self.train_dataset) +
    '\nTrain Models: %s' % str(self.train_i) +
    '\nTest Samples: %d' % len(self.test_dataset) +
    '\nTest Models: %s' % str(self.test_i)
    )
    return log

if __name__ == '__main__':
  pheno4d_dataset = Pheno4dDataset()
  pheno4d_dataset.pheno4d_train_test_split()
  #pheno4d_dataset.print_stats()