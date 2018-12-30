Please note that TransferLearning.py includes cifar100vgg.py due to inheritance
(hence you will see the import statement: import from cifar100vgg import *).
In order for this import to work in the PyCharm IDE, the containing folder must be marked as Source Root
(right click -> Mark Directory as -> Source Root), or modified otherwise. This should not be an issue if run
from command line.
