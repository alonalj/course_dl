Please note that TransferLearning.py includes cifar100vgg.py due to inheritance
(hence you will see the import statement: import from cifar100vgg import *).
If TransferLearning.py is run from command line (from within the src folder) - this should not be an issue.
If this is run from the PyCharm IDE, the containing src folder must be marked as Source Root (right click -> Mark Directory as -> Source Root), or similarly modified on other IDEs.

For the fine-tuning section, we've included a download script to load our solution weights. In case this process
fails, please download them from the following link, and place them under the src folder:
https://drive.google.com/open?id=1GeT20aVI4X_DW9e665UA8T7O2B1_B-S5


