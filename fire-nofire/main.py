
#################################
# Main File
#################################

# ############# import libraries
# General Modules

# Customized Modules
from config import Mode

from training import train_keras
from classification import classify

if __name__ == "__main__":
    if Mode == 'Training':
        train_keras()
    elif Mode == 'Classification':
        classify()
    else:
        print("Mode is not correct")
