import os

# Path to data depending of USER variable
DATAPATH = '../../DATA/' if os.getenv('USER') == 'bemootzer' else ( '/home/ubuntu/DATA/' if os.getenv('USER') == 'ubuntu' else '/Users/Julian/Desktop/Dropbox/synthbeedata/')
    
