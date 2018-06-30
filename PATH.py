import os

# Path to data depending of USER variable
DATAPATH = '/home/Julian/' if os.getenv('USER') == 'bemootzer' else ( '/home/ubuntu/DATA/' if os.getenv('USER') == 'ubuntu' else '/home/Julian/')
    
