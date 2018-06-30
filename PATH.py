import os

# Path to data depending of USER variable
DATAPATH = '/home/Julian/' if os.getenv('USER') == 'bemootzer' else ( '/home/ubuntu/DATA/' if os.get('USER') == 'ubuntu' else '/Users/Julian/Desktop/Dropbox/synthbeedata/')

