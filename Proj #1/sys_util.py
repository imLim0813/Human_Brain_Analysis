import os


def makedir(name):
    if not os.path.exists('./Data/RL_model'):
        os.makedirs('./Data/RL_model')

    directory = './Data/RL_model/{}'.format(name)
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory + '/actor')
        os.makedirs(directory + '/critic')

    print('=' * 50)
    print('Directory has been made.')
    print('Dir path : {}'.format(directory))
    print('=' * 50)

    return directory
