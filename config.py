import torch

class Config(object):

    def __init__(self):

        self.train_path = 'clue/train.json'
        self.dev_path = 'clue/dev.json'
        self.test_path = 'clue/test.json'
        
        self.bert_path = './bert-base-chinese'

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.bert_hidden_size = 768
        self.lstm_hidden_size = 1024
        self.lstm_layer_num = 2
        self.lstm_is_bi = True
        self.lstm_dropout = 0.1
        self.tag_num = 32

        self.dropout = 0.1
        self.lr = 5e-5

        self.epochs = 20
        
    def __str__(self):
        return 
    """
    (   train_path = 'clue/train.json'
        dev_path = 'clue/dev.json'
        test_path = 'clue/test.json'

        bert_path = './bert-base-chinese'

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size = 24
        bert_hidden_size = 768
        lstm_hidden_size = 1024
        lstm_layer_num = 2
        lstm_dropout = 0.5
        lstm_is_bi = True
        tag_num = 33

        dropout = 0.1
        lr = 5e-5

        epochs = 20
    )
    """

