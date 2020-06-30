import fasttext
from config import parse_config
from data_loader import DataBatchIterator


def main():
    # 读配置文件
    config = parse_config()
    # 载入测试集合
    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        # batch_size=config.batch_size)
    )
    # test_data.set_vocab(vocab)
    test_data.load()
    print(test_data.vocab)

    train_data = DataBatchIterator(
        config=config,
        is_train=True,
        dataset="train",
        batch_size=config.batch_size,
        shuffle=True)
    train_data.load()

    model = fasttext.skipgram(train_data, 'model')
    classifier = fasttext.supervised(test_data, 'model')
