import torch
from config import parse_config
from data_loader import DataBatchIterator
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


def test_textcnn_model(model, test_data, config):
    model.eval()
    total_loss = 0
    test_data_iter = iter(test_data)
    corrects = 0
    for idx, batch in enumerate(test_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        outputs = model(batch.sent)
        # print("ground:")
        # print(ground_truth)
        # print("outputs:")
        # print(outputs)
        result = torch.max(outputs, 1)[1]
        # print("result:")
        # print(result)
        corrects += (result.view(ground_truth.size()).data ==
                     ground_truth.data).sum()
    size = 5000
    accuracy = 100.0 * corrects / size
    return accuracy, corrects, size


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
    # 加载textcnn模型
    model = torch.load('./results/model.pt')

    # 打印模型信息
    print(model)

    # 测试
    accuracy, corrects, size = test_textcnn_model(model, test_data, config)

    # 打印结果
    print(
        '\nEvaluation - acc: {:.4f}%({}/{}) \n'.format(accuracy, corrects, size))


if __name__ == "__main__":
    main()
