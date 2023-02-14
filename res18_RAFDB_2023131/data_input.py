import torch
from torchvision import transforms, datasets, utils
import json


# 数据预处理，定义data_transform这个字典
data_transform = {
    "train": transforms.Compose([transforms.Resize((224, 224)),  # 随机裁剪，裁剪到224*224
                                 transforms.RandomHorizontalFlip(),  # 水平方向随机翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

def train_data(root_dir,batch_size):
    train_dataset = datasets.ImageFolder(root=root_dir + "train",transform=data_transform["train"])
    train_data_num = len(train_dataset)
    # print(train_dataset.classes)
    flower_list = train_dataset.class_to_idx  # 获取分类的名称所对应的索引，即{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    cla_dict = dict((val, key) for key, val in flower_list.items())  # 遍历获得的字典，将key和value反过来，即key变为0，val变为daisy
    # 将key和value反过来的目的是，预测之后返回的索引可以直接通过字典得到所属类别
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:  # 保存入json文件
        json_file.write(json_str)
    # class_sample_count = [23327, 3665, 3677, 5979, 131222, 71777, 23893, 13535] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    # weights = 1 / torch.Tensor(class_sample_count)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size) # 注意这里的weights应为所有样本的权重序列，其长度为所有样本长度。
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,shuffle=True,
                                            num_workers=8)
    return train_loader,train_data_num

def val_data(root_dir,batch_size):
    validate_dataset = datasets.ImageFolder(root=root_dir + "test",transform=data_transform["val"])
    val_data_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=8)
    return validate_loader,val_data_num

# if __name__== "__main__":
#     data_dir = '../../../datasets/flower/'
#     batch_size = 32
#     validate_loader,val_data_num = train_data(data_dir,batch_size)
#     print(val_data_num)
#     # val_num = 0
#     # for data_test in validate_loader:
#     #         test_images, test_labels = data_test
#     #         test_labels_len = len(test_labels)
#     #         val_num = val_num + test_labels_len
#     #         print(test_labels_len)
#  