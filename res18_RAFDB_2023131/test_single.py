import torch
# from model import AlexNet
from PIL import Image
from torchvision import transforms
import json,os
import data_input
# import input_data
from model3 import ResNet18_ARM___RAF

# 标签，0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
# ferplus 

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



def get_model():
    model = ResNet18_ARM___RAF()
    # load model weights
    model_weight_path = "./pretrain_AlexNet3_vit_.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    return model


def get_inputs(img_path):
    image = Image.open(img_path)
    inputs = data_transform(image)
    inputs = torch.unsqueeze(inputs, dim=0)
    # print(type(input))
    return inputs


def predict(model, inputs):
    with torch.no_grad():
         # output = torch.squeeze(model(img))
        outputs = model(inputs)
        index = outputs.max(1).indices.item()
        # print(index)

    return index


# 预测一张图像
def main(model,img_path = './1.png'):
    inputs = get_inputs(img_path)
    index= predict(model, inputs)
    return index,class_indict[str(index)]


# test_dir = 'G:/dataset/RAFDB/basic/Image/test/'

def pre_write_txt(pred, file):
    for i in pred:
        f = open(file, 'a', encoding='utf-8_sig')
        f.write(str(i) + ',')
        f.close()
    print("-----------------预测结果已经写入文本文件--------------------")

# 测试整个测试集，通采用data loder测试
validate_loader, val_num =  data_input.val_data(root_dir='/datasets/rafdb/',batch_size=128)
def test_acc(model):
    acc = 0.0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_index = 0
    with torch.no_grad():
        for data_test in validate_loader:
            batch_index +=1
            test_images, test_labels = data_test
            test_labels_len = len(test_labels)    
            # print(test_labels_len)    
            outputs = model(test_images.to(device))
            # print(outputs)
            predict_y = outputs.max(1, keepdim=True)[1]
            # print(predict_y.shape)    
            # print(test_labels.shape) 
            # # 为了保证标签的维度与网络输出的维度一样 方便后面的计算     
            test_labels = test_labels.view_as(predict_y)
            # print(test_labels.shape)
            # 这一步必须保证标签与网络输出的维度是一样的才可计算，也就导致必须.view_as(predict_y)
            batch_acc = (predict_y == test_labels.to(device)).sum().item()
            acc += batch_acc
            print("第 {} 个test batch acc:{:.4f}".format(batch_index,batch_acc/test_labels_len))
            
            # # 将预测结果和标签保存
            # cpu_output = predict_y.cpu()
            # cpu_targets = test_labels.cpu()
            # targets_numpy = [q.numpy() for q in cpu_targets]
            # output_numpy = [t.numpy() for t in cpu_output]
            # int_output = [int(i) for i in cpu_output]
            # int_targets = [int(j) for j in cpu_targets]
            # pre_write_txt(int_output, 'flower_pred.txt')
            # pre_write_txt(int_targets, 'flower_truelabel.txt')
            # print(acc)
            

        accurate_test = acc / val_num
    print("\n测试集一共 {} 图像, 共测试了{}个 batch, predict acc : {}".format(val_num,batch_index,accurate_test))

# # 测试某一个类别
# def class_acc():
#     q = 0
#     test_dir = 'G:/dataset/RAFDB/basic/Image/test/4/'
#     imgs = os.listdir(test_dir)
#     for img in imgs:
#         img_path = test_dir + img
#         index, true_label = main(img_path)
#         if index == 3:
#             q = q +1

#     print("\n predict acc : {}".format(q/len(imgs)))

def testset_acc_one_by_one(test_dir = '/datasets/rafdb/'):
    class_name = os.listdir(test_dir)

    # new_class_indict = dict((value,key) for key,value in class_indict.items())
    # print(class_indict)
    # print(new_class_indict)
    sum = 0
    model = get_model()
    for dir in class_name:
        right_num = 0
        # true_label = new_class_indict[dir]
        
        class_name = test_dir+dir+'/'
        num_dir = len(os.listdir(class_name))
        # print(class_name)
        # print(num_dir)
        # print(dir,true_label)
        for img in os.listdir(class_name):
            lable_num, true_label = main(model,class_name+img)
            if true_label == dir:
                right_num += 1
        sum = sum + right_num
        print(class_name,num_dir,right_num)
    print(sum)


if __name__ == '__main__':
    # 测试单张图像
    model = get_model()
    # lable_num, true_label = main(model,img_path = "../tulips.jpg")
    # print(lable_num, true_label)
    # class_acc()
    # test_acc(model)
    # testset_acc_one_by_one()

