from torchvision import transforms
from PIL import Image
import torch,os,json
from model import *
from draw_box_utils import draw_box, get_element_images
import numpy as np

def IoU(box1, box2):
    """
    :param box1: list in format [xmin1, ymin1, xmax1, ymax1]
    :param box2:  list in format [xmin2, ymin2, xamx2, ymax2]
    :return:    returns IoU ratio (intersection over union) of two boxes
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    intersection = x_overlap * y_overlap
    union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection
    return float(intersection) / union
def IoM(box1, box2):
    """
    :param box1: list in format [xmin1, ymin1, xmax1, ymax1]
    :param box2:  list in format [xmin2, ymin2, xamx2, ymax2]
    :return:    returns IoM ratio (intersection over min-area) of two boxes
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    intersection = x_overlap * y_overlap
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    min_area = min(box1_area, box2_area)
    return float(intersection) / min_area


def classification(img_path,device,txt_path,data_root):

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img = Image.open(img_path)
    # plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)


    # 读取类别信息
    json_path = os.path.join(data_root,'class_indices.json')
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    num_classes=len(class_indict)

    #载入模型
    model = resnet34(num_classes=num_classes).to(device)
    # load model weights
    weights_path = os.path.join(data_root,"resNet34.pth")
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))


    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # plt.title(print_res)
    # 每种预测类别的概率
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                predict[i].numpy()))
    # plt.show()
    #写入结果信息

    with open(txt_path, "a", encoding="utf-8") as f:
        f.write("Holistic style classification：\n    ")
        f.write(print_res)
        f.write("\n    Probability of each category：\n")
        for i in range(len(predict)):
            f.write("       class: {:10}   prob: {:.3} \n".format(class_indict[str(i)],
                                              predict[i].numpy()))
    if class_indict[str(predict_cla)]=="Neoclassism":
        return 1
    else:
        return 0

def element_classification(img_path,device,data_root):

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img = Image.open(img_path)
    # plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    # read class_indict
    json_path = os.path.join(data_root,'class_indices.json')
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    num_classes=len(class_indict)

    model = resnet34(num_classes=num_classes).to(device)
    # load model weights
    weights_path = os.path.join(data_root,"resNet34.pth")
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "--->class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                               predict[predict_cla].numpy())
    Baroque_list=["Broken/semicircular","Polygonal"]
    Gothic_list=["Pointed"]
    Romanesque_list=["Multilayer"]
    if class_indict[str(predict_cla)] in Baroque_list :
        print_res+="   implicit style: Baroque"
    elif class_indict[str(predict_cla)] in Gothic_list:
        print_res += "   implicit style: Gothic"
    elif class_indict[str(predict_cla)] in Romanesque_list:
        print_res += "   implicit style: Romanesque"

    return print_res


def element_detection(img_path,device,save_dir,txt_path,data_root,task,line_thickness,font_size):

    # return colonnade arcade info
    return_info = []


    thresh=0.5
    if task==1:
        task_string="imitation element combo"
        img_save_dir=os.path.join(save_dir,"element_combination")
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("imitation element combo:\n")
    elif task==2:
        ##关键元素检测
        task_string="key_element"
        img_save_dir=os.path.join(save_dir,"key_element")
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("key element detection :\n")
    elif task==3:
        ##关键元素检测
        task_string="arch_element"
        img_save_dir=os.path.join(save_dir,"arch_element")
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("arch element detection:\n")
        thresh=0.5
    elif task==4:
        ##关键元素检测
        task_string="column_combination"
        img_save_dir=os.path.join(save_dir,"column_combination")
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("column combination detection:\n")
        thresh=0.6
    elif task==5:
        ##关键元素检测
        task_string="single_column"
        img_save_dir=os.path.join(save_dir,"single_column")
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("single column detection:\n")
        thresh = 0.5

    label_json_path = os.path.join(data_root, 'my_source_data_class2num.json')
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)


    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()

    num_classes = len(class_dict)
    category_index = {v: k for k, v in class_dict.items()}
    # create model
    model = create_model(num_classes=num_classes + 1)


    # load train weights
    train_weights = os.path.join(data_root, "resNetFpn-model.pth")
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])

    original_img = Image.open(img_path)
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]

        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        predictions = model(img.to(device))[0]

        predict_boxes = predictions["boxes"].to("cpu").numpy().tolist()

        predict_classes = predictions["labels"].to("cpu").numpy().tolist()
        predict_scores = predictions["scores"].to("cpu").numpy().tolist()
        # print(predict_boxes)
        print(predict_classes)
        print(predict_scores)
        predict_boxes_list=[]
        predict_classes_list=[]
        predict_scores_list=[]

        pediment_box_list=[]
        pediment_classes_list =[]
        pediment_scores_list=[]
        pediment_area=[]
        pediment_index=[]



        for i in range(len(predict_scores)):
            if category_index[predict_classes[i]]=="pediment":
                pediment_index.append(i)
                pediment_box_list.append(predict_boxes[i])
                pediment_classes_list.append(predict_classes[i])
                pediment_scores_list.append(predict_scores[i])
                pediment_area.append((predict_boxes[i][2]-predict_boxes[i][0])*(predict_boxes[i][3]-predict_boxes[i][1]))
            else:
                predict_boxes_list.append(predict_boxes[i])
                predict_classes_list.append(predict_classes[i])
                predict_scores_list.append(predict_scores[i])


        if len(pediment_area)!=0:
            max_pediment_index=np.array(pediment_area).argmax()
            predict_scores_list.append(pediment_scores_list[max_pediment_index])
            predict_boxes_list.append(pediment_box_list[max_pediment_index])
            predict_classes_list.append(pediment_classes_list[max_pediment_index])

        predict_scores=np.array(predict_scores_list)
        predict_boxes = np.array(predict_boxes_list)
        predict_classes = np.array(predict_classes_list,dtype=np.int64)


        crop_imgs,string_list,labels,scores= get_element_images(original_img,
                                          predict_boxes,
                                          predict_classes,
                                          predict_scores,
                                          category_index,
                                          thresh=thresh,
                                          )

        if len(labels)==0:
            print("No object detected!")
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write("   No object detected!\n")
        else:
            # 在原图上绘制边界框
            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=thresh,
                     line_thickness=line_thickness,
                     font_size=font_size)

            original_img.save(os.path.join(save_dir, task_string + ".jpg"))


            for img,string, label,score in zip(crop_imgs,string_list, labels,scores):
                img_path=os.path.join(img_save_dir, string + ".jpg")
                img.save(img_path)
                print_res=""
                if label=="pediment":
                    print_res=element_classification(img_path,device,"./source/pediment")
                elif label=="tower":
                    print_res = element_classification(img_path, device, "./source/tower")
                elif label=="dome":
                    print_res = element_classification(img_path, device, "./source/dome")
                elif label=="arch":
                    print_res = element_classification(img_path, device, "./source/Arch_classification")
                elif label=="single_column":
                    print_res = element_classification(img_path, device, "./source/column_order")

                if label=="colonnade" or label=="arcade":
                    return_info.append([label,img_path])

                with open(txt_path, "a", encoding="utf-8") as f:
                    f.write("    ")
                    f.write(string)
                    f.write("    ")
                    f.write("score:"+score)
                    f.write("    ")
                    f.write(print_res)
                    f.write("    ")
                    f.write("--->img_path:"+img_path)
                    f.write("\n")

    return return_info

