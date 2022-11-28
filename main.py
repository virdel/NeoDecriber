import os
import torch
import shutil
from utils import *
def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device.type))

    img_path = os.path.abspath(parser_data.img_path)
    assert os.path.exists(img_path), "Image {} dose not exist.".format(img_path)

    save_dir =os.path.abspath(parser_data.output_dir)
    ##print(save_dir)

    outcome_txt_path = os.path.join(save_dir,"result.txt")
    ## 写入图像文件路径信息
    print("****************")
    print("Writing image information...")
    with open(outcome_txt_path,"w",encoding="utf-8") as f:
        f.write("image path: "+img_path)
        f.write("\n")
    print("Done")


    # 整体风格分类
    print("——————————————————————————————————————————")
    print("Holistic style classification:")
    status=classification(img_path, device, outcome_txt_path, r".\source\holistic_style")

    if status:

        print("——————————————————————————————————————————")
        print("Imitation element combo detection:")
        element_detection(img_path, device, save_dir,outcome_txt_path,r".\source\imitation",1,parser_data.line_thickness,parser_data.font_size)

        print("——————————————————————————————————————————")
        print("Key element detection:")
        infos=element_detection(img_path, device, save_dir,outcome_txt_path,r".\source\key_element",2,parser_data.line_thickness,parser_data.font_size)

        print("——————————————————————————————————————————")
        print("Arch element detection:")
        # for info  in infos:
        #     if info[0]=="arcade":
        #         element_detection(info[1], device, save_dir,outcome_txt_path,r".\source\arch",3,parser_data.line_thickness,parser_data.font_size)
        element_detection(img_path, device, save_dir, outcome_txt_path, r".\source\arch", 3, parser_data.line_thickness,
                          parser_data.font_size)
        print("——————————————————————————————————————————")
        print("Column combination detection:")
        # for info  in infos:
        #     if info[0]=="colonnade":
        #         element_detection(info[1], device, save_dir,outcome_txt_path,r".\source\column_combination",4,parser_data.line_thickness,parser_data.font_size)
        element_detection(img_path, device, save_dir, outcome_txt_path, r".\source\column_combination", 4,
                          parser_data.line_thickness, parser_data.font_size)


        print("——————————————————————————————————————————")
        print("Single column detection:")
        # for info in infos:
        #     if info[0] == "colonnade":
        #         element_detection(info[1], device, save_dir, outcome_txt_path, r".\source\single_column", 5,parser_data.line_thickness,parser_data.font_size)
        element_detection(img_path, device, save_dir, outcome_txt_path, r".\source\single_column", 5,
                          parser_data.line_thickness, parser_data.font_size)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(VOCdevkit)

    parser.add_argument('--img-path', default='./images/北卡教堂山.webp', help='dataset')

    # 文件保存地址
    parser.add_argument('--output-dir', default='./北卡教堂山', help='path where to save')

    parser.add_argument('--font_size', default=10, help='font size')


    parser.add_argument('--line_thickness', default=4, help='bounding box line_thickness')

    args = parser.parse_args()
    print(args)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    main(args)
    print("\n\n\n****************************")
    print("Detection complete!")
    print("****************************")