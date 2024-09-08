import warnings
import argparse
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import datetime
from models.PosterV2_7cls import *
from PIL import Image

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r'/home/Dataset/RAF')
parser.add_argument('-e', '--evaluate', default=None, type=str, help='evaluate model on test set')
args = parser.parse_args()


def predict_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Apply the necessary transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    # 加载检查点
    checkpoint = torch.load(args.evaluate)

    # # 删除checkpoint中多余的键
    # state_dict = checkpoint['state_dict']
    # new_state_dict = {k: v for k, v in state_dict.items() if 'RecorderMeter1' not in k.lower()}
    # torch.save({'state_dict': new_state_dict}, args.evaluate)

    # Load the model and checkpoint
    model = pyramid_trans_expr2(img_size=224, num_classes=7)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.evaluate)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()


if __name__ == '__main__':
    image_path = '/mnt/d/Code/PythonCode/POSTER_V2-main/data/RAF-DB/valid/0/test_0008.jpg'
    predicted_class = predict_image(image_path)
    print('Predicted class:', predicted_class)
