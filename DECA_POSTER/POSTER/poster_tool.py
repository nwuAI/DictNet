import warnings
import pickle
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from POSTER.models.PosterV2_7cls import *
from PIL import Image
from decalib.utils.config import cfg

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


class PosterTool:
    def __init__(self, config=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        # self.image_path = cfg.poster.image_path
        self.dict_path = cfg.poster.dict_path
        self.evaluate = cfg.poster.evaluate

    def predict_image(self, image_path):
        # Load the image
        image = Image.open(image_path)

        # Apply the necessary transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # load the model and checkpoint
        image = transform(image).unsqueeze(0)
        model = pyramid_trans_expr2(img_size=224, num_classes=7)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(self.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        image = image.to(device)

        with torch.no_grad():
            output = model(image)

        _, predicted_class = torch.max(output, 1)

        return predicted_class.item()

    def read_dictionary(self):
        with open(self.dict_path, 'rb') as f:
            dict_exp, dict_pose = pickle.load(f)

        return dict_exp, dict_pose

    def predict_images2(self, images):
        model = pyramid_trans_expr2(img_size=224, num_classes=7)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(self.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)

        _, predicted_classes = torch.max(outputs, 1)

        return predicted_classes.tolist()


# poster_tool = PosterTool(cfg)
# image_path = './TestSamples/examples/10405424_1_crop.jpg'
# exp_class = poster_tool.predict_image(image_path)

