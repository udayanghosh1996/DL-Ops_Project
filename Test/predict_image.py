from torchvision import transforms
from models.SimCLR import *

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(32)
                                ])


def image_prediction(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = Classifier()
    model = clf.load_model()
    model.eval()
    image_ = transform(image)
    output = model(image_)
    prob, obj = output.topk(10)
    if torch.cuda.is_available():
        prob = prob.cpu()
        obj = obj.cpu()
    pred = []
    for i in range(len(prob)):
        pred.append({'Prediction Probability': prob, 'Prediction Class': obj})
    return pred

