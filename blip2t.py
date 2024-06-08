from PIL import Image
from utils.blip2t import BLIP2T

blip2t = BLIP2T("Salesforce/blip-itm-large-coco", "cpu")

prompt = "photo of a dog"
image = Image.open("data/dog/00.jpg")

score = blip2t.text_similarity(prompt, image)[0]
print(score)
