import clip
import torch
from PIL import Image


class Similarity:

    def __init__(self, model_name, device):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device)

    @torch.no_grad()
    def text_similarity(self, prompt, image):
        """
        Calculate text similarity between prompt and image

        Args:
            prompt: str
            image: PIL.Image

        Return
            score: float
        """
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize([prompt]).to(self.device)

        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(text_features, image_features.T).squeeze()

        score = similarity.detach().cpu().numpy()

        return score

    def image_similarity(self, source, generate):
        """
        Calculate image similarity between source image and generate image

        Args:
            prompt: PIL.Image
            image: PIL.Image

        Return
            score: float
        """
        source = self.preprocess(source).unsqueeze(0).to(self.device)
        generate = self.preprocess(generate).unsqueeze(0).to(self.device)

        image_features_source = self.model.encode_image(source)
        image_features_generate = self.model.encode_image(generate)

        image_features_source = image_features_source / image_features_source.norm(
            dim=-1, keepdim=True
        )
        image_features_generate = (
            image_features_generate / image_features_generate.norm(dim=-1, keepdim=True)
        )

        similarity = torch.matmul(
            image_features_source, image_features_generate.T
        ).squeeze()

        score = similarity.detach().cpu().numpy()

        return score
