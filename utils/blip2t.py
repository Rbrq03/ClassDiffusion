import torch
from transformers import BlipProcessor, BlipForImageTextRetrieval


class BLIP2T:

    def __init__(self, model_name, device):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)

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
        inputs = self.processor(image, prompt, return_tensors="pt").to(
            self.device, torch.float16
        )
        scores = self.model(**inputs, use_itm_head=False)[0]

        if self.device == "cpu":
            scores = scores.detach().numpy()[0]
        else:
            scores = scores.detach().cpu().numpy()[0]

        return scores
