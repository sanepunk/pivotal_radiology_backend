# import torch
# from torchvision import transforms
# from PIL import Image
# import os

# class TBModel:
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = None
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5], std=[0.5])
#         ])
#         self.load_model()

#     def load_model(self):
#         try:
#             # Get the directory where the script is located
#             current_dir = os.path.dirname(os.path.abspath(__file__))
#             model_path = os.path.join(current_dir, '..', '..', 'tb_model.pth')
            
#             # Load the model
#             self.model = torch.load(model_path, weights_only=False)
#             self.model.eval()
#             print("TB model loaded successfully")
#         except Exception as e:
#             print(f"Error loading TB model: {str(e)}")
#             raise

#     def predict(self, image_path):
#         try:
#             # Load and preprocess the image
#             img = Image.open(image_path)
#             if img.mode == 'L':  # Grayscale
#                 img = img.convert('RGB')
#                 print(f"Converted grayscale image to RGB")
#             elif img.mode == 'RGBA':  # RGBA
#                 img = img.convert('RGB')
#                 print(f"Converted RGBA image to RGB")
#             elif img.mode != 'RGB':
#                 img = img.convert('RGB')
#                 print(f"Converted {img.mode} image to RGB")
#             input_tensor = self.transform(img).unsqueeze(0).to(self.device)

#             # Get prediction
#             with torch.no_grad():
#                 output = self.model(input_tensor)
#                 probabilities = torch.softmax(output, dim=1)
#                 print(probabilities)
#                 prediction = torch.argmax(probabilities, dim=1).item()
#                 confidence = probabilities[0][prediction].item()

#             return {
#                 "prediction": prediction,  # 0 for negative, 1 for positive
#                 "confidence": round(confidence, 2),
#                 "result": "TB Positive" if prediction == 1 else "TB Negative"
#             }
#         except Exception as e:
#             print(f"Error making prediction: {str(e)}")
#             raise

# # Create a singleton instance
# tb_model = TBModel() 