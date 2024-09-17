import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Load the Stable Diffusion model (ensure you're using GPU if available for better performance)
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Move model to GPU if available
    return pipe

# Initialize the model
model = load_model()

def generate_image(prompt: str) -> Image.Image:
    with torch.no_grad():
        # Generate image using the correct output key
        image = model(prompt).images[0]
    return image

# Streamlit UI
st.title("Image Generation with Stable Diffusion")
st.write("Enter a text prompt to generate an image:")

prompt = st.text_input("Prompt", "")

if st.button("Generate Image"):
    if prompt:
        st.write("Generating image...")
        with st.spinner("Please wait..."):
            image = generate_image(prompt)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.image(byte_im, caption="Generated Image", use_column_width=True)
    else:
        st.error("Please enter a prompt.")
