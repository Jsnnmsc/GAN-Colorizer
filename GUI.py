from PIL import Image
import tempfile
import os
import torch
import gc
from time import sleep
import random

# streamlit components and plugins
import streamlit as st
from streamlit_image_comparison import image_comparison

# colorization components
from colorization.src.config import cfg
from colorization.src.inference import Colorizer


# init model and save in cache. So we don't have to load it every time.
@st.cache_resource
def init_colorize_model(model_path):
    CLRZ = Colorizer(model_path=model_path, cfg=cfg())
    return CLRZ


st.title("Image Colorization")

with st.sidebar:

    st.write("Upload your gray-scale image")

    uploaded_file = st.file_uploader(
        "Please choose a image", type=["jpg", "jpeg", "png"]
    )

    render_factor = st.slider(
        "Render Factor",
        min_value=10,
        max_value=60,
        value=30,
        step=1,
    )

    no_rf = st.checkbox("Disable Render Factor", value=False)

    clr_model_path = "colorization/src/models/gen_110ep.pt"

    gc_status = st.button("Clean Memory")
    if gc_status:
        amt = gc.collect()
        text = st.info(f"Memory Cleaned:{amt}")
        sleep(2)
        text.empty()

    run_status = st.button("Run!")

colorizer = init_colorize_model(clr_model_path)

if uploaded_file and os.path.exists(clr_model_path):
    # Save the upload image to temporary folder
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Start the processes
    if run_status:

        if not os.path.exists("temp"):
            os.makedirs("temp")

        with st.spinner("Colorizing..."):
            result_img = colorizer.infer(
                input_image=tmp_path,
                render_factor=render_factor,
                post_process=True,
                comparison=False,
                save_path="temp/colorized.jpg",
                no_rf=no_rf,
            )

        orig_image = Image.open(tmp_path).convert("RGB")

        st.success("Success!")

        image_comparison(
            img1=orig_image,
            img2=result_img,
            label1="Original",
            label2="Colorized",
            # width=500,
        )

        save_path = "temp/result_final.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_img.save(save_path)
        st.info(f"Saved to: {save_path}")
else:
    st.warning("Please upload your gray-scale image and run!")
