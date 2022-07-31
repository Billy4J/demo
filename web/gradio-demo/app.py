import gradio as gr
import cv2


def analyze_image(img):
    return cv2.imread(img)

title = "web测试展示界面"
description = ""
article = ""
examples = [[]]
css = ".output-image, .input-image, .image-preview {height: 600px !important}"

iface = gr.Interface(fn=analyze_image,
                     inputs=gr.inputs.Image(type="numpy", label="document image"),
                     outputs=gr.outputs.Image(type="numpy", label="annotated document"),
                     title=title,
                     description=description,
                     examples=examples,
                     article=article,
                     css=css,
                     enable_queue=True)
iface.launch(debug=True,server_name="0.0.0.0",server_port=9999)