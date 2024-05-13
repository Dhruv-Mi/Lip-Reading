import streamlit as st
import imageio
import os
from moviepy.editor import VideoFileClip

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Set up the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')
# Generating a list of options or videos
video_file = st.file_uploader("Upload a video", type=["mpg", "avi", "mov"])
# Generate two columns
col1, col2 = st.columns(2)

if video_file is not None:
    st.write("You selected a video!")
    # Rendering the video
    with col1:
        output_mp4 = "output_video.mp4"
        file_path = os.path.join('..', 'data', 's1', video_file.name)
        video_clip = VideoFileClip(file_path)
        video_clip.write_videofile(output_mp4, codec="libx264")
        video_clip.close()
        st.video(output_mp4)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
