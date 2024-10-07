# © 2024 Numantic Solutions
# https://github.com/Numantic-NMoroney
# MIT License
#

import numpy as np
import colour
import matplotlib.pyplot as plt
import streamlit as st
import zipfile
from sklearn.neighbors import KNeighborsClassifier


def lightness_slice(min_, max_, n, lightness):
    x = np.linspace(min_, max_, n)
    y = np.linspace(min_, max_, n)

    aa, bb = np.meshgrid(x, y)

    ll = x = np.empty(aa.shape)
    ll.fill(lightness)

    return list(zip(ll.flatten(), aa.flatten(), bb.flatten()))


def in_srgb_gamut(labs):
    labs_in, srgbs_in = [], []
    for lab in labs:
        xyz = colour.Lab_to_XYZ(lab)
        srgb = colour.XYZ_to_sRGB(xyz)
        clipped = np.clip(srgb, 0, 1)
        if np.array_equal(srgb, clipped):
            labs_in.append(lab)
            srgbs_in.append(srgb)
    return np.array(labs_in), np.array(srgbs_in)

if 'lightness' not in st.session_state:
    st.session_state.lightness = 65
if 'steps' not in st.session_state:
    st.session_state.steps = 51
if 'k_value' not in st.session_state:
    st.session_state.k_value = 10
if 'show' not in st.session_state:
    st.session_state.show = False

path_data = "data/"
@st.cache_data
def read_data():
    name_zip = "ml_color-11_terms-min_670-rgbn.tsv.zip"
    lines = []
    with zipfile.ZipFile(path_data + name_zip) as archive :
        item = archive.read(name_zip[:-4])
        s = item.decode()
        lines = s.split('\n')

    rs, gs, bs, names = [], [], [], []
    for line in lines :
        ts = line.split('\t')
        if len(ts) == 4 :
            rs.append(int(ts[0]))
            gs.append(int(ts[1]))
            bs.append(int(ts[2]))
            names.append(ts[3])

    unique_names = list(set(names))

    to_index, to_name = {}, {}
    i = 0
    for name in unique_names :
        to_index[name] = i
        to_name[i] = name
        i += 1

    classes = []
    for name in names :
        classes.append(to_index[name])

    rgbs = list(zip(rs, gs, bs))

    to_centroids = {}
    with open(path_data + "/ml_color-11_terms-centroids_rgbn.tsv") as file: 
        for line in file:
            ts = line.strip().split()
            if len(ts) > 0:
                rgb = [ int(ts[0]), int(ts[1]), int(ts[2]) ]
                to_centroids[ts[3]] = rgb

    return rgbs, classes, to_name, to_centroids


rgbs, classes, to_name, to_centroids = read_data()


knn = KNeighborsClassifier(n_neighbors = st.session_state.k_value)
knn.fit(rgbs, classes)


slice_ = lightness_slice(-120, 120, 
                         st.session_state.steps, 
                         st.session_state.lightness)

labs, srgbs = in_srgb_gamut(slice_)


if not st.session_state.show:
    for i in range(srgbs.shape[0]):
        rgb_in = ( srgbs[i,0] * 255, srgbs[i,1] * 255, srgbs[i,2] * 255 )
        prediction = knn.predict( [rgb_in] )
        prediction_name = to_name[prediction.item()]
        centroid = to_centroids[prediction_name]
        srgbs[i,0] = float(centroid[0]) / 255.0
        srgbs[i,1] = float(centroid[1]) / 255.0
        srgbs[i,2] = float(centroid[2]) / 255.0
    

st.subheader("K-Nearest Colors")
st.markdown("[K-nearsest neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) classification of constant lightness [CIELAB](https://en.wikipedia.org/wiki/CIELAB_color_space) slice, limited to in [gamut](https://en.wikipedia.org/wiki/Gamut) for [sRGB](https://en.wikipedia.org/wiki/SRGB).")
st.markdown("Training labels : *red, green, yellow, blue, purple, pink, orange, brown, black, gray & white*.")

# col1, col2 = st.columns([0.9, 0.1])
col1, col2 = st.columns([0.8, 0.2])

with col1:

    col_a, col_b = st.columns([0.5, 0.5])
    with col_a:
        _ = st.slider("Lightness : ", 0, 99, 65, key='lightness')
    with col_b:
        _ = st.slider("Steps : ", 11, 101, 51, key='steps')

    col_c, col_d = st.columns([0.5, 0.5])
    with col_c:
        _ = st.slider("K value : ", 1, 20, 10, key='k_value')
    with col_d:
        on = st.toggle("Show input colors", key='show')

    plt.scatter(labs[:,1], labs[:,2], c=srgbs)
    plt.xlabel('a*')
    plt.ylabel('b*')
    plt.title('CIELAB Lightness Slice')
    plt.axis('equal')

    with st.spinner(""):
        st.pyplot(plt.gcf())

        st.markdown("[**CIC 32**](https://www.imaging.org/IST/IST/Conferences/CIC/CIC2024/CIC_Home.aspx) &mdash; [**Courses & Workshops**](https://www.imaging.org/IST/Conferences/CIC/CIC2024/CIC_Home.aspx?WebsiteKey=6d978a6f-475d-46cc-bcf2-7a9e3d5f8f82&8a93a38c6b0c=3#8a93a38c6b0c) &mdash; **[Comments & Questions?](https://www.linkedin.com/feed/update/urn:li:activity:7248911412008214528/?actorCompanyId=104756822)** ")

