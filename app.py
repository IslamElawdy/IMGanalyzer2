import streamlit as st
from utils.localization import get_text
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="IMGanalyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'language' not in st.session_state:
    st.session_state['language'] = 'de'
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

def main():
    # Sidebar: Language & Navigation
    with st.sidebar:
        st.header("IMGanalyzer")
        st.markdown("""
        <style>
        .subtitle {
            font-size: 12px;
            color: #555;
            margin-top: -15px;
            margin-bottom: 20px;
        }
        a {
            text-decoration: none;
            color: #555;
        }
        a:hover {
            color: #000;
            text-decoration: underline;
        }
        </style>
        <div class="subtitle">
            <a href="https://linktr.ee/Is.elawdy" target="_blank">AwadyStudioLab [Islam Elawdy]</a>
        </div>
        """, unsafe_allow_html=True)

        # Language Selector
        lang_choice = st.radio(
            "Sprache / Language",
            ('Deutsch', 'English'),
            index=0 if st.session_state['language'] == 'de' else 1
        )
        st.session_state['language'] = 'de' if lang_choice == 'Deutsch' else 'en'
        lang = st.session_state['language']

        st.markdown("---")

        # Module Selector
        module_options = [
            get_text('module_upload', lang),
            get_text('module_pixel', lang),
            get_text('module_channels', lang),
            get_text('module_patch', lang),
            get_text('module_conv', lang),
            get_text('module_activation', lang),
            get_text('module_noise', lang),
            get_text('module_augmentation', lang),
            get_text('module_cnn_features', lang),
            get_text('module_classification', lang),
            get_text('module_adversarial', lang),
            get_text('module_autoencoder', lang),
            get_text('module_training', lang),
            get_text('module_detection', lang),
            get_text('module_style', lang),
            get_text('module_segmentation', lang),
            get_text('module_basics', lang),
            get_text('module_manipulation', lang),
            get_text('module_cnn_explainer', lang),
        ]

        selected_module_name = st.radio(get_text('choose_module', lang), module_options)

    # Main Content Area
    st.title(get_text('app_title', lang))

    # Route to Module

    if selected_module_name == get_text('module_upload', lang):
        import modules.upload as upload_module
        upload_module.render(lang)

    elif st.session_state['uploaded_image'] is None:
        st.warning(get_text('no_image_warning', lang))
        # Fallback to upload if no image
        import modules.upload as upload_module
        upload_module.render(lang)

    else:
        # Dispatch to other modules
        if selected_module_name == get_text('module_pixel', lang):
            import modules.pixel_explorer as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_channels', lang):
            import modules.channels as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_patch', lang):
            import modules.patch_viewer as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_conv', lang):
            import modules.conv_playground as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_activation', lang):
            import modules.activation_pooling as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_noise', lang):
            import modules.noise_robustness as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_augmentation', lang):
            import modules.augmentation as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_cnn_features', lang):
            import modules.cnn_features as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_classification', lang):
            import modules.classification as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_adversarial', lang):
            import modules.adversarial as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_autoencoder', lang):
            import modules.autoencoder as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_training', lang):
            import modules.training as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_detection', lang):
            import modules.detection as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_style', lang):
            import modules.style_transfer as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_segmentation', lang):
            import modules.segmentation as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_basics', lang):
            import modules.image_basics as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_manipulation', lang):
            import modules.basic_manipulation as mod
            mod.render(lang)
        elif selected_module_name == get_text('module_cnn_explainer', lang):
            import modules.cnn_explainer as mod
            mod.render(lang)

if __name__ == "__main__":
    main()
