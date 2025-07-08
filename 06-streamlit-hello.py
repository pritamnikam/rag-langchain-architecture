import streamlit as st

# Set up the page configuration (title, layout, etc.)
st.set_page_config(page_title='Streamlit Demo', page_icon=':books:', layout='centered')

# Add a title and description to the app
st.title('📚 Streamlit Demo')
st.write('''
Welcome to your first Streamlit app! This simple demo shows how to build interactive web apps with Python.
''')

# Sidebar for additional info or navigation
with st.sidebar:
    st.header('About')
    st.info('This is a minimal Streamlit demo. You can extend it for data apps, dashboards, and more!')
    st.markdown('---')
    st.write('Made with [Streamlit](https://streamlit.io/)')

# Main interactive section
st.subheader('Try it out:')

# Text input for user name
name = st.text_input('Enter your name:', '')

# Button to trigger greeting
if st.button('Say Hello'):
    if name.strip():
        st.success(f'Hello, {name}! 👋')
    else:
        st.warning('Please enter your name above.')

# Add a simple expander for extra info
with st.expander('How does this work?'):
    st.write('''
    - Streamlit lets you build web apps with pure Python.
    - UI elements like buttons, text inputs, and more are available out of the box.
    - Try editing this script to add your own features!
    ''')

# Footer
st.markdown('---')
st.caption('© 2025 Streamlit Demo | Powered by Streamlit')