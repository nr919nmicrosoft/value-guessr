from src.valueguessr.utils import greet_user
import streamlit as st

def main():
    st.title("Value Guessr")
    name = st.text_input("Enter your name:")
    if name:
        greeting = greet_user(name)
        st.write(greeting)

if __name__ == "__main__":
    main()