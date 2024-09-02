from dotenv import load_dotenv
import streamlit as st
import PIL.Image
import redis
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()
import pandas as pd
import json
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import os

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def analyze_product_image(uploaded_file):
    # Open the uploaded image file
    img = PIL.Image.open(uploaded_file)

    # Initialize the Google Generative AI model
    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "application/json"})

    # Generate content using the model
    response = model.generate_content(
        [img,  """Can you analyze the following product image and please provide the details of all the 
                    ingredients found in the product,A list of ingredients that are potentially harmful and 
                    the reason for the same,
                    A list of preservatives found in the product,a final conclusion 
                    of whether the product is safe to use based on the ingredients,
                    with this schema: 
                    { "Ingredients": List[str], "Potentially_Harmful_Ingredients": str, 
                    "Preservatives": str, "Final_Conclusion": str}"""
                    ])

    # Check if the response contains valid text
    if hasattr(response, 'text') and response.text:
        response_dict = json.loads(response.text)
        Ingredients = response_dict.get("Ingredients", [])
        Potentially_Harmful_Ingredients = response_dict.get("Potentially_Harmful_Ingredients", "")
        Preservatives = response_dict.get("Preservatives", "")
        Final_Conclusion = response_dict.get("Final_Conclusion", "")

        # Load the FAISS index and initialize the retrieval chain
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        # Define the query to identify carcinogenic ingredients
        query = f"Identify any carcinogenic ingredients from the following list: {Ingredients}"

        # Get the results from the chain
        result = chain({"question": query}, return_only_outputs=True)
        if "I don't know" in result['answer'] or "does not contain a list" in result['answer']:
            carcinogenic_ingredients = "None"
        else:
            carcinogenic_ingredients = result['answer']

        # Prepare the data for display
        data = {
            "Category": ["Carcinogenic Ingredients", "Potentially Harmful Ingredients", "Preservatives", "Final Conclusion"],
            "Details": [carcinogenic_ingredients, Potentially_Harmful_Ingredients, Preservatives, Final_Conclusion]
        }

        # Convert the dictionary into a DataFrame
        df = pd.DataFrame(data)

        return df

    else:
        st.write("The response was blocked or is invalid. Please try again.")
        return None

def add_to_redis(df,product_name,category_key):
    df_json = df.to_json(orient='split')
    redis_client.hset(category_key, product_name, df_json)

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
st.title("Label Lens ")

category = st.selectbox(
    "Choose a category:",
    ['','Food', 'Bath Products', 'Shampoo']
)

if(category):
    category.replace(" ","")

    category_key = category.replace(" ", "")
    cat = redis_client.hgetall(category_key)
    analysis_type = st.radio("Select analysis type:", ["Product Label Analysis", "Compare Products"])
    filtered_keys = [key for key in cat.keys() if '__' not in key]
    if analysis_type == "Product Label Analysis":
        item = st.selectbox(
            f"Search within {category}:",
            [''] + filtered_keys
        )

        # Display the information of the selected item
        if item:

            df_json = cat[item]

            # Convert the JSON string back into a DataFrame
            df = pd.read_json(df_json, orient='split')

            # Display the DataFrame in Streamlit
            st.table(df)

        else:


            st.write("If no results found. Please turn the product and click the ingredients to analyze.")
            with st.form(key='my_form'):
                uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"], key="file_uploader")
                product_name = st.text_input("Enter the name of the product")

                go_button = st.form_submit_button("Go")


            if go_button:
                uploaded_file = uploaded_file
                product_name = product_name
                if not uploaded_file:
                    st.warning("Please upload a file.")
                elif not product_name:
                    st.warning("Please enter the name of the product.")
                else:

                    st.write("File uploaded successfully! Processing...")
                    df = analyze_product_image(uploaded_file)

                    if df is not None:
                        # Display the table in Streamlit
                        st.table(df)

                        # Convert the DataFrame to JSON and store it in Redis
                        add_to_redis(df,product_name,category_key)


    elif  analysis_type == "Compare Products":
        with st.form(key='compare_products_form'):
            col1, col2 = st.columns([2, 2])
            filtered_keys = [key for key in cat.keys() if '__' not in key]
            with col1:
                st.write("**Product 1**")
                item1 = st.selectbox(
                    f"Search within {category} for Product 1:",
                    [''] + filtered_keys
                )
                if item1:

                    first_json = cat[item1]

                else:
                    # If no product is selected, allow the user to upload a file and input the product name
                    st.write("If no results found. Please turn the product and click the ingredients to analyze.")

                    # File uploader and text input for Product 1
                    uploaded_file1 = st.file_uploader(
                        "Choose a file for Product 1", type=["jpg", "jpeg", "png"], key="product1_file"
                    )

                    # Enable the text input only if a file is uploaded
                    product_name1 = st.text_input("Enter the name of Product 1")

                    # Process the uploaded file and analyze it if the file is uploaded and product name is provided
                    if uploaded_file1 and product_name1:
                        df1 = analyze_product_image(uploaded_file1)
                        if df1 is not None:
                            add_to_redis(df1, product_name1, category_key)
                            cat[product_name1] = df1.to_json(orient='split')
                            # Update item1 with the new product name
                            item1 = product_name1
                            first_json = cat[product_name1]
                            print(first_json)


            with col2:
                st.write("**Product 2**")
                item2 = st.selectbox(
                    f"Search within {category} for Product 2:",
                    [''] + filtered_keys,
                    key="product2_item"
                )
                if item2:
                    second_json = cat[item2]
                else:
                    st.write("If no results found. Please turn the product and click the ingredients to analyze.")
                    uploaded_file2 = st.file_uploader("Choose a file ", type=["jpg", "jpeg", "png"],
                                                      key="product2_file")
                    product_name2 = st.text_input("Enter the name of Product 2")
                    if uploaded_file2 and product_name2:
                        df2 = analyze_product_image(uploaded_file2)
                        if df2 is not None:
                            add_to_redis(df2, product_name2, category_key)
                            cat[product_name2] = df2.to_json(orient='split')
                            # Update item1 with the new product name
                            item2 = product_name2
                            second_json = cat[product_name2]


            # Compare button between the columns
            compare_button = st.form_submit_button("Compare")

            if compare_button:

                if item1 and item2:
                    if(item1==item2):
                        st.write("Both Products are same.Please select a different product for comparison")
                    # Convert JSON strings back into Python dictionaries
                    elif redis_client.hexists(category_key, f"{item1}__{item2}"):

                        existing_result = redis_client.hget(category_key,  f"{item1}__{item2}")
                        st.markdown(f"**Comparison Result:**\n\n{existing_result}")

                    elif redis_client.hexists(category_key, f"{item2}__{item1}"):

                        existing_result = redis_client.hget(category_key, f"{item2}__{item1}")
                        st.markdown(f"**Comparison Result:**\n\n{existing_result}")
                    else:
                        product1_data = json.loads(first_json)
                        product2_data = json.loads(second_json)

                        data1 = product1_data["data"]
                        data2 = product2_data["data"]

                        # Convert data to a dictionary for easier access
                        data1_dict = {row[0]: row[1] for row in data1}
                        data2_dict = {row[0]: row[1] for row in data2}
                        comparison_query = f"""
                                Compare the following two products based on their ingredients, harmful substances, preservatives, 
                                and overall safety. Determine which product is better and explain why.
    
                                Product 1:
                                Carcinogenic Ingredients: {data1_dict.get("Carcinogenic Ingredients", [])}
                                Potentially Harmful Ingredients: {data1_dict.get("Potentially Harmful Ingredients", "")}
                                Preservatives: {data1_dict.get("Preservatives", "")}
                                Final Conclusion: {data1_dict.get("Final Conclusion", "")}
    
                                Product 2:
                                Carcinogenic Ingredients: {data2_dict.get("Carcinogenic Ingredients", [])}
                                Potentially Harmful Ingredients: {data2_dict.get("Potentially Harmful Ingredients", "")}
                                Preservatives: {data2_dict.get("Preservatives", "")}
                                Final Conclusion: {data2_dict.get("Final Conclusion", "")}
    
                                Which product is better, and why?
                                """

                        # Initialize the LLM
                        model = genai.GenerativeModel("gemini-1.5-flash")

                        # Pass the query to the LLM
                        result = model.generate_content(comparison_query)

                        # Display the result
                        if result and result.text:
                            st.markdown(f"**Comparison Result:**\n\n{result.text}")
                            strr=f"{item1}__{item2}"
                            redis_client.hset(category_key, strr, result.text)

                        else:
                            st.write("The comparison could not be performed. Please try again.")
                else:
                    st.write("Please select both products to compare.")
