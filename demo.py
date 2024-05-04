import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import sklearn
import plotly.express as px
import xgboost
import joblib

df_eda=pd.read_csv("sample.csv")

#layout
st.set_page_config(page_title="Zomato Analysis",layout="wide")

#read data


# Visualization Function
def bar(data_frame,x,y,title_text,color=None):
    fig=px.bar(data_frame=data_frame,x=x,y=y,color=color, barmode='group',text_auto="0.2s")
    fig.update_traces(textfont_size=12,textposition="outside")
    fig.update_layout(title_text=title_text,title_x=0.5)
    return fig

def sunburst(data,names,path,values,title_text):
    fig=px.sunburst(data_frame=data,names=names,path=path,values=values,
            width=900,height=900)
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(title_text=title_text,title_x=0.5)
    return fig

pages = st.sidebar.radio("Pages",
                            ["Home","Data","Analysis","Predict state your restaurant"])

if pages=="Home":
    # Page content    
    st.markdown(''' <h6>
        Zomato is an Indian multinational restaurant aggregator and food delivery company. It was founded by Deepinder Goyal and Pankaj Chaddah in 2008. Zomato provides information, menus and user-reviews of restaurants as well as food delivery options from partner restaurants in more than 1,000 Indian cities and towns, as of 2022–23. Zomato rivals Swiggy in food delivery and hyperlocal space.</center> </h6> ''', unsafe_allow_html=True)
    
    st.image('image.png', caption='Logo Zomato',width=500)
    st.markdown("[link zomato website](https://www.zomato.com/)")

if pages=="Data":
    st.header("[Link of data](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)")
    st.subheader("Data Describtion : ")
    st.write('''
                | Attribute | Description |
                |----------|----------|
                |name	|Name's restaurant
                |online_order |The customer can book an order online or not 
                |book_table	|The customer can book a table or not
                |rate	|Rate's restaurant
                |votes  |votes of restaurant
                |phone	|phone of restaurant
                |location	|location of restaurant
                |rest_type	|The type of service provided by the restaurant
                |dish_liked	|The type of dish provided by the restaurant
                |cuisines	|The type of cuisines provided by the restaurant
                |approx_cost(for two people)		|approx cost for two people
                |menu_item	|the list has content menu of the restaurant
                |listed_in(type)		|The type of service provided by the restaurant
                |reviews_list	|list contents reviews of restaurant
                |listed_in(city)	|city of restaurant
                                                        ''')
    st.subheader("Display first 10 rows after cleaning : ")
    st.dataframe(df_eda.head(10))
if pages=="Analysis":

    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))

    with row0_1:
        st.title('Zomato Analysis')
        st.markdown(''' <h6>
                        This app is created to analyze the data of a resturants in zomato website to predict if your resturant will be successful or not </center> </h6> ''', unsafe_allow_html=True)
    with row0_2:
        st.text("")
        st.subheader('Varshitha N ')
        st.subheader('Sowjanya Rai K')
    # Dividing our analysis into tabs, each tab contains
    over_view,Influencing_factors,tab_conclusion = st.tabs(['Over View',"Influencing factors",'Conclusion'])
    with over_view:
        st.title("Analytical overview :")
        st.markdown("Our target is rate :")
        st.markdown(" 1 : represtend to succesfull resturant")
        st.markdown(" 0: represtend to failed resturant")
        st.markdown("*"*50)
        # insights in this tab
        st.write('The information in this tab can answer the following questions :') 
        st.write('    1- Top 10 restaurant successful have rate =1')
        st.write('    2- Average votes for rate ')
        st.write('    3- Average approx cost for two people for each rate')
        # First question
        st.subheader("1- Top 10 restaurant successful have rate =1")
        data=df_eda[df_eda.rate==1].groupby("name")["rate"].count().reset_index().sort_values(by="rate",ascending=False).head(10)
        fig=bar(data,"name","rate","Top 10 restaurant successful have rate =1")
        st.plotly_chart(fig)
        st.markdown("*"*50)
        # second question
        st.subheader("2- Average votes for rate")
        data=df_eda.groupby("rate")["votes"].mean().reset_index()
        fig=bar(data,"rate","votes","Average votes for rate")
        st.plotly_chart(fig)
        st.markdown("- Most restaurant successful have vote")
        st.markdown("*"*50)
        # third question
        st.subheader("3- Average approx cost for two people for each rate")
        data=df_eda.groupby("rate")["approx_cost(for two people)"].mean().reset_index()
        fig=bar(data,"rate","approx_cost(for two people)","Average approx cost for two people for each rate")
        st.plotly_chart(fig)
        st.markdown("*"*50)
    with Influencing_factors:
        st.title("Influencing factors :")
        st.write('The information in this tab can answer the following questions :')
        st.subheader("What are the Influencing factors that make any restaurant is successful?")
        st.write("To answer the previous question I will show some graph to show Influencing factors")
        st.markdown("*"*50)
        st.write('    1- Relation between online_order and rate')
        st.write('    2- Relation between book_table and rate ')
        st.write('    3- Relation between phone and rate')
        st.write('    4- Top 10 locations that have a higher Competition or low competition')
        st.write('    5- Relation between city and rate')
        st.write('    6- Top 10 rest type has good reputation (rate=1)')
        st.write('    7- Top 10 cuisines has good reputation (rate=1)')
        st.write('    8- Relation between  type of service provided and rate')
        st.write('    9- Relation between menu_item and rate')
        #question1
        st.subheader("1- Relation between online_order and rate")
        #pandas
        data=df_eda.groupby(["rate","online_order"]).agg({"online_order":"count"}).rename(columns={"online_order":"count"}).reset_index()
        fig= bar(data,"rate","count",color="online_order",title_text="Relation between online_order and rate")
        st.plotly_chart(fig)
        st.markdown("- Note : To be restaurant sucessful you have to online_order service is available")
        st.markdown("*"*50)

        #question2
        st.subheader("2- Relation between book_table and rate")
        #pandas
        data=df_eda.groupby(["rate","book_table"]).agg({"book_table":"count"}).rename(columns={"book_table":"count"}).reset_index()
        fig=bar(data,"rate","count",color="book_table",title_text="Relation between book_table and rate")
        st.plotly_chart(fig)
        st.markdown("- Note : There is not relation between book_table and rate , that mean you can be restaurant successful without book table service is available")

        st.markdown("*"*50)
        
        #question3
        st.subheader("3- Relation between phone and rate")
        data=df_eda.groupby(["rate","phone"]).agg({"phone":"count"}).rename(columns={"phone":"count"}).reset_index()
        fig=bar(data,"rate","count",color="phone",title_text="Relation between phone and rate")
        st.plotly_chart(fig)
        st.markdown("- Note : To open any resturant must be have phone number")
        st.markdown("*"*50)

        #question4
        st.subheader("4- Top 10 locations that have a higher Competition or low competition ")
        data_higher=df_eda[df_eda.rate==1].groupby("location")["rate"].count().reset_index().sort_values(by="rate",ascending=False).head(10)
        data_lower=df_eda[df_eda.rate==0].groupby("location")["rate"].count().reset_index().sort_values(by="rate",ascending=False).tail(10)
        select=st.radio("Select your interest",["Highest Competition","Lowest competition"])
        if select=="Highest Competition":
            container = st.container()
            figuer,col2, data_show = container.columns(3)
            with figuer:
                fig=bar(data_higher,"location","rate",title_text="Top 10 locations that have a successful restaurant and Competition is high((rate=1)")
                st.plotly_chart(fig)
            with data_show:
                st.dataframe(data_higher)
            st.markdown("- Note : Koramangala 5th Block, BTM and Indiranagar have successful restaurant and Competition is high.")

        if select=="Lowest competition":
            container = st.container()
            figuer,col2, data_show = container.columns(3)
            with figuer:
                fig=bar(data_lower,"location","rate",title_text="Top 10 locations with low competition (rate=0)")
                st.plotly_chart(fig)
            with data_show:
                st.dataframe(data_lower)
            st.markdown("- Note : in Mysore Road and Hebbal, competition is low.")

        st.markdown("__in my opinion__: you can open resturant in Mysore Road or Hebbal that have lower competition make good reputation after that open in Koramangala 5th Block or BTM that have higher competition.")
        st.markdown("*"*50)

        #question5
        st.subheader("5- Relation between city and rate")
        #pandas
        data=df_eda.groupby(["rate","listed_in(city)"]).agg({"listed_in(city)":"count"}).rename(columns={"listed_in(city)":"count"}).reset_index().sort_values(by="count",ascending=False)
        data.rate=data.rate.astype("O")
        bar(data,"listed_in(city)","count",color="rate",title_text="Relation between city and rate")
        st.plotly_chart(fig)
        st.markdown(" Note : BTM , Koramangala 7th Block and Koramangala 5th Block, have the highest percentage of restaurant successful or failed , so competition is very high")
        st.markdown("*"*50)

        #question6
        st.subheader("6- Top 10 rest type has good reputation (rate=1)")
        data=df_eda[df_eda.rate==1].groupby("rest_type")["rate"].count().reset_index().sort_values(by="rate",ascending=False).head(10)
        fig=bar(data,"rest_type","rate",title_text="Top 10 rest type has good reputation (rate=1)")
        st.plotly_chart(fig)
        st.markdown(" Note : To be successful restaurant you have to rest type will be Casual Dining or Quick Bites")
        st.markdown("*"*50)
        
        #question7
        st.subheader("7- Top 10 cuisines has good reputation (rate=1)")
        data=df_eda[df_eda.rate==1].groupby("cuisines")["rate"].count().reset_index().sort_values(by="rate",ascending=False).head(10)
        fig=bar(data,"cuisines","rate",title_text="Top 10 cuisines has good reputation (rate=1)")
        st.plotly_chart(fig)
        st.markdown(" Note : To be successful restaurant you have to cuisines type will be North Indian or Chinese or South Indian")
        st.markdown("*"*50)


        #question8
        st.subheader("8- Relation between  type of service provided and rate")
        data=df_eda.groupby(["rate","listed_in(type)"]).agg({"listed_in(type)":"count"}).rename(columns={"listed_in(type)":"count"}).reset_index().sort_values(by="count",ascending=False)
        data.rate=data.rate.astype("O")
        fig=bar(data,"listed_in(type)","count",color="rate",title_text="Relation between  type of service provided and rate")
        st.plotly_chart(fig)
        st.markdown("-Note : The type of service provided by the most resturant succsfull or failed is Delivery or Dine-out.")
        st.markdown("*"*50)

        #question9
        st.subheader("9- Relation between menu_item and rate")
        data=df_eda.groupby(["rate","menu_item"]).agg({"menu_item":"count"}).rename(columns={"menu_item":"count"}).reset_index()
        fig=bar(data,"rate","count",color="menu_item",title_text="Relation between menu_item and rate")
        st.plotly_chart(fig)
        st.markdown("- Note : There is not relation between successful or failed restaurant have menu or not , but if you want your restaurant is successful perfer put menu depend on percentage between (1 , 0 , have menu or not) in graph")
        st.markdown("*"*50)

    with tab_conclusion:
        st.subheader("Some advice to be your successful restaurant :")
        st.write("1) online_order service is available")
        st.write("2) must be have phone number")
        st.write("3) open your resturant in Mysore Road or Hebbal that have lower competition make good reputation after that open in Koramangala 5th Block or BTM that have higher competition.")
        st.write("4) you have to rest type will be Casual Dining or Quick Bites")
        st.write("5) you have to cuisines type will be North Indian or Chinese or South Indian")
        st.write("6) prefer put menu in website")
        st.write("7) you have to service provided will be Delivery or Dine-out")

if pages=="Predict state your restaurant":
    
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))

    with row0_1:
        st.title('Prediction your resturant')
        st.markdown(''' <h6>
                        You will be open your restaurant and you want to know if your restaurant will be succesfull or not.  </center> </h6> ''', unsafe_allow_html=True)
    with row0_2:
        st.text("")
        st.subheader('Linkedin : App by [Ahmed Ramadan](https://www.linkedin.com/in/ahmed-ramadan-18b873230/) ')
        st.subheader('Github : App by [Ahmed Ramadan](https://github.com/AhmedRamadan74/zomato)')

    model=joblib.load("model.pkl") #load model
    inputs=joblib.load("input.pkl") #load input

    def Make_Prediction(online_order,book_table,phone,location,rest_type,cuisines,approx_cost,menu_item,listed_in_type,listed_in_city):
        df_pred = pd.DataFrame(columns=inputs)
        df_pred.at[0,"online_order"] = online_order
        df_pred.at[0,"book_table"] = book_table
        df_pred.at[0,"phone"] = phone
        df_pred.at[0,"location"] = location
        df_pred.at[0,"rest_type"] = rest_type
        df_pred.at[0,"cuisines"] = cuisines
        df_pred.at[0,"approx_cost(for two people)"] = approx_cost
        df_pred.at[0,"menu_item"] = menu_item
        df_pred.at[0,"listed_in(type)"] = listed_in_type
        df_pred.at[0,"listed_in(city)"] = listed_in_city
        #prediction output
        result = model.predict(df_pred)
        if result[0] ==1:
            return "This restaurant will be successful"
        else:
            return "This restaurant will be not successful"
    
    list1=df_eda["location"].unique().tolist()
    list2=df_eda["rest_type"].unique().tolist()
    list3=df_eda["cuisines"].unique().tolist()
    list4=df_eda["listed_in(type)"].unique().tolist()
    list5=df_eda["listed_in(city)"].unique().tolist()

    st.write("Frist , entry some inforamtion for your restaurant ")
    online_order=st.selectbox("The customer can book an order online or not :",['Yes', 'No'])
    book_table=st.selectbox("The customer can book a table or not :",['Yes', 'No'])
    phone=st.selectbox("The resturant have number or not :",['have phone', 'not have phone'])
    location=st.selectbox("Location of a restaurant :",list1)
    rest_type=st.selectbox("The type of service provided by the restaurant :",list2)
    cuisines=st.selectbox("The type of cuisines provided by the restaurant :",list3)
    approx_cost=st.number_input("approx cost for two people :")
    menu_item=st.selectbox("The resturant have menu or not :",['have menu', 'not have menu'])
    listed_in_type=st.selectbox("The type of service provided by the restaurant :",list4)
    listed_in_city=st.selectbox("City of a resturant :",list5)

    #show the result
    btn=st.button("Predict")
    if btn:
        st.write(Make_Prediction(online_order,book_table,phone
                                 ,location,rest_type,cuisines,
                                 approx_cost,menu_item,
                                 listed_in_type,listed_in_city))
