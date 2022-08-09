import pandas as pd
import numpy as np
from flask import Flask, jsonify, request,send_file
from flask_restful import Resource, Api, reqparse
from sklearn.neighbors import NearestNeighbors
app = Flask(__name__)
api = Api(app)
# Replace the reading logic for CSV to reading data from data source
df = pd.read_csv('data2.csv')
cross_table = pd.crosstab(df['DescriptionArticle'], df['ClientCode'])

@app.route("/get-popular-wines", methods=['GET'])
def get_popular_wines():
    top_10_wines = df[['DescriptionArticle']].groupby('DescriptionArticle')['DescriptionArticle'].count().nlargest(10)
    top_10_wines_list = top_10_wines.keys().tolist()
    return jsonify({
        "message": "Top 10 Wines",
        "data": top_10_wines_list.to_dict()
        })

@app.route("/get-top-customers", methods=['GET'])
def get_top_customers():
    top_10_clients = df[['ClientCode']].groupby('ClientCode')['ClientCode'].count().nlargest(10)
    top_10_client_list = top_10_clients.keys().tolist()
    return jsonify({
        "message": "Top 10 Customers",
        "data": top_10_client_list.to_dict()
        })

@app.route("/get-wine-based-recommendation", methods=['POST'])
def get_wine_based_recommendation():
    wine_title = request.args.get('wine')
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(cross_table.values)
    distances, indices = knn.kneighbors(cross_table.values, n_neighbors=5)
    index_user_likes = cross_table.index.tolist().index(wine_title) # get an index for a wine
    sim_wines = indices[index_user_likes].tolist() # make list for similar wines
    wine_distances = distances[index_user_likes].tolist() # the list for distances of similar wines
    id_wine = sim_wines.index(index_user_likes) # get the position of the wine itself in indices and distances
    sim_wines.remove(index_user_likes) # remove the wine itself in indices
    wine_distances.pop(id_wine) # remove the wine itself in distances
    j = 1
    wine_arr = []
    for i in sim_wines:
        wine_arr.append(cross_table.index[i])
        j = j + 1

    return jsonify({
        "message": "Recommended Wine",
        "data": wine_arr.to_dict()
        })

@app.route("/get-user-based-wines", methods=['GET', 'POST'])
def get_user_based_wines():
    number_neighbors = request.args.get('number')
    user = request.args.get('user')
    num_recommendation = request.args.get('num_recommendation')
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(cross_table.values)
    distances, indices = knn.kneighbors(cross_table.values, n_neighbors=number_neighbors)
    user_index = cross_table.columns.tolist().index(user)
    for m,t in list(enumerate(cross_table.index)):
        if cross_table.iloc[m, user_index] == 0:
            sim_wines = indices[m].tolist()
            wine_distances = distances[m].tolist()
    
            if m in sim_wines:
                id_wine = sim_wines.index(m)
                sim_wines.remove(m)
                wine_distances.pop(id_wine) 

            else:
                sim_wines = sim_wines[:number_neighbors-1]
                wine_distances = wine_distances[:number_neighbors-1]
           
            wine_similarity = [1-x for x in wine_distances]
            wine_similarity_copy = wine_similarity.copy()
            nominator = 0

            for s in range(0, len(wine_similarity)):
                if df.iloc[sim_wines[s], user_index] == 0:
                    if len(wine_similarity_copy) == (number_neighbors - 1):
                        wine_similarity_copy.pop(s)
                    else:
                        wine_similarity_copy.pop(s-(len(wine_similarity)-len(wine_similarity_copy)))
                else:
                    nominator = nominator + wine_similarity[s]*cross_table.iloc[sim_wines[s],user_index]
          
            if len(wine_similarity_copy) > 0:
                if sum(wine_similarity_copy) > 0:
                    predicted_r = nominator/sum(wine_similarity_copy)
                else:
                    predicted_r = 0
            else:
                predicted_r = 0
        
            df.iloc[m,user_index] = predicted_r
    recommended_wines = []
    for m in cross_table[cross_table[user] == 0].index.tolist():
        index_df = cross_table.index.tolist().index(m)
        predicted_rating = df.iloc[index_df, df.columns.tolist().index(user)]
        recommended_wines.append((m, predicted_rating))
    sorted_rm = sorted(recommended_wines, key=lambda x:x[1], reverse=True)
    rank = 1
    recommendation_user = []
    for recommended_wines in sorted_rm[:num_recommendation]:
        recommendation_user.append(format(rank, df[df['Item code'] == recommended_wines[0]]['DescriptionArticle'].iloc[0], recommended_wines[1]))
        rank = rank + 1

    return jsonify({
        "message": "USer based recommendation",
        "data": recommendation_user.to_dict()
        })

if __name__ == '__main__':
    app.run(debug=True, port=8000)