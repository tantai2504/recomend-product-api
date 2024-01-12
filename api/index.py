import os

import replicate
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
os.environ.get("REPLICATE_API_TOKEN")


@app.route('/api/recommend', methods=['POST'])
def recommend_products():
    try:
        # Đọc dữ liệu từ file CSV
        data = pd.read_csv(request.files['csv_file'])

        # Chuyển cột 'rating_id' từ số sang kiểu số để có thể tính trung bình
        data['rating'] = data['rating'].astype(float)

        # Gom nhóm theo 'user_id' và 'product_id' và tính trung bình rating
        average_ratings = data.groupby(['productId'])['rating'].mean().reset_index()

        # Kết hợp các trường dữ liệu thành một trường duy nhất
        average_ratings['combined_data'] = average_ratings['productId'] + ' ' + average_ratings['rating'].astype(str)

        # Sử dụng CountVectorizer để chuyển đổi dữ liệu thành ma trận term-document
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(average_ratings['combined_data'])

        # Tính cosine similarity
        cosine_sim = cosine_similarity(X, X)

        # Nhập user_id của người dùng từ đầu vào
        input_user_id = request.form.get('productId')

        # Tìm index của người dùng trong dữ liệu
        user_index = average_ratings[average_ratings['productId'] == input_user_id].index.values[0]

        # Lấy các giá trị cosine similarity liên quan đến người dùng
        similarities = cosine_sim[user_index]

        # Lấy index của các sản phẩm có cosine similarity cao nhất (loại bỏ sản phẩm đã xem/rated)
        top_similar_products_index = similarities.argsort()[::-1][1:6]

        # Lấy thông tin về các sản phẩm được đề xuất
        recommended_products = average_ratings.iloc[top_similar_products_index][['productId', 'rating']]

        # Chuyển dữ liệu kết quả thành định dạng JSON và trả về
        result = recommended_products.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route("/")
def index():
    
    return "This is an alt tag generator!"

@app.route('/generate')
def home():
  # Get imageUrl query param
  args = request.args
  imageUrl = args.to_dict().get('imageUrl')

  # Run ML Model with imageUrl
  model = replicate.models.get("salesforce/blip")
  version = model.versions.get("2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746")

  # Get the alt text result and return it
  return version.predict(image=imageUrl)
