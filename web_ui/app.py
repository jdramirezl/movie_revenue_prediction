from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
app.debug = True


# Load the model
with open("./final_pipeline.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        original_language = request.form["original_language"]
        has_famous_company = int(request.form["has-famous-company"])
        release_date = request.form["release_date"]
        has_famous_credits = int(request.form["has-famous-credits"])
        budget = float(request.form["budget"])
        runtime = float(request.form["runtime"])

        # Get selected genres and create columns
        selected_genres = request.form.getlist("genres")
        genre_columns = {
            "action": 0,
            "adventure": 0,
            "animation": 0,
            "comedy": 0,
            "crime": 0,
            "documentary": 0,
            "drama": 0,
            "family": 0,
            "fantasy": 0,
            "history": 0,
            "horror": 0,
            "music": 0,
            "mystery": 0,
            "romance": 0,
            "sciencefiction": 0,
            "thriller": 0,
            "tvmovie": 0,
            "war": 0,
            "western": 0,
        }

        # Mark selected genres as 1
        for genre in selected_genres:
            if genre in genre_columns:
                genre_columns[genre] = 1

        # Create input data for prediction
        data = {
            "original_language": original_language,
            "has-famous-company": has_famous_company,
            "release_date": release_date,
            "has-famous-credits": has_famous_credits,
            "budget": budget,
            "runtime": runtime,
            **genre_columns,
        }
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        predicted_value = prediction.tolist()[0]  # Assuming a single value output
        # update to a dollarized format with only 2 decimal places, adding the $ sign and points every 3 numbers
        predicted_value = f"${predicted_value:,.2f}"

        # Render the result template with the prediction
        return render_template("result.html", prediction=predicted_value)
    return render_template("index.html")
