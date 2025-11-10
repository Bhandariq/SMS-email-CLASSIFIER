from flask import Flask,render_template,url_for,request
import pickle


app = Flask(__name__)

#load the model
clf = pickle.load(open('NB_spam_model.pkl','rb'))
cv = pickle.load(open('transform.pkl','rb'))

@app.route('/', methods = ["GET", "POST"])
def home():
	if request.method == "GET":

		return render_template('index.html', prediction = None)
	
	elif request.method == "POST":
		

		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		mssge_prediction = clf.predict(vect)

		return render_template('index.html', prediction = mssge_prediction)

if __name__ == '__main__':
	app.run(debug=True)