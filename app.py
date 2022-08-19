from flask import Flask, render_template, request, redirect, url_for
import torch
import os
import lib
from torch.utils.data import Dataset
import re
#from werkzeug.utils import secure_filename
import pickle as pkl


"""**Prediction**"""
#path
#res_path,h1,h2,h3,h4,h5 = lib.prediction( path)

 #return res_path

app = Flask(__name__)

# enable debugging mode
app.config['DEBUG'] = True

# Upload folder
UPLOAD_FOLDER = "static\\input_files"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#class="u-border-none u-btn u-button-style u-custom-color-2 u-hover-palette-1-dark-3 u-btn-3"
# Download folder
DOWNLOAD_FOLDER = "static\\results"
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# Root URL
@app.route('/') 
def home():
	return render_template("home.html")

@app.route('/about')
def about():
	return render_template("about.html")

@app.route('/page_1')
def page_1():

	return render_template("page_1.html")

@app.route('/result', methods=['POST'])
def result():
	
	# PHASE : INPUT 
	all_files = request.files.getlist('files')
	#print(len(all_files))
	
	if all_files[0].filename == '':
		return redirect(url_for('page_1_resubmit'))
	else:
		
		for file in all_files:
			if file.filename != " ":
				
				text = (file.filename).split('/')
				# set the file path
				file_path = os.path.join(app.config['UPLOAD_FOLDER'], text[1])
				print(file.filename)
				print(file_path)
				# save the file
				file.save(file_path)
		
		res_path,h1,h2 = "","", ""
		#PHASE : PREDICTION
		path = app.config['UPLOAD_FOLDER']
		res_path = lib.prediction(path)
		print("\n Prediction Phase Processing Successful\n")


		'''
		h1=h1.split('"')[1]
		h1 =h1.split('"')[0]
		
		h2=h2.split('"')[1]
		h2 =h2.split('"')[0]
		'''
		# result file saving location
		re_file = os.path.join(app.config['DOWNLOAD_FOLDER'],"file.pkz")
		print(re_file, "location of saved file in the server")
		with open(re_file, 'wb') as f:
			pkl.dump(res_path,f)
		print("\nResult saving in browser server : successful\n")

		return render_template("result.html")
 	
	'''
	#length = all_files[0] #l is an instance of class FileStorage, which Flask imports from Werkzeug
	for file in all_files:
		if file.filename != " ":
				
			text = (file.filename).split('/')
			# set the file path
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], text[1])
			print(file.filename)
			print(file_path)
			# save the file
			file.save(file_path)
	'''		
	'''
	file = request.files['files']
	#file = all_files[0]
	#text = (file.filename).split('/')
	#file_path = os.path.join(app.config['UPLOAD_FOLDER'], text[0], text[1])
	m = secure_filename(file.filename)
	file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], m)
	#print(file)
	print(app.config['UPLOAD_FOLDER'])
	print(file.filename)
	#print(secure_filename(file.filename))
	##print(file_path)
	print(file_path1)
	file.save(file_path1)
	print("file saved")
	os.remove(file_path1)
	print("file removed")
	'''
	#return render_template("result.html")

@app.route('/page_1_resubmit')
def page_1_resubmit():

		return render_template('page_1_resubmit.html')

@app.route('/read_more')
def read_more():
	return render_template("read_more.html")

if __name__ == '__main__':
	app.run(port = 5000)


