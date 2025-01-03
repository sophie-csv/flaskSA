import os

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
from regular import process_csv, positive_word_cloud, negative_word_cloud, positive_frequency_graph, negative_frequency_graph
from custom import process_aspect_csv, density_plot, box_plot, correlation, pos_wordcloud, neg_wordcloud
app = Flask(__name__)


ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# in upload create radioboxes
# create a routing function that reroutes based on aspect selection/not. 

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_filename = f'{filename.split(".")[0]}_{str(datetime.now().date())}.csv'
            save_location = os.path.join('input', new_filename)
            file.save(save_location)

            if 'nouns' in request.form and request.form['nouns'] == 'on':
                output_file = process_aspect_csv(save_location)
                return redirect(url_for('aspect_download'))
            else:
                output_file = process_csv(save_location)
            #return send_from_directory('output', output_file)
                return redirect(url_for('download'))

    return render_template('upload.html')

import zipfile
from flask import send_file

# <---------------- ABSA ROUTES --------------------->
# downloads the correct graph, but displays the wrong one?

@app.route('/aspect_download', methods=['GET'])
def aspect_download():
    # Logic for handling aspect_download
    return render_template('aspect_download.html')

@app.route('/aspect_download/density')
def plot_density():
    density = density_plot()
    return send_from_directory('output', density, as_attachment=True)

@app.route('/aspect_download/boxplot')
def plot_boxplot():
    box_plot_image = box_plot()
    return send_from_directory('output', box_plot_image, as_attachment=True)

@app.route('/aspect_download/correlational')
def plot_correlational():
    corr = correlation()
    return send_from_directory('output', corr, as_attachment=True)

@app.route('/aspect_download/positivewordcloud')
def plot_pos_wc():
    pwc = pos_wordcloud()
    return send_from_directory('output', pwc, as_attachment=True)

@app.route('/aspect_download/negativewordcloud')
def plot_neg_wc():
    nwc = neg_wordcloud()
    return send_from_directory('output', nwc, as_attachment=True)


# <---------------- ABSA ROUTES --------------------->


@app.route('/download_all')
def download_all():
    # Name of the zip file
    zip_filename = 'output_files.zip'
    zip_filepath = os.path.join('zip_file', zip_filename)
    
    # Create a zip file containing all files from the 'output' directory
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('zip_file'):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), 'zip_file'))
    
    # Serve the zip file for download
    return send_file(zip_filepath, as_attachment=True)


@app.route('/download')
def download():
    return render_template('download.html', files=os.listdir('output'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('output', filename)

@app.route('/download/positive_wordcloud')
def download_positive_word_cloud():
    pos_wc = positive_word_cloud()
    return send_from_directory('output', pos_wc, as_attachment=True)

@app.route('/download/negative_wordcloud')
def download_negative_word_cloud():
    neg_wc = negative_word_cloud()
    return send_from_directory('output', neg_wc, as_attachment=True)

@app.route('/download/positive_frequency_graph')
def download_positive_frequency_graph():
    pfg = positive_frequency_graph()
    return send_from_directory('output', pfg, as_attachment=True)

@app.route('/download/negative_frequency_graph')
def download_negative_frequency_graph():
    pfg = negative_frequency_graph()
    return send_from_directory('output', pfg, as_attachment=True)


# ONLY WORKS WHEN YOU TERMINATE WITH CONTROL + C 
import atexit

def remove(directory_path):
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if not os.path.isdir(item_path):
            os.remove(item_path)
        else:
            remove(item_path)

atexit.register(lambda: remove('input'))
atexit.register(lambda: remove('output'))


if __name__ == '__main__':
    app.run(debug=True)