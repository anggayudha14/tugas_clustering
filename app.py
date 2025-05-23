from flask import Flask, render_template, request
import os
import pandas as pd
from model.clustering import preprocess_data, plot_elbow, perform_clustering, plot_cluster

app = Flask(__name__)
UPLOAD_FOLDER = 'dataset'
PLOT_FOLDER = 'static'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    clustered_data = None
    plot_paths = {}
    table_html = ''
    k_selected = None

    if request.method == 'POST':
        file = request.files['file']
        k = request.form.get('k_value')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
            file.save(filepath)

            try:
                df_selected, X_scaled = preprocess_data(filepath)
            except ValueError as e:
                return f"<h3>Error: {e}</h3>"

            plot_elbow(X_scaled, PLOT_FOLDER)
            plot_paths = {
                'elbow_inertia': 'static/elbow_inertia.png',
                'silhouette': 'static/silhouette.png'
            }

            if k:
                k = int(k)
                clustered_data, labels = perform_clustering(X_scaled, df_selected, k)
                plot_cluster(clustered_data, labels, os.path.join(PLOT_FOLDER, 'cluster_plot.png'))
                plot_paths['cluster'] = 'static/cluster_plot.png'
                table_html = clustered_data.to_html(classes='table table-bordered', index=False)
                k_selected = k

    return render_template('index.html', plot_paths=plot_paths, table_html=table_html, k_selected=k_selected)

if __name__ == '__main__':
    app.run(debug=True)
