from flask import Flask, render_template, request, redirect, url_for
from flask import session
from werkzeug.utils import secure_filename

import os
import pickle
import sys
sys.path.append("D:/share/TOSS/toss")
from Get_Initial_Guess import get_the_valid_t
from get_fos import GET_FOS
from Get_TOS import get_Oxidation_States


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'cif'}  # Assuming the structures are in CIF format.

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'some_secret_key_here'  # Change this to a random string


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    valid_t = []
    squared_value = None

    # Check if the "process_file" button was pressed.
    if 'process_file' in request.form:
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            info, valid_t = get_the_valid_t(m_id=None, i=None, server=True, filepath=filepath)
            file.save(filepath)
            session['info'] = info
            session['valid_t'] = valid_t
            session['filename'] = filepath

    # Check if the "assemble_action" button was pressed.
    if 'initial_guess_action' in request.form:
        # Retrieving the data
        file_input = session.get('filename')
        print(file_input)
        sites_output = session.get('valid_t', [])
        entered_value = float(request.form.get('number_to_assemble', None))

        if file_input and sites_output and entered_value:    
            #result_init = initial_guess(entered_value, sites_output, file_input)
            GFOS = GET_FOS()
            result_init = GFOS.initial_guess(m_id=None, delta_X=0.1, tolerance=entered_value, tolerance_list=sites_output, res=None, server=True, filepath=file_input)
            session['init_result'] = result_init.to_html(classes='dataframe')
        else:
            print(f"Missing data: {file_input}, {sites_output}, {entered_value}")  # Print missing data for debugging

    if 'loss_loop_action' in request.form:
        # Retrieving the data
        file_input = session.get('filename')
        sites_output = session.get('valid_t', [])

        if file_input and sites_output:
            #try:
                #result_loss_loop = loss_loop(entered_value, sites_output, file_input)  # Assuming the function loss_loop exists in app_func
                #result_loop = LOOP(sites_output, file_input)
            result_loop = get_Oxidation_States(m_id=None,i=None, atom_pool="all", server=True, filepath=file_input,input_tolerance_list=sites_output)
            session['loop_result'] = result_loop.to_html(classes='dataframe')
            #except Exception as e:
                #session['loop_result'] = f"An error occurred: {e}"
        else:
            print(f"Missing data: {file_input}, {sites_output}")  # Print missing data for debugging


    return render_template('index.html', sites=session.get('info'), squared_value=squared_value)



file_get= open("../global_normalized_normed_dict_loop_2.pkl","rb")
global_normalized_normed_dict = pickle.load(file_get)
file_get.close()
#matched_dict = global_normalized_normed_dict

file_get= open("../global_mean_dict_loop_2.pkl","rb")
global_mean_dict = pickle.load(file_get)
file_get.close()

file_get= open("../global_sigma_dict_loop_2.pkl","rb")
global_sigma_dict = pickle.load(file_get)
file_get.close()


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)