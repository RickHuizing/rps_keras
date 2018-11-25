import os
from keras.models import load_model


def select_model(manual=False):
    global model
    if manual:
        model_location = os.getcwd() + '\\neuralnetwork\\models\\'
        file_list = os.listdir(model_location)
        select_string = ''
        print('select model to load')
        for index, file in enumerate(file_list):
            print(str(index) + ': ' + file)
        selected_file = input(select_string)
        print(model_location + file_list[int(selected_file)])
        model = load_model(model_location + file_list[int(selected_file)])
    else:
        model_location = os.getcwd() + '\\neuralnetwork\\'
        print(model_location + '99percent.hdf5')
        model = load_model(model_location + '99percent.hdf5')


# best model to date
select_model()


# adds the winning player to the output of get_prediction
def add_winner(prediction_output):
    global model
    p1, p2 = prediction_output
    if p1 == 'q' or p2 == 'q':
        return prediction_output.append('no_winner')
    if p1 == 'r':
        if p2 == 'r':
            return prediction_output.append('draw')
        elif p2 == 'p':
            return prediction_output.append('p2')
        elif p2 == 's':
            return prediction_output.append('p1')

    if p1 == 'p':
        if p2 == 'r':
            return prediction_output.append('p1')
        elif p2 == 'p':
            return prediction_output.append('draw')
        elif p2 == 's':
            return prediction_output.append('p2')

    if p1 == 's':
        if p2 == 'r':
            return prediction_output.append('p2')
        elif p2 == 'p':
            return prediction_output.append('p1')
        elif p2 == 's':
            return prediction_output.append('draw')


# p, q, r, s
def get_prediction(img):
    pred = model.predict(img).tolist()
    pred_dict_list = []
    for file in pred:
        pred_dict_list.append({'r': file[2], 'p': file[0], 's': file[3], 'q': file[1]})
    # print(pred_dict_list)

    highest_values = [0] * len(pred_dict_list)
    predicted_classes = ['q'] * len(pred_dict_list)
    current_img = 0
    for img in pred_dict_list:
        for p_class, p_value in img.items():
            if p_value > highest_values[current_img]:
                highest_values[current_img] = p_value
                predicted_classes[current_img] = p_class
        current_img += 1
    add_winner(predicted_classes)
    return predicted_classes
